"""Remeshing of an initial UV mesh with an iterative triangle-error + edge-length metric.

Edge-length metric
------------------
The first fundamental form [[E,F],[F,G]] scaled by factor = 1/L² so that unit
metric length corresponds to a 3D edge length of L.

Iterative error refinement
--------------------------
After each MMG pass the per-triangle approximation error is computed.  Where
the average triangle error exceeds target_error the factor is multiplied by
avg_error / target_error (exploiting the h² error scaling) and MMG is
run again on the new mesh.  This repeats until the error is within budget or
max_iterations is reached.
"""

import numpy as np
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Tuple
from utils.mesh_utils import compute_VF, create_edge
from operations.align import _get_boundary_edges
from utils.print import print_if_verbose, dump_oversized_patch_state
import utils.config as config
GEOM_TOL = 1e-10
# ---------------------------------------------------------------------------
# Medit .mesh file I/O
# ---------------------------------------------------------------------------

def write_medit_mesh(
    path: str,
    V: np.ndarray,
    F: np.ndarray,
    feature_edges: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    required_vertices: Optional[List[int]] = None,
    required_triangles: Optional[List[int]] = None,
    required_edges: Optional[List[Tuple[int, int]]] = None,
):
    """Write a 2D triangle mesh in Medit .mesh format (1-based indexing).

    feature_edges : {tag: [(v0, v1), ...]} — each tag becomes the Medit
        reference id for that group of edges.
    required_edges : edges written to both Edges and RequiredEdges — MMG
        preserves them exactly (no split / collapse / swap).
    """
    with open(path, "w") as fh:
        fh.write("MeshVersionFormatted 2\n\n")
        fh.write("Dimension 2\n\n")

        nv = V.shape[0]
        fh.write(f"Vertices\n{nv}\n")
        for i in range(nv):
            fh.write(f"{V[i, 0]:.17g} {V[i, 1]:.17g} 0\n")

        nf = F.shape[0]
        fh.write(f"\nTriangles\n{nf}\n")
        for i in range(nf):
            fh.write(f"{F[i, 0] + 1} {F[i, 1] + 1} {F[i, 2] + 1} 0\n")

        all_edges = []
        edge_tags = []
        if feature_edges:
            for tag, edges in feature_edges.items():
                for v0, v1 in edges:
                    all_edges.append((v0, v1))
                    edge_tags.append(tag)
        req_start = len(all_edges)
        if required_edges:
            for v0, v1 in required_edges:
                all_edges.append((v0, v1))
                edge_tags.append(1)

        if all_edges:
            fh.write(f"\nEdges\n{len(all_edges)}\n")
            for (v0, v1), tag in zip(all_edges, edge_tags):
                fh.write(f"{v0} {v1} {tag}\n")

        if required_edges:
            fh.write(f"\nRequiredEdges\n{len(required_edges)}\n")
            for idx in range(req_start + 1, req_start + len(required_edges) + 1):
                fh.write(f"{idx}\n")

        if required_vertices:
            fh.write(f"\nRequiredVertices\n{len(required_vertices)}\n")
            for vid in required_vertices:
                fh.write(f"{vid}\n")  # already 1-based

        if required_triangles:
            fh.write(f"\nRequiredTriangles\n{len(required_triangles)}\n")
            for tid in required_triangles:
                fh.write(f"{tid}\n")  # already 1-based

        fh.write("\nEnd\n")


def write_medit_sol(
    path: str,
    EE: np.ndarray,
    FF: np.ndarray,
    GG: np.ndarray,
    factor: float = 1.0,
):
    """Write a 2D anisotropic tensor metric in Medit .sol format.

    The symmetric 2x2 metric tensor at each vertex has 3 components: m11, m12, m22.
    These correspond to E, F, G of the first fundamental form scaled by ``factor``.

    A tensor that is non-finite, not positive definite, or anisotropic beyond
    1e8:1 (det < 1e-8 * s^2, s = (E + G) / 2; e.g. at a pole or apex) is
    replaced by the isotropic tensor (s, 0, s) of the same scale, or of the
    median scale of the valid tensors when s itself is unusable. The test is
    scale-free, so it does not depend on the model's units or on ``factor``.
    """
    nv = len(EE)
    E = np.array(EE, dtype=float).reshape(-1)
    Fm = np.array(FF, dtype=float).reshape(-1)
    G = np.array(GG, dtype=float).reshape(-1)
    with np.errstate(invalid="ignore", over="ignore"):
        s = 0.5 * (E + G)
        det = E * G - Fm * Fm
        valid = (np.isfinite(E) & np.isfinite(Fm) & np.isfinite(G)
                 & (E > 0) & (G > 0) & (det > 0) & (det >= 1e-8 * s * s))
        bad_scale = ~valid & ~(np.isfinite(s) & (s > 0))
    if bad_scale.any():
        if not valid.any():
            raise ValueError("write_medit_sol: no valid metric tensor to take "
                             "a fallback scale from")
        s[bad_scale] = np.median(s[valid])
    E[~valid], Fm[~valid], G[~valid] = s[~valid], 0.0, s[~valid]
    with open(path, "w") as fh:
        fh.write("MeshVersionFormatted 2\n\n")
        fh.write("Dimension 2\n\n")
        fh.write(f"SolAtVertices\n{nv}\n")
        fh.write("1 3\n")  # 1 solution of type 3 (tensor)
        for i in range(nv):
            e, f, g = float(E[i]), float(Fm[i]), float(G[i])
            fh.write(f"{factor * e:.17g} {factor * f:.17g} {factor * g:.17g}\n")
        fh.write("\nEnd\n\n")


def read_medit_mesh(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[Tuple[int, int]]]]:
    """Read a 2D triangle mesh from Medit .mesh format.

    Returns
    -------
    V : (nv, 2)
    F : (nf, 3), 0-based
    tagged_edges : {tag: [(v0, v1), ...]} — 0-based edge pairs grouped by
        Medit reference id. Empty dict if no Edges block.
    """
    V = None
    F = None
    tagged_edges: Dict[int, List[Tuple[int, int]]] = {}
    with open(path, "r") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "Vertices":
            i += 1
            nv = int(lines[i].strip())
            V = np.zeros((nv, 2))
            for vi in range(nv):
                i += 1
                parts = lines[i].strip().split()
                V[vi, 0] = float(parts[0])
                V[vi, 1] = float(parts[1])
        elif line == "Triangles":
            i += 1
            nf = int(lines[i].strip())
            F = np.zeros((nf, 3), dtype=int)
            for fi in range(nf):
                i += 1
                parts = lines[i].strip().split()
                F[fi, 0] = int(parts[0]) - 1
                F[fi, 1] = int(parts[1]) - 1
                F[fi, 2] = int(parts[2]) - 1
        elif line == "Edges":
            i += 1
            ne = int(lines[i].strip())
            for ei in range(ne):
                i += 1
                parts = lines[i].strip().split()
                v0, v1, tag = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2])
                tagged_edges.setdefault(tag, []).append((v0, v1))
        i += 1

    if V is None or F is None:
        raise RuntimeError(f"Failed to parse mesh from {path}")
    return V, F, tagged_edges


# ---------------------------------------------------------------------------
# Edge meshing: per-segment error (quadrature-based)
# ---------------------------------------------------------------------------

# 5-point Gauss-Legendre quadrature on [0, 1]
# (transformed from standard [-1, 1] nodes and weights)
_EDGE_QUAD_POINTS = np.array([
    0.046910077030668003601186567510974,
    0.230765344947158454481842813728340,
    0.500000000000000000000000000000000,
    0.769234655052841545518157186271660,
    0.953089922969331996398813432489026,
])
_EDGE_QUAD_WEIGHTS = np.array([
    0.118463442528094543757132020359958,
    0.239314335249683234020645757417817,
    0.284444444444444444444444444444444,
    0.239314335249683234020645757417817,
    0.118463442528094543757132020359958,
])  # sum = 1.0


def compute_segment_approximation_errors(
    curve,
    t_values: np.ndarray,
) -> np.ndarray:
    """Compute per-segment approximation error using 5-point Gauss-Legendre quadrature.

    Returns the weighted average distance between the true curve and its
    piecewise-linear approximation, analogous to the triangle quadrature in
    compute_triangle_approximation_errors.

    Parameters
    ----------
    curve : abs curve object with ``.sample(t)`` where t is (n, 1) → (n, 3).
    t_values : (n_verts,) sorted parameter values defining the edge mesh.

    Returns
    -------
    seg_errors : (n_segs,) per-segment approximation errors [3D length units].
    """
    n_segs = len(t_values) - 1
    seg_errors = np.zeros(n_segs)

    P = curve.sample(t_values.reshape(-1, 1))  # (n_verts, 3)

    for si in range(n_segs):
        t0, t1 = t_values[si], t_values[si + 1]
        p0, p1 = P[si], P[si + 1]

        t_samples = t0 + _EDGE_QUAD_POINTS * (t1 - t0)
        p_true = curve.sample(t_samples.reshape(-1, 1))
        p_linear = ((1 - _EDGE_QUAD_POINTS)[:, None] * p0[None, :]
                    + _EDGE_QUAD_POINTS[:, None] * p1[None, :])

        dists = np.linalg.norm(p_true - p_linear, axis=1)
        seg_errors[si] = float(np.dot(_EDGE_QUAD_WEIGHTS, dists))

    return seg_errors


# ---------------------------------------------------------------------------
# Edge meshing: optimal split point via dense sampling
# ---------------------------------------------------------------------------

def find_max_error_parameter(curve, t0, t1, p0, p1):
    """Find the parameter in (t0, t1) where curve-to-chord distance is maximized.

    Placing the new vertex here gives the largest error reduction because the
    point of worst approximation becomes an exact interpolation vertex.

    Uses a dense sample sweep (only curve.sample, no derivatives) to find the
    argmax of the chord-distance function.
    """
    dt = t1 - t0
    n_sweep = 20
    alphas = np.linspace(0.0, 1.0, n_sweep + 2)[1:-1]
    t_batch = t0 + alphas * dt
    p_true = curve.sample(t_batch.reshape(-1, 1))
    p_linear = ((1 - alphas)[:, None] * p0[None, :]
                + alphas[:, None] * p1[None, :])
    dists = np.linalg.norm(p_true - p_linear, axis=1)
    return float(t_batch[np.argmax(dists)])


# ---------------------------------------------------------------------------
# Edge meshing: iterative refinement with optimal splitting
# ---------------------------------------------------------------------------

def refine_edge_mesh(
    curve,
    target_error: float = 5e-2,
    max_iterations: int = 20,
    initial_segments: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively refine a curve discretisation by splitting segments at the
    point of maximum approximation error.

    Uses a quadrature-based error metric and a dense sample sweep to find the
    split locations (see ``find_max_error_parameter``).

    Parameters
    ----------
    curve : abs curve object with ``.sample()``, ``.derivative()``, ``.interval``.
    target_error : max allowed per-segment approximation error [3D length units].
    max_iterations : maximum refinement passes.
    initial_segments : number of uniform segments to start with.

    Returns
    -------
    t_values : (n_verts,) refined parameter values.
    P : (n_verts, 3) 3D positions on the curve.
    """
    t_min = float(curve.interval[0, 0])
    t_max = float(curve.interval[0, 1])

    # Initial uniform discretisation
    t_values = np.linspace(t_min, t_max, initial_segments + 1)

    for _ in range(max_iterations):
        seg_errors = compute_segment_approximation_errors(curve, t_values)

        bad_mask = seg_errors > target_error
        if not np.any(bad_mask):
            break

        # Split bad segments at the point of maximum error
        P = curve.sample(t_values.reshape(-1, 1))
        new_t = []
        for si in range(len(seg_errors)):
            new_t.append(t_values[si])
            if bad_mask[si]:
                t_split = find_max_error_parameter(
                    curve, t_values[si], t_values[si + 1], P[si], P[si + 1]
                )
                new_t.append(t_split)
        new_t.append(t_values[-1])
        t_values = np.array(new_t)

    P = curve.sample(t_values.reshape(-1, 1))
    return t_values, P

# ---------------------------------------------------------------------------
# MMG remeshing
# ---------------------------------------------------------------------------

def mmg_remesh(
    V: np.ndarray,
    F: np.ndarray,
    EE: Optional[np.ndarray] = None,
    FF_m: Optional[np.ndarray] = None,
    GG: Optional[np.ndarray] = None,
    metric_factor: float = 1.0,
    hgrad: float = 1.5,
    mmg2d_path: str = "mmg2d_O3",
    feature_edges: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    required_vertices: Optional[List[int]] = None,
    required_edges: Optional[List[Tuple[int, int]]] = None,
    nosizreq: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[Tuple[int, int]]]]:
    """Run MMG2D remeshing via the CLI.

    Parameters
    ----------
    feature_edges : {tag: [(v0, v1), ...]} — 1-based edge pairs grouped by
        Medit reference tag.
    required_edges : list of (v0, v1) 1-based vertex index pairs frozen in the
        output (no split / collapse / swap).

    Returns
    -------
    V_out : (nv_out, 2) remeshed vertices.
    F_out : (nf_out, 3) remeshed triangles (0-based).
    tagged_edges : {tag: [(v0, v1), ...]} — 0-based output edges grouped by
        Medit reference tag.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        mesh_in = os.path.join(tmpdir, "input.mesh")
        sol_in = os.path.join(tmpdir, "input.sol")
        mesh_out = os.path.join(tmpdir, "output.mesh")

        write_medit_mesh(mesh_in, V, F,
                         feature_edges=feature_edges,
                         required_vertices=required_vertices,
                         required_edges=required_edges)
        write_medit_sol(sol_in, EE, FF_m, GG, metric_factor)

        cmd = [
            mmg2d_path,
            mesh_in,
            "-sol", sol_in,
            "-out", mesh_out,
            "-v", "-1"
        ]
        if hgrad != 0:
            cmd += ["-hgrad", str(hgrad)]
        if nosizreq:
            cmd += ["-nosizreq"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=config.mmg_timeout_seconds)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"mmg2d timed out after {config.mmg_timeout_seconds} s") from None
        if result.returncode != 0:
            raise RuntimeError(
                f"mmg2d failed (return code {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        V_out, F_out, tagged_edges_out = read_medit_mesh(mesh_out)
        if len(F_out) == 0:
            raise ValueError(f"mmg2d returned no triangles ({len(V_out)} vertices)")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Monitoring only: dump oversized state for later investigation, never
    # interfere with the run.
    dump_oversized_patch_state(V_out, F_out, tag="mmg_remesh")

    return V_out, F_out, tagged_edges_out


# ---------------------------------------------------------------------------
# Post-MMG singularity fan stitching
# ---------------------------------------------------------------------------

def stitch_singularity_fans_3d(
    V_3d: list,
    F_3d: list,
    face,
    sing_edges
) -> Tuple[list, list, dict]:
    """Create fan triangles connecting singularity strip boundaries to apex vertices.

    For each singularity, adds an apex vertex at its 3D point and builds fan
    triangles from that singularity's strip-boundary edges to the apex.

    Parameters
    ----------
    sing_edges : dict mapping singularity index → list of (a, b) strip-boundary
        edges (vertex ids into V_3d / F_3d).

    Returns
    -------
    V_3d, F_3d : input lists extended in-place.
    sing_info : dict mapping singularity index → ``{"center_vertex_id": int}``.
    """
    sing_info = {}
    if face.singularities is None:
        return V_3d, F_3d, sing_info
    VF = compute_VF(F_3d)
    # Fan only edges that are still open (one triangle) in the stitched mesh,
    # each once. sing_edges is selected on the unstitched uv mesh, where the
    # seam sides are open too; a seam edge that slipped in becomes interior
    # after the seam stitch, and both of its uv copies map to the same 3D
    # edge, so fanning it would give that edge 4 triangles.
    open_edges = _get_boundary_edges(F_3d)
    fanned = set()

    # Per singularity: insert 3D center vertex, create fan triangles.
    for idx, edges in sing_edges.items():
        if not edges:
            continue
        pos_3d = np.asarray(face.singularities[idx]['point3d'], dtype=float)
        cv = len(V_3d)
        V_3d.append(pos_3d.tolist())
        sing_info[idx] = {"center_vertex_id": cv}

        for a, b in edges:
            e = create_edge(a, b)
            if e not in open_edges or e in fanned:
                continue
            fanned.add(e)
            common_faces = set(VF.get(a, [])) & set(VF.get(b, []))
            if not common_faces:
                continue
            fi = next(iter(common_faces))
            tri = F_3d[fi]
            for j in range(3):
                k = (j + 1) % 3
                if {tri[j], tri[k]} == {a, b}:
                    F_3d.append([tri[k], tri[j], cv])
                    break
    return V_3d, F_3d, sing_info


# ---------------------------------------------------------------------------
# First fundamental form
# ---------------------------------------------------------------------------

def compute_first_fundamental_form(
    surface,
    V_uv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the first fundamental form (E, F, G) at each UV vertex.

    Uses ``surface.derivative(points, order)`` from the ``abs`` module.

    Parameters
    ----------
    surface : abs surface object with ``.derivative(uv, order)``.
    V_uv : (nv, 2) vertex UV coordinates.

    Returns
    -------
    EE, FF, GG : (nv,) first fundamental form coefficients.
    """
    duv = surface.derivative(V_uv, order=1)  

    du = duv[:, :, 0]  # (nv, 3)
    dv = duv[:, :, 1]  # (nv, 3)

    EE = np.sum(du * du, axis=1)  # E = du . du
    FF = np.sum(du * dv, axis=1)  # F = du . dv
    GG = np.sum(dv * dv, axis=1)  # G = dv . dv

    return EE, FF, GG


# Fraction of the minimum principal-curvature radius allowed as approximation
# error. The discretization (mesh triangles, edge chords) must deviate from the
# true surface by strictly less than R_min = 1/|kappa|_max — the surface's
# local reach — or closest-point projection becomes many-to-one and distinct
# nearby features alias onto the same region (e.g. on a torus band).
# 0.5 (half the reach) prevents the aliasing catastrophe but still allows
# 120-degree arc elements — too lax for stable closest-point/trace
# localization: on a cylinder face (r=1.55) it gave 3 elements per
# circumference and element-ridge projection ambiguity broke the embedded
# feature chains (0.1 gave 18 triangles and zero interior breaks there).
# 0.2 (sag <= 20% of R, ~5 elements per full circle) is the working
# compromise between projection stability and coarse-mesh density.
CURVATURE_CAP_BETA = 0.2


def max_principal_curvature(surface, uv_points: np.ndarray) -> Optional[np.ndarray]:
    """|kappa|_max at each UV point from the kernel's exact derivatives.

    kappa_1 * kappa_2 = K = det(II)/det(I), kappa_1 + kappa_2 = 2H =
    (E*N + G*L - 2*F*M)/det(I), so |kappa|_max = |H| + sqrt(H^2 - K).
    Parametrically degenerate points (|s_u x s_v| ~ 0) report 0 curvature.
    """
    uv = np.asarray(uv_points, dtype=float)
    d1 = surface.derivative(uv, order=1)   # (n, 3, 2)
    # Temporary guard until abs is fixed: returns None when the kernel has no
    # second derivative. scipy raises ValueError for B-splines whose interior
    # knots repeat up to the degree (a kink, e.g. the extrusion of a cubic
    # profile with an interior knot repeated 3 times), and abs's Offset returns None.
    try:
        d2 = surface.derivative(uv, order=2)   # (n, 3, 2, 2)
    except ValueError as e:
        print_if_verbose(f"  no second derivative for {type(surface).__name__}: {e}")
        return None
    if d2 is None:
        return None
    su, sv = d1[:, :, 0], d1[:, :, 1]
    E = np.sum(su * su, axis=1)
    F = np.sum(su * sv, axis=1)
    G = np.sum(sv * sv, axis=1)
    n = np.cross(su, sv)
    nn = np.linalg.norm(n, axis=1)
    det_I = E * G - F * F
    # Degeneracy tests — a sample only votes on curvature if its tangent pair
    # is well conditioned. Two independent failure modes:
    #  * collapsed tangent magnitude (parametric pole: a whole domain edge
    #    maps to one 3D point, |s_v| ~ 1e-15). Real
    #    anisotropy is ~20:1 here; 1e-8 relative is far below any real ratio.
    #  * collapsed tangent angle: det(I)/(E*G) = sin^2(angle).
    # Rejected samples report kappa = 0 ("no opinion"); R_min comes from the
    # healthy samples.
    su_n, sv_n = np.sqrt(E), np.sqrt(G)
    scale = max(float(su_n.max()), float(sv_n.max()), 1e-300)
    ok = ((su_n > 1e-8 * scale) &
          (sv_n > 1e-8 * scale) &
          (det_I > 1e-12 * np.maximum(E * G, 1e-300)))
    n = n / np.maximum(nn, 1e-300)[:, None]
    L = np.sum(d2[:, :, 0, 0] * n, axis=1)
    M = np.sum(d2[:, :, 0, 1] * n, axis=1)
    N = np.sum(d2[:, :, 1, 1] * n, axis=1)
    det_I = np.maximum(det_I, 1e-300)
    K = (L * N - M * M) / det_I
    H = (E * N + G * L - 2.0 * F * M) / (2.0 * det_I)
    kmax = np.abs(H) + np.sqrt(np.maximum(H * H - K, 0.0))
    kmax[~ok] = 0.0
    return kmax


def min_curvature_radius(surface) -> float:
    """Minimum principal-curvature radius over the surface's trim domain.

    Closed form from the surface parameters for the primitive types; the
    exact-derivative evaluator (sampled over the trim window) for the rest.
    Returns inf for flat surfaces.
    """
    dom = np.asarray(surface.trim_domain, dtype=float)  # [[u0, u1], [v0, v1]]
    stype = type(surface).__name__
    if stype == "Plane":
        return np.inf
    if stype == "Offset":
        # Temporary: abs gets Offset derivatives wrong (order 1 is incorrect,
        # order 2 returns None), so skip the curvature cap for offset faces
        # until abs is fixed.
        return np.inf
    if stype in ("Cylinder", "Sphere"):
        return float(surface.radius)
    if stype == "Torus":
        # kappa_tube = 1/r; kappa_azimuthal(v) = cos(v) / (R + r*cos(v)),
        # extremal at the v-window endpoints or at the inner equator v=pi.
        R, r = float(surface.max_radius), float(surface.min_radius)
        v0, v1 = dom[1]
        cand = [v0, v1, 0.0]
        for vpi in (-np.pi, np.pi):
            if v0 - 1e-12 <= vpi <= v1 + 1e-12:
                cand.append(vpi)
        k_az = max(abs(np.cos(v)) / max(abs(R + r * np.cos(v)), 1e-300)
                   for v in cand)
        return 1.0 / max(1.0 / max(r, 1e-300), k_az)
    # Cone: curvature varies only along the ruling -> a line of samples is
    # exact enough; BSpline & others: dense grid. Both use exact derivatives.
    n_u, n_v = (2, 33) if stype == "Cone" else (24, 24)
    u = np.linspace(dom[0, 0], dom[0, 1], n_u)
    v = np.linspace(dom[1, 0], dom[1, 1], n_v)
    UU, VV = np.meshgrid(u, v)
    kmax = max_principal_curvature(surface, np.column_stack([UU.ravel(), VV.ravel()]))
    if kmax is None:
        # No second derivative from the kernel: skip the curvature cap.
        return np.inf
    k = float(kmax.max())
    return np.inf if k <= 1e-30 else 1.0 / k


def curvature_capped_error(surface, target_error: float,
                           beta: float = CURVATURE_CAP_BETA) -> float:
    """min(target_error, beta * R_min): the effective approximation-error
    target that keeps the discretization inside the surface's reach.

    Floored at target_error/256 so genuinely singular geometry (a cusp,
    R_min ~ 0) bounds the extra refinement work (~16x more points) instead
    of chasing an impossible tolerance.
    """
    r_min = min_curvature_radius(surface)
    if not np.isfinite(r_min):
        return target_error
    return max(min(target_error, beta * r_min), target_error / 256.0)


def get_sampled_3d_arc_length(surface, uv_start: np.ndarray, uv_end: np.ndarray,
                               num_samples: int = 100) -> float:
    """Compute 3D arc length along a UV-axis-aligned path by sampling.

    Matches C++ ``get_sampled_axis_align_patch_length``.
    """
    length = 0.0
    p_prev = surface.sample(uv_start.reshape(1, 2))[0]
    for i in range(1, num_samples):
        t = i / (num_samples - 1)
        uv = (1 - t) * uv_start + t * uv_end
        p_cur = surface.sample(uv.reshape(1, 2))[0]
        length += float(np.linalg.norm(p_cur - p_prev))
        p_prev = p_cur
    return length


def compute_mmg_factor(surface, V_uv: np.ndarray) -> float:
    """Compute the metric scaling factor exactly as in C++.

    ``factor = min(100, uv_ratio) / bnd_diag``

    where ``uv_ratio`` is the ratio of the longer to shorter 3D edge length
    of the UV domain, and ``bnd_diag`` is the 3D bounding box diagonal.
    """
    trim_domain = surface.trim_domain
    u_lb, u_ub = float(trim_domain[0, 0]), float(trim_domain[0, 1])
    v_lb, v_ub = float(trim_domain[1, 0]), float(trim_domain[1, 1])

    # 3D arc lengths along domain edges (same as C++ uv_edge_length_to_init)
    u_length = get_sampled_3d_arc_length(
        surface, np.array([u_lb, v_lb]), np.array([u_ub, v_lb]))
    v_length = get_sampled_3d_arc_length(
        surface, np.array([u_lb, v_lb]), np.array([u_lb, v_ub]))

    if u_length < 1e-8:
        u_length = get_sampled_3d_arc_length(
            surface, np.array([u_lb, v_ub]), np.array([u_ub, v_ub]))
    elif v_length < 1e-8:
        v_length = get_sampled_3d_arc_length(
            surface, np.array([u_ub, v_lb]), np.array([u_ub, v_ub]))

    # 3D bounding box diagonal
    V_3d = surface.sample(V_uv)
    bnd_diag = float(np.linalg.norm(V_3d.max(axis=0) - V_3d.min(axis=0)))

    # UV aspect ratio (always >= 1)
    if u_length > v_length and v_length > 1e-8:
        uv_ratio = u_length / v_length
    elif v_length > u_length and u_length > 1e-8:
        uv_ratio = v_length / u_length
    else:
        uv_ratio = 1.0

    factor = 0.1 if bnd_diag < 1e-5 else min(100.0, uv_ratio) / bnd_diag
    return factor


# ---------------------------------------------------------------------------
# WV(5) 7-point degree-5 triangle quadrature
# (TriangleQuadrature.cpp, case 5 – raw weights sum to 1.0)
# ---------------------------------------------------------------------------
_QD5_U = np.array([
    0.333333333333333314829616256247390992939472198486328125,
    0.1012865073234563428883348024100996553897857666015625,
    0.470142064105115109473587153843254782259464263916015625,
    0.797426985353087314223330395179800689220428466796875,
    0.05971587178976978105282569231349043548107147216796875,
    0.1012865073234563428883348024100996553897857666015625,
    0.470142064105115109473587153843254782259464263916015625,
])
_QD5_V = np.array([
    0.333333333333333314829616256247390992939472198486328125,
    0.797426985353087314223330395179800689220428466796875,
    0.05971587178976978105282569231349043548107147216796875,
    0.1012865073234563428883348024100996553897857666015625,
    0.470142064105115109473587153843254782259464263916015625,
    0.1012865073234563428883348024100996553897857666015625,
    0.470142064105115109473587153843254782259464263916015625,
])
# Barycentric coords: (lambda0, lambda1, lambda2) = (1-u-v, u, v)
_TRI_QUAD_BARY = np.column_stack([1.0 - _QD5_U - _QD5_V, _QD5_U, _QD5_V])  # (7,3)
_TRI_QUAD_WEIGHTS = np.array([
    0.2250000000000000055511151231257827021181583404541015625,
    0.1259391805448271395295734009778243489563465118408203125,
    0.1323941527885061919533882246469147503376007080078125,
    0.1259391805448271395295734009778243489563465118408203125,
    0.1323941527885061919533882246469147503376007080078125,
    0.1259391805448271395295734009778243489563465118408203125,
    0.1323941527885061919533882246469147503376007080078125,
])  # sum ≈ 1.0


# ---------------------------------------------------------------------------
# Triangle approximation error
# ---------------------------------------------------------------------------

def compute_triangle_approximation_errors(
    surface,
    V_uv: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """Port of PatchMeshDistMax::max_distance across all triangles.

    For each triangle computes the quadrature-weighted mean 3D distance between
    the true surface and the linearly interpolated mesh triangle.

    Parameters
    ----------
    surface : surface object with ``.sample(uv)`` → (n, 3).
    V_uv : (nv, 2) UV vertex positions.
    F : (nf, 3) triangle connectivity (0-based).

    Returns
    -------
    tri_errors : (nf,) per-triangle approximation errors [3D length units].
    """
    nf = F.shape[0]
    tri_errors = np.zeros(nf)
    p3d = surface.sample(V_uv)  # (nv, 3) – batch evaluate all vertices

    for fi in range(nf):
        i0, i1, i2 = F[fi]
        uv0, uv1, uv2 = V_uv[i0], V_uv[i1], V_uv[i2]

        # Degenerate UV triangles raise (test mirrors BarycentricTriangle::is_degenerate)
        e1 = uv1 - uv0
        e2 = uv2 - uv0
        if abs(e1[0] * e2[1] - e1[1] * e2[0]) < 1e-15:
            print_if_verbose(f"Warning: degenerate UV triangle at index {fi} with vertices {uv0}, {uv1}, {uv2}")
            raise Exception(f"COMPUTE TRIANGLE ERROR: Degenerate UV triangle at index {fi} with vertices {i0}, {i1}, {i2}")

        # Quadrature points in actual UV space via barycentric interpolation
        uv_q = (_TRI_QUAD_BARY[:, 0:1] * uv0
              + _TRI_QUAD_BARY[:, 1:2] * uv1
              + _TRI_QUAD_BARY[:, 2:3] * uv2)   # (7, 2)

        p_true = surface.sample(uv_q)            # (7, 3) true surface positions
        p_lin  = (_TRI_QUAD_BARY[:, 0:1] * p3d[i0]
               + _TRI_QUAD_BARY[:, 1:2] * p3d[i1]
               + _TRI_QUAD_BARY[:, 2:3] * p3d[i2])  # (7, 3) linear interp

        dists = np.linalg.norm(p_true - p_lin, axis=1)  # (7,)
        tri_errors[fi] = float(np.dot(_TRI_QUAD_WEIGHTS, dists))

    return tri_errors


# ---------------------------------------------------------------------------
# Edge-length metric
# ---------------------------------------------------------------------------

def compute_edge_length_metric(V, F, surface):
    """3D chord length of each triangle edge (UV vertices lifted by ``surface``).

    Returns
    -------
    edge_lengths : (nf, 3) 3D lengths of the edges i0→i1, i1→i2, i2→i0 of each triangle.
    """
    nf = F.shape[0]
    edge_lengths = np.zeros((nf, 3))
    V3d= surface.sample(V)  # (nv, 3) – batch evaluate all vertices
    for fi in range(nf):
        i0, i1, i2 = F[fi]
        p0, p1, p2 = V3d[i0], V3d[i1], V3d[i2]

        edge_lengths[fi, 0] = np.linalg.norm(p1 - p0)  # edge 0: i0 → i1
        edge_lengths[fi, 1] = np.linalg.norm(p2 - p1)  # edge 1: i1 → i2
        edge_lengths[fi, 2] = np.linalg.norm(p0 - p2)  # edge 2: i2 → i0

    return edge_lengths

# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def remesh_with_error_and_edge_metric(
    surface,
    V_uv: np.ndarray,
    F: np.ndarray,
    target_error: Optional[float] = 5e-2,
    hgrad: float = -1,
    mmg2d_path: str = "mmg2d_O3",
    max_iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Remesh a UV mesh using an anisotropic tensor metric from the first
    fundamental form, with iterative factor refinement.

    The metric ``M = factor * [[E, F], [F, G]]`` makes MMG create
    unit-metric-length edges, so 3D edge length ~ ``1 / sqrt(factor)``.
    The factor is iteratively increased when the triangle approximation
    error exceeds ``target_error`` (exploiting h^2 error scaling).

    Parameters
    ----------
    surface : surface object with ``.sample()`` and ``.derivative()``.
    V_uv : (nv, 2) UV vertex positions of the initial mesh.
    F : (nf, 3) triangle connectivity (0-based).
    target_error : if given, iterate until avg triangle approximation error
        is within this threshold.
    hgrad : MMG gradation parameter.
    mmg2d_path : path to the mmg2d executable.
    max_iterations : maximum number of MMG passes.

    Returns
    -------
    V_out : (nv_out, 2) remeshed UV vertices.
    F_out : (nf_out, 3) remeshed triangles (0-based).
    factor_out : tensor scaling factor of the MMG pass that produced
        V_out, F_out.
    """
    # Stagnation thresholds (rationale, not tunables):
    #   STAGNATION_RATIO = 0.9 — an iteration that fails to reduce avg_error
    #     by at least 10% counts as no progress.  Wider than the ~1.6% noise
    #     observed when mmg can't follow the requested factor (on one surface
    #     avg_error stayed in [2.808, 2.855] across 9 iterations while factor
    #     scaled by ~15^9).
    #   STAGNATION_LIMIT = 5 — one stagnant iteration could be a coincidence
    #     (mmg producing a slightly different connectivity at the same
    #     effective h); five in a row signals a structural inability to
    #     refine, so further factor scaling is wasted work.
    STAGNATION_RATIO = 0.9
    STAGNATION_LIMIT = 5

    factor = compute_mmg_factor(surface, V_uv)
    mesh_factor = factor  # factor of the mmg pass that produced V_cur, F_cur
    V_cur, F_cur = V_uv, F
    prev_avg_error: Optional[float] = None
    stagnation_count = 0

    # mmg RuntimeError, mmg_remesh's ValueError on an empty result and
    # degenerate-UV-triangle exceptions from
    # compute_triangle_approximation_errors propagate to the caller;
    # run_initial_meshing_for_face's outer try/except routes them to the
    # insert_feature_edges_uv fallback. Stagnation is not an error: the
    # loop stops and returns the current mesh and factor.
    for it in range(max_iterations):
        EE, FF_m, GG = compute_first_fundamental_form(surface, V_cur)
        print_if_verbose(f"\n ---Initial remesh Iteration {it} ---")
        mesh_factor = factor
        V_cur, F_cur, _ = mmg_remesh(
            V_cur, F_cur,
            EE=EE, FF_m=FF_m, GG=GG,
            metric_factor=factor,
            hgrad=hgrad,
            mmg2d_path=mmg2d_path,
        )
        if target_error is None:
            break

        tri_errors = compute_triangle_approximation_errors(surface, V_cur, F_cur)
        avg_error = float(np.average(tri_errors))
        edge_lengths = compute_edge_length_metric(V_cur, F_cur, surface)
        max_edge_length = float(np.max(edge_lengths))
        print_if_verbose(f"Iteration {it}: avg triangle error = {avg_error:.3e}, "
                         f"max edge length = {max_edge_length:.3e}")

        if prev_avg_error is not None and avg_error >= STAGNATION_RATIO * prev_avg_error:
            stagnation_count += 1
            if stagnation_count >= STAGNATION_LIMIT:
                return V_cur, F_cur, mesh_factor
        else:
            stagnation_count = 0
        prev_avg_error = avg_error

        if avg_error > target_error:
            # Too coarse: scale up using h² relationship (error ∝ h² ∝ 1/factor)
            ratio = avg_error / target_error
            factor *= ratio
            print_if_verbose(f"  error {avg_error:.3e} > target {target_error:.3e}, "
                             f"scaling factor up by {ratio:.2f}x -> factor={factor:.3e}")
        elif avg_error == 0.0:
            # Flat patch (plane): the error carries no size signal, and
            # scaling by ratio = 0 would give factor = 0 and factor/ratio = NaN.
            break
        elif avg_error < target_error * 0.1:
            # Over-refined: scale down to approach target_error from above
            ratio = avg_error / target_error
            factor *= ratio
            print_if_verbose(f"  error {avg_error:.3e} << target {target_error:.3e}, "
                             f"scaling factor down by {ratio:.2e}x -> factor={factor:.3e}")
            if factor < 1e-8:
                return V_cur, F_cur, mesh_factor
        else:
            break

    return V_cur, F_cur, mesh_factor


# ---------------------------------------------------------------------------
# Feature edge projection onto UV mesh
# ---------------------------------------------------------------------------

def insert_feature_edges_uv(
    V_uv: np.ndarray,
    F: np.ndarray,
    face,
    edge_geometry_data: dict,
    target_3d_length: float = 5e-2,
    mmg2d_path: str = "mmg2d_O3",
    max_iterations: int = 5,
    base_factor: Optional[float] = None,
    local_finer_ratio: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine the UV mesh near feature edges (two MMG passes).

    Collects the 3D feature-edge points (polyline vertices plus one
    interpolated midpoint per segment), then:

    1. remeshes the whole patch uniformly at a 3D edge length equal to the
       median feature-point spacing;
    2. if several feature points project (via nearest face centroid) onto the
       same triangle, remeshes once more with the metric scaled up around
       those triangles by their point count.

    Parameters
    ----------
    V_uv : (nv, 2) UV vertex positions from remeshing.
    F : (nf, 3) triangle connectivity from remeshing (0-based).
    face : abs Face object with ``.loops``, ``.surface``.
    edge_geometry_data : dict mapping edge_id -> (n, 3) refined 3D polyline.
    mmg2d_path : path to the mmg2d executable.
    target_3d_length, max_iterations, base_factor, local_finer_ratio : unused;
        the baseline metric always comes from the feature-point spacing.

    Returns
    -------
    V_uv_out : (nv_out, 2) refined UV vertices.
    F_out : (nf_out, 3) refined triangle connectivity.
    """
    from scipy.spatial import KDTree

    surface = face.surface

    # ------------------------------------------------------------------
    # 1. Collect all 3D feature edge points (with hallucinated intermediates)
    # ------------------------------------------------------------------
    n_hallucinate = 1
    all_edge_pts = []
    edge_pt_spacings = []
    for loop in face.loops:
        for halfedge in loop.halfedges:
            edge_id = halfedge.edge.id
            if edge_id not in edge_geometry_data:
                raise ValueError(f"Missing geometry data for edge {edge_id}")
            P = edge_geometry_data[edge_id]  # (n, 3)
            if P.shape[0] < 2:
                continue
            for i in range(len(P) - 1):
                seg_len = float(np.linalg.norm(P[i + 1] - P[i]))
                all_edge_pts.append(P[i])
                if seg_len < GEOM_TOL:
                    continue  # degenerate segment, skip hallucination
                spacing = seg_len / (n_hallucinate + 1)
                edge_pt_spacings.append(spacing)
                for j in range(n_hallucinate):
                    t = (j + 1) / (n_hallucinate + 1)
                    all_edge_pts.append((1 - t) * P[i] + t * P[i + 1])
            all_edge_pts.append(P[-1])

    if not all_edge_pts:
        return V_uv, F

    edge_pts_3d = np.array(all_edge_pts)  # (N, 3)

    # Globally deduplicate: remove points that are too close to an already-kept point.
    if len(edge_pts_3d) > 1:
        min_spacing = 1e-6
        dedup_radius = min_spacing 
        tree = KDTree(edge_pts_3d)
        kept = np.ones(len(edge_pts_3d), dtype=bool)
        for i in range(len(edge_pts_3d)):
            if not kept[i]:
                continue
            neighbors = tree.query_ball_point(edge_pts_3d[i], dedup_radius)
            for j in neighbors:
                if j != i:
                    kept[j] = False
        edge_pts_3d = edge_pts_3d[kept]

    # ------------------------------------------------------------------
    # 2. Iterative refinement
    # ------------------------------------------------------------------
    V_cur, F_cur = V_uv, F
    # Always anchor the baseline metric to the feature-point spacing; ignore any
    # passed-in base_factor (compute_mmg_factor normalizes by bnd_diag and can
    # collapse to ~1e-8 on large/flat patches -> target edge ~1/sqrt(factor)
    # huge -> MMG coarsens instead of refining).  Since the FFF converts
    # UV->3D length, target 3D edge = 1/sqrt(factor), so factor = 1/L_target^2.
    # Use half the median 3D feature-point spacing so consecutive points land
    # in distinct triangles.  Median (not mean) is robust to the highly
    # heterogeneous spacings.
    if not edge_pt_spacings:
        raise ValueError(
            "insert_feature_edges_uv: every feature-edge segment of the face is "
            f"shorter than GEOM_TOL = {GEOM_TOL:g}, so there is no feature "
            "spacing to size the mesh from")
    L_target = float(np.median(edge_pt_spacings))
    base_factor = 1.0 / (L_target ** 2)
    # ------------------------------------------------------------------
    # Pass 1: uniform remesh at the feature-spacing baseline.  This brings the
    # whole patch to ~median feature spacing so that the per-face point count
    # below is a meaningful *local* signal (on the original coarse mesh every
    # point piles onto the same 1-2 faces and the count is useless).
    # ------------------------------------------------------------------
    EE, FF_m, GG = compute_first_fundamental_form(surface, V_cur)
    V_cur, F_cur, _ = mmg_remesh(
        V_cur, F_cur,
        EE=EE, FF_m=FF_m, GG=GG,
        metric_factor=base_factor,
        hgrad=1.5,
        mmg2d_path=mmg2d_path,
    )

    # ------------------------------------------------------------------
    # Pass 2: local refinement where several feature points share one triangle.
    # ------------------------------------------------------------------
    V_3d = surface.sample(V_cur)
    nv = V_cur.shape[0]
    nf = F_cur.shape[0]
    F_np = np.asarray(F_cur)

    # Project each edge point onto its single nearest face (via centroids).
    face_centroids = V_3d[F_np].mean(axis=1)  # (nf, 3)
    centroid_tree = KDTree(face_centroids)
    _, proj_face_ids = centroid_tree.query(edge_pts_3d, k=1)  # (N,)
    proj_count = np.bincount(proj_face_ids, minlength=nf)     # points per face

    if int(proj_count.max()) <= 1:
        return V_cur, F_cur

    # local_scale[v] = max points-per-face over faces incident to v.  Scaling
    # the FFF by k splits a k-point face into ~k sub-triangles because node
    # density scales with sqrt(det(M)) ~ scale, so one point per triangle.
    local_scale = np.ones(nv)
    for fi in range(nf):
        if proj_count[fi] > 1:
            for v in F_np[fi]:
                local_scale[v] = max(local_scale[v], proj_count[fi])

    EE, FF_m, GG = compute_first_fundamental_form(surface, V_cur)
    V_cur, F_cur, _ = mmg_remesh(
        V_cur, F_cur,
        EE=EE * local_scale, FF_m=FF_m * local_scale, GG=GG * local_scale,
        metric_factor=base_factor,
        hgrad=1.5,
        mmg2d_path=mmg2d_path,
    )

    return V_cur, F_cur


# ---------------------------------------------------------------------------
# Conflict-driven background refinement
# ---------------------------------------------------------------------------
# Refine the background mesh only where distinct pieces of feature curve
# compete for the same triangles, so the later tracing can give each feature
# its own element strip. "Distinct pieces" is decided by arc distance along
# the loop, not by edge id: a U-shaped edge whose prongs come close needs
# refinement just as much as two separate edges do, while a single curve
# passing through a triangle (or two edges meeting at their shared corner)
# must never trigger it — no refinement can separate a curve from itself.

CONFLICT_BETA = 3.0            # same-piece window: arc_dist <= beta * h_face
CONFLICT_MAX_HALVINGS = 5      # bounded work: h never forced below h0 / 2^5
SEAM_SIDE_TOL_REL = 1e-6       # uv vertex on a periodic side: within this fraction of the period
CONFLICT_MAX_SAMPLES = 2_000_000


def per_vertex_edge_length_3d(V3d: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Mean 3D length of the edges incident to each vertex."""
    nv = len(V3d)
    h_sum = np.zeros(nv)
    h_cnt = np.zeros(nv)
    for i in range(3):
        a = F[:, i]
        b = F[:, (i + 1) % 3]
        L = np.linalg.norm(V3d[a] - V3d[b], axis=1)
        np.add.at(h_sum, a, L)
        np.add.at(h_cnt, a, 1)
        np.add.at(h_sum, b, L)
        np.add.at(h_cnt, b, 1)
    h_v = h_sum / np.maximum(h_cnt, 1)
    pos = h_v > GEOM_TOL
    if not np.all(pos):
        h_v[~pos] = np.median(h_v[pos]) if np.any(pos) else 1.0
    return h_v


def _build_loop_chains(face, edge_geometry_data):
    """Concatenate each loop's halfedge polylines into one closed 3D chain.

    Returns a list of (segments, total_length) where segments is a list of
    (a, b, s_start, seg_length) in loop order and s is cumulative arc length.
    A seam edge appearing twice in the same loop is sampled only once (it is
    only traced once); its second occurrence still advances s as a phantom
    span so arc positions stay continuous around the loop.

    Also returns, per chain, the glue sets: for every B-rep vertex the loop
    visits more than once, the arc positions of those visits. The loop is
    one curve there (e.g. both ends of a seam edge), so runs of samples
    ending at two visits of the same vertex are one piece, not a conflict.
    """
    chains = []
    chain_glues = []
    for loop in face.loops:
        seen_ids = set()
        segments = []
        vertex_visits = {}   # B-rep vertex id -> arc positions of its visits
        s = 0.0
        for he in loop.halfedges:
            eid = he.edge.id
            start_vid = he.edge.start_vertex if he.orientation_wrt_edge else he.edge.end_vertex
            vertex_visits.setdefault(int(start_vid), []).append(s)
            P = edge_geometry_data.get(eid)
            if P is None or len(P) < 2:
                continue
            P = np.asarray(P, dtype=float)
            if not he.orientation_wrt_edge:
                P = P[::-1]
            seg_lens = np.linalg.norm(np.diff(P, axis=0), axis=1)
            if eid in seen_ids:
                s += float(seg_lens.sum())
                continue
            seen_ids.add(eid)
            for a, b, L in zip(P[:-1], P[1:], seg_lens):
                if L <= GEOM_TOL:
                    continue
                segments.append((a, b, s, float(L)))
                s += float(L)
        if segments:
            chains.append((segments, s))
            chain_glues.append([visits for visits in vertex_visits.values()
                                if len(visits) > 1])
    return chains, chain_glues


def _sample_chains(chains, centroid_tree, h_face):
    """Sample every chain at ~half the local element size.

    Density must track the current mesh: samples coarser than h can miss a
    face entirely and hide a conflict. Returns (points (n,3), arc positions
    (n,), chain ids (n,)).
    """

    pts, arcs, cids = [], [], []
    total = 0
    for ci, (segments, _L) in enumerate(chains):
        for a, b, s0, L in segments:
            mid = 0.5 * (a + b)
            _, fi = centroid_tree.query(mid)
            step = max(0.5 * float(h_face[fi]), 1e-12)
            n = int(np.ceil(L / step))
            n = max(1, min(n, 10_000))
            t = (np.arange(n) + 0.5) / n
            pts.append(a + t[:, None] * (b - a))
            arcs.append(s0 + t * L)
            cids.append(np.full(n, ci, dtype=np.int32))
            total += n
    pts = np.vstack(pts)
    arcs = np.concatenate(arcs)
    cids = np.concatenate(cids)
    if total > CONFLICT_MAX_SAMPLES:
        stride = int(np.ceil(total / CONFLICT_MAX_SAMPLES))
        pts, arcs, cids = pts[::stride], arcs[::stride], cids[::stride]
    return pts, arcs, cids


def _conflict_in_candidate_set(s_vals, c_vals, chain_lens, tol, glues=None):
    """True if the samples contain >= 2 distinct pieces of feature curve.

    Distinct chains always conflict. Within one closed chain, sort the arc
    positions and count circular gaps larger than tol: 1 gap = one contiguous
    run (a single piece passing through — fine); >= 2 gaps = the curve left and
    came back (U-prong, facing teeth); 0 gaps = the whole closed loop fits in
    this neighborhood (a hole swallowed by one coarse triangle) — also a
    conflict, since tracing the loop needs elements of its own.

    glues (from _build_loop_chains): per chain, the arc positions of each
    B-rep vertex visited more than once. The runs are glued at those
    positions before counting: two runs that end at two visits of the same
    vertex are the curve meeting itself there (seam corners), which no
    refinement can separate.
    """
    if np.unique(c_vals).size > 1:
        return True
    ci = int(c_vals[0])
    L = chain_lens[ci]
    if s_vals.size < 2:
        return False
    s = np.sort(s_vals)
    gaps = np.flatnonzero(np.diff(s) > tol)
    wraps = L - (s[-1] - s[0]) <= tol   # first and last run join across s = 0
    big = gaps.size + (0 if wraps else 1)
    if big == 0:
        return True
    if big == 1:
        return False
    glue = glues[ci] if glues is not None else []
    if not glue:
        return True

    # Runs as arc intervals; with wrap-around the last and first run are one.
    intervals = list(zip(np.concatenate(([s[0]], s[gaps + 1])),
                         np.concatenate((s[gaps], [s[-1]]))))
    run_of = list(range(len(intervals)))
    if wraps:
        run_of[-1] = 0
    parent = list(range(len(intervals)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def runs_touching(p):
        return {run_of[k] for k, (a, b) in enumerate(intervals)
                if any(a - tol <= q <= b + tol for q in (p - L, p, p + L))}

    for visits in glue:
        touching = set()
        for p in visits:
            touching |= runs_touching(p)
        touching = sorted(touching)
        for r in touching[1:]:
            parent[find(r)] = find(touching[0])
    return len({find(r) for r in set(run_of)}) >= 2


def _point_triangle_sqr_dist_batch(P, A, B, C):
    """Squared distance from each P[i] to triangle (A[i], B[i], C[i]).

    Vectorized form of internal_operations.point_triangle_sqr_dist (same
    Voronoi-region cases as igl's squared_distance_to_triangle).
    """
    ab, ac, ap = B - A, C - A, P - A
    bp, cp = P - B, P - C
    dot = lambda x, y: np.einsum('ij,ij->i', x, y)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = 1.0 / (va + vb + vc)
        Q = A + (vb * denom)[:, None] * ab + (vc * denom)[:, None] * ac   # interior
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)                  # edge BC
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        Q[m] = B[m] + w[m, None] * (C[m] - B[m])
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)                            # edge AC
        w = d2 / (d2 - d6)
        Q[m] = A[m] + w[m, None] * ac[m]
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)                            # edge AB
        w = d1 / (d1 - d3)
        Q[m] = A[m] + w[m, None] * ab[m]
    m = (d6 >= 0) & (d5 <= d6)                                           # vertex C
    Q[m] = C[m]
    m = (d3 >= 0) & (d4 <= d3)                                           # vertex B
    Q[m] = B[m]
    m = (d1 <= 0) & (d2 <= 0)                                            # vertex A
    Q[m] = A[m]
    diff = P - Q
    return dot(diff, diff)


def _faces_within(points, reach, tri, centroid_tree):
    """(point index, face index) pairs with exact distance <= reach[point].

    Candidates come from a centroid ball query bounded by each face's
    circumradius, so no face within range is missed; the exact
    point-triangle distance then decides.
    """
    from scipy.spatial import cKDTree
    centroids = tri.mean(axis=1)
    face_radius = np.linalg.norm(tri - centroids[:, None, :], axis=2).max(axis=1)
    # Faces are grouped by circumradius (factor-2 bins), each with its own
    # centroid tree queried with that bin's radius. One query radius set by
    # the largest face would make every point collect every face on a mesh
    # with a few long slivers (O(points x faces) memory).
    rows_all, cols_all = [], []
    level = np.floor(np.log2(np.maximum(face_radius, 1e-300))).astype(np.int64)
    for lv in np.unique(level):
        fids = np.flatnonzero(level == lv)
        tree = centroid_tree if len(fids) == len(tri) else cKDTree(centroids[fids])
        cand = tree.query_ball_point(points, reach + face_radius[fids].max())
        rows = np.repeat(np.arange(len(points)), [len(c) for c in cand])
        cols = fids[np.fromiter((f for c in cand for f in c), dtype=np.int64, count=len(rows))]
        near = (np.linalg.norm(points[rows] - centroids[cols], axis=1)
                <= reach[rows] + face_radius[cols])
        rows, cols = rows[near], cols[near]
        d = np.sqrt(np.maximum(_point_triangle_sqr_dist_batch(
            points[rows], tri[cols, 0], tri[cols, 1], tri[cols, 2]), 0.0))
        keep = d <= reach[rows]
        rows_all.append(rows[keep])
        cols_all.append(cols[keep])
    return np.concatenate(rows_all), np.concatenate(cols_all)


def _faces_within_subset(points, reach, tri, face_ids):
    """_faces_within restricted to face_ids: (point index, global face id) pairs."""
    from scipy.spatial import cKDTree
    if len(points) == 0 or len(face_ids) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    sub_tri = tri[face_ids]
    rows, cols = _faces_within(points, reach, sub_tri, cKDTree(sub_tri.mean(axis=1)))
    return rows, face_ids[cols]


def _seam_side_pairs(V_uv, F, periodic_sides):
    """Per periodic side: (vertex mask on it, faces touching it, faces touching the opposite side).

    The uv mesh is not stitched across a periodic seam: the two sides are the
    same surface points but different vertices and faces, so the conflict
    test has to look across explicitly.

    periodic_sides : [(axis, value), ...], the lower and upper side of each
        periodic axis (as built in embed_loops_one_patch).
    """
    pairs = []
    for axis in sorted({a for a, _ in periodic_sides}):
        vals = [x for a, x in periodic_sides if a == axis]
        lo, hi = min(vals), max(vals)
        tol = SEAM_SIDE_TOL_REL * (hi - lo)
        on = [np.abs(V_uv[:, axis] - x) <= tol for x in (lo, hi)]
        touch = [np.flatnonzero(m[F].any(axis=1)) for m in on]
        pairs.append((on[0], touch[0], touch[1]))
        pairs.append((on[1], touch[1], touch[0]))
    return pairs


def _assign_samples_to_faces(pts, V3d, F, tri, h_face, seam_pairs):
    """(sample index, face index) pairs: each sample's closest face (igl).

    A sample whose closest face touches a periodic side also goes to the
    faces of the opposite side within d_min + h of it (h: the closest face's
    size). The other side's chords lie within their sag of the same seam
    curve, so a seam sample lands on both sides; a sample more than about
    one face away from the seam reaches nothing across it.
    """
    import igl
    d2_min, I, _ = igl.point_mesh_squared_distance(pts, V3d, F)
    I = np.asarray(I, dtype=np.int64)
    if not seam_pairs:
        return np.arange(len(pts)), I
    reach = np.sqrt(np.maximum(d2_min, 0.0)) + h_face[I]
    rows, cols = [np.arange(len(pts))], [I]
    for _, side_faces, opposite_faces in seam_pairs:
        near = np.flatnonzero(np.isin(I, side_faces))
        r, c = _faces_within_subset(pts[near], reach[near], tri, opposite_faces)
        rows.append(near[r])
        cols.append(c)
    pairs = np.unique(np.stack([np.concatenate(rows), np.concatenate(cols)], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _vertex_face_neighborhoods(V3d, F, tri, face_ids, h_v, seam_pairs):
    """vertex id -> faces (among face_ids) around it, for the conflict test.

    The faces incident to the vertex, plus, for a vertex on a periodic side,
    the faces of the opposite side within h_v (its mean edge length) in 3D:
    the faces around the same surface point across the seam, which share no
    vertex index with it in the unstitched uv mesh.
    """
    from collections import defaultdict
    face_ids = np.asarray(sorted(face_ids), dtype=np.int64)
    vert_to_faces = defaultdict(list)
    for fi in face_ids:
        for v in F[fi]:
            vert_to_faces[int(v)].append(int(fi))
    if not seam_pairs:
        return vert_to_faces
    face_verts = np.unique(F[face_ids].ravel())
    for on_side, _, opposite_faces in seam_pairs:
        verts = np.intersect1d(np.flatnonzero(on_side), face_verts)
        r, c = _faces_within_subset(V3d[verts], h_v[verts], tri,
                                    np.intersect1d(opposite_faces, face_ids))
        for a, b in zip(r, c):
            vert_to_faces[int(verts[a])].append(int(b))
    return vert_to_faces


def refine_background_for_feature_conflicts(
    V_uv: np.ndarray,
    F: np.ndarray,
    face,
    edge_geometry_data: dict,
    mmg2d_path: str = "mmg2d_O3",
    beta: float = CONFLICT_BETA,
    max_halvings: int = CONFLICT_MAX_HALVINGS,
    periodic_sides=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively refine the background mesh where feature curves conflict.

    Each iteration: sample the feature loop chains at ~h/2, project the
    samples onto the current mesh (closest face, see _assign_samples_to_faces),
    mark faces whose vertex-neighborhood receives >= 2 distinct curve pieces,
    halve the local target size there (x4 on the metric tensor), and remesh
    with MMG preserving the current size elsewhere. Stops when no face is
    marked or after max_halvings. Purely an accelerator for tracing: leftover
    conflicts are still resolved correctly (but slowly) by exact insertion later.

    periodic_sides : [(axis, value), ...] of a periodic face. Samples and
        vertices on a periodic side also see the faces across the seam within
        a ball of the local face size, so both seam sides are marked.
    """
    from collections import defaultdict
    from scipy.spatial import cKDTree

    chains, chain_glues = _build_loop_chains(face, edge_geometry_data)
    if not chains:
        return V_uv, F
    chain_lens = [L for _, L in chains]
    surface = face.surface

    V_cur = np.asarray(V_uv, dtype=float)
    F_cur = np.asarray(F, dtype=int)

    for it in range(max_halvings + 1):
        V3d = surface.sample(V_cur)
        tri = V3d[F_cur]  # (nf, 3, 3)
        h_face = (np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
                  + np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
                  + np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)) / 3.0
        h_v = per_vertex_edge_length_3d(V3d, F_cur)
        centroid_tree = cKDTree(tri.mean(axis=1))
        seam_pairs = _seam_side_pairs(V_cur, F_cur, periodic_sides) if periodic_sides else []

        pts, arcs, cids = _sample_chains(chains, centroid_tree, h_face)
        sample_ids, face_ids = _assign_samples_to_faces(
            pts, V3d, F_cur, tri, h_face, seam_pairs)

        face_to_samples = defaultdict(list)
        for k, fi in zip(sample_ids, face_ids):
            face_to_samples[int(fi)].append(int(k))
        vert_to_faces = _vertex_face_neighborhoods(
            V3d, F_cur, tri, face_to_samples.keys(), h_v, seam_pairs)

        conflicted = []
        for fi, own in face_to_samples.items():
            cand_faces = {fi}
            for v in F_cur[fi]:
                cand_faces.update(vert_to_faces[int(v)])
            idx = np.unique(np.fromiter(
                (k for cf in cand_faces for k in face_to_samples[cf]),
                dtype=np.int64))   # a sample may sit on several faces
            if _conflict_in_candidate_set(arcs[idx], cids[idx], chain_lens,
                                          beta * float(h_face[fi]),
                                          glues=chain_glues):
                conflicted.append(fi)

        print_if_verbose(
            f"[conflict-refine] iter {it}: {len(pts)} samples, "
            f"{len(conflicted)} conflicted faces, "
            f"{len(V_cur)} verts / {len(F_cur)} faces")
        if not conflicted or it == max_halvings:
            break

        # Per-vertex metric: preserve the current local size everywhere
        # (a plain global factor would coarsen previously refined regions
        # back), halve it at conflicted faces.
        factor_v = 1.0 / h_v ** 2
        marked = np.unique(F_cur[conflicted].ravel())
        factor_v[marked] *= 4.0

        EE, FF_m, GG = compute_first_fundamental_form(surface, V_cur)
        V_cur, F_cur, _ = mmg_remesh(
            V_cur, F_cur,
            EE=EE * factor_v, FF_m=FF_m * factor_v, GG=GG * factor_v,
            metric_factor=1.0,
            hgrad=1.5,
            mmg2d_path=mmg2d_path,
        )
        V_cur = np.asarray(V_cur, dtype=float)
        F_cur = np.asarray(F_cur, dtype=int)

    return V_cur, F_cur


# ---------------------------------------------------------------------------
# Post-CDT Triangle refine pass (conflict size field, insertion-only)
# ---------------------------------------------------------------------------

def _split_mesh_edges_at_points(V, F, splits):
    """Split existing mesh edges at given collinear interior points.

    splits : list of ((i0, i1), pts) where pts is (k, 2) UV points ordered
        from V[i0] to V[i1], all lying on the segment.

    Returns (V_out, F_out, chains) with chains[(i0, i1)] = [i0, new..., i1]
    (vertex ids along the edge, in i0 -> i1 order). Triangle winding is
    preserved. Edges are processed sequentially so a face adjacent to two
    split edges is handled correctly.
    """
    V_list = [list(map(float, p)) for p in np.asarray(V, dtype=float)]
    F_list = [list(map(int, f)) for f in np.asarray(F, dtype=int)]
    chains = {}
    for (i0, i1), pts in splits:
        pts = np.asarray(pts, dtype=float)
        if len(pts) == 0:
            continue
        new_ids = []
        for p in pts:
            new_ids.append(len(V_list))
            V_list.append([float(p[0]), float(p[1])])
        chain = [int(i0)] + new_ids + [int(i1)]
        chains[(int(i0), int(i1))] = chain
        # Find faces currently containing edge {i0, i1}
        adj = [fi for fi, f in enumerate(F_list)
               if i0 in f and i1 in f]
        if not adj:
            raise RuntimeError(
                f"_split_mesh_edges_at_points: edge ({i0}, {i1}) not found "
                "in mesh; constrained segment is not a triangle edge.")
        new_faces = []
        for fi in adj:
            x, y, z = F_list[fi]
            c = [v for v in (x, y, z) if v != i0 and v != i1][0]
            # Directed occurrence of the edge in this face's cyclic order
            cyc = [(x, y), (y, z), (z, x)]
            forward = (int(i0), int(i1)) in [(a, b) for a, b in cyc]
            ch = chain if forward else chain[::-1]
            for k in range(len(ch) - 1):
                new_faces.append([ch[k], ch[k + 1], c])
        for fi in sorted(adj, reverse=True):
            del F_list[fi]
        F_list.extend(new_faces)
    return (np.asarray(V_list, dtype=float),
            np.asarray(F_list, dtype=int),
            chains)


def refine_cdt_with_size_field(
    V: np.ndarray,
    F: np.ndarray,
    segments,
    seg_markers,
    split_markers,
    surface,
    size_field,
    refine_margin: float = 1.2,
    n_dense: int = 48,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Insertion-only refinement of the post-embedding CDT to a 3D size field.

    Purely additive (no vertex is moved or removed, no constrained edge is
    ever collapsed): every segment whose marker is in ``split_markers`` is
    subdivided at uniform *3D arc-length* spacing matched to the local size
    field (the two adjacent triangles are fan-split so the mesh stays
    conforming). Doing this ourselves — rather than letting a mesher split
    at UV midpoints — keeps the on-edge spacing correct in 3D even under an
    anisotropic parameterization. Interior refinement is left entirely to
    the anisotropic MMG pass that follows (a former Triangle 'rpa'
    scaffolding stage was removed; see the comment at the end).

    segments / seg_markers : all constrained edges of the CDT *including the
        full domain boundary* with their tags.
    size_field : (cKDTree over background vertices in 3D, per-vertex 3D edge
        length h_bg), as built in embed_loops_one_patch.

    Returns (V_out, F_out, segments_out, markers_out), 0-based; subdivided
    segments inherit their parent's marker.
    """
    tree_bg, h_bg = size_field
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)
    segments = np.asarray(segments, dtype=int)
    seg_markers = np.asarray(seg_markers, dtype=int).reshape(-1)
    split_markers = set(int(m) for m in split_markers)

    # --- Stage 1: 3D arc-length subdivision of the marked segments -------
    t_dense = np.linspace(0.0, 1.0, n_dense + 1)
    uv_span = float(np.max(V.max(axis=0) - V.min(axis=0)))
    uv_eps = 1e-5 * max(uv_span, 1e-300)   # coordinate noise floor for Triangle
    splits = []
    for (i0, i1), m in zip(segments, seg_markers):
        if int(m) not in split_markers:
            continue
        uv0, uv1 = V[int(i0)], V[int(i1)]
        uv = uv0 + t_dense[:, None] * (uv1 - uv0)
        xyz = surface.sample(uv)
        dL = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        _, nn = tree_bg.query(0.5 * (xyz[:-1] + xyz[1:]))
        w = dL / np.maximum(h_bg[nn], 1e-12)   # metric length per dense piece
        W = float(w.sum())
        nsub = int(np.floor(W + 0.5))          # W in [1.5, 2.5) -> 2 pieces
        if nsub < 2:
            continue
        # Robustness: never create UV subsegments near the coordinate noise
        # floor (Triangle 'runs out of precision' on such input), and bound
        # the per-segment work.
        seg_uv_len = float(np.linalg.norm(uv1 - uv0))
        nsub = min(nsub, 256, max(2, int(seg_uv_len / uv_eps)))
        cw = np.concatenate([[0.0], np.cumsum(w)])
        targets = W * np.arange(1, nsub) / nsub
        tt = np.interp(targets, cw, t_dense)
        splits.append(((int(i0), int(i1)), uv0 + tt[:, None] * (uv1 - uv0)))

    if splits:
        V, F, chains = _split_mesh_edges_at_points(V, F, splits)
        seg_out, mk_out = [], []
        for (i0, i1), m in zip(segments, seg_markers):
            ch = chains.get((int(i0), int(i1)))
            if ch is None:
                seg_out.append([int(i0), int(i1)])
                mk_out.append(int(m))
            else:
                for k in range(len(ch) - 1):
                    seg_out.append([ch[k], ch[k + 1]])
                    mk_out.append(int(m))
        segments = np.asarray(seg_out, dtype=int)
        seg_markers = np.asarray(mk_out, dtype=int)

    # Stage 2 (Triangle 'rpa' interior scaffolding) removed: Ruppert's
    # encroachment rule splits constrained segments at midpoints with no
    # lower bound, and near-coincident feature chords can drive
    # it to ~1e-12 required edges that send mmg2d into an infinite collapse
    # loop. The interior refinement it provided is scaffolding only — the
    # MMG loop right after applies the same size field and inserts interior
    # points itself; the essential part (3D arc-length subdivision of the
    # feature edges, Stage 1 above) is kept.
    if splits:
        # _split_mesh_edges_at_points keeps the mesh conforming but produces
        # zero/negative-area triangles when the vertex opposite a split
        # segment is (nearly) collinear with it (on one face, 7 inverted
        # + 17 near-zero fan triangles -> mmg2d infinite loop). Discard the
        # fan connectivity and rebuild it with a plain constrained Delaunay
        # triangulation of the SAME vertices and SAME constrained segments:
        # 'p' only — no refinement, no Steiner points on segments — so the
        # output is consistently oriented by construction and the feature
        # subdivision / boundary density are unchanged.
        import triangle as tr
        try:
            result = tr.triangulate(
                {
                    'vertices': V,
                    'segments': segments.astype(np.int32),
                    'segment_markers': seg_markers.astype(np.int32).reshape(-1, 1),
                },
                'p',
            )
            return (np.asarray(result['vertices'], dtype=float),
                    np.asarray(result['triangles'], dtype=int),
                    np.asarray(result['segments'], dtype=int),
                    np.asarray(result['segment_markers'], dtype=int).reshape(-1))
        except TimeoutError:
            raise  # alarm_timeout fired: let it reach its caller
        except Exception as e:
            print_if_verbose(
                f"[conflict-refine] plain CDT rebuild failed ({e}); "
                "keeping fan-split mesh")

    return V, F, segments, seg_markers
