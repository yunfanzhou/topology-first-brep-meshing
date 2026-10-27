import signal
import warnings
from contextlib import contextmanager
import numpy as np
from typing import Optional, Tuple
from collections import defaultdict
from scipy.spatial import cKDTree
from operations.internal_operations import split_edge_in_mesh_overloaded
from operations.stitch import stitch_periodic_edges, stitch_periodic_edges_with_split, build_surface_merged_edges_with_apex
from utils.print import print_if_verbose
from utils.data_structures import (
    HalfEdge, LoopTopologyData, LoopGeometryData, PatchInputData, CornerData, SingularityData
)
from operations.remesh_background import remesh_with_error_and_edge_metric, refine_edge_mesh, stitch_singularity_fans_3d, insert_feature_edges_uv, curvature_capped_error, CURVATURE_CAP_BETA
from operations.align import _get_boundary_edges, embed_and_align_seam_edges, align_seam_edges_from_tags
from run_one_patch import construct_connectivity_data
from operations.embed_geodesic import embed_loops_one_patch, fix_nonmanifold_after_stitch, TAG_U_PERIODIC_LB, TAG_U_PERIODIC_UB, TAG_V_PERIODIC_LB, TAG_V_PERIODIC_UB, TAG_SINGULARITY_BASE
import igl
GEOMETRY_TOL = 1e-10
TOPOLOGY_TOL = 1e-7
DEFAULT_STRIP_WIDTH_RATIO = 0.01
SAME_VERTEX_TOL = 1e-10
PERIODIC_TOL = 1e-3 # Absolute but comparing to 2pi
BOUNDARY_TOL = 0.01 # relative to uv domain size
EMBED_TIMEOUT_SECONDS = 3 * 60 * 60 # 3h

@contextmanager
def alarm_timeout(seconds):
    """Raise TimeoutError after *seconds* via SIGALRM.

    No-op when SIGALRM is unavailable (e.g. not in the main thread).
    """
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    try:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
    except (ValueError, AttributeError):
        # Not main thread or no SIGALRM support
        yield
        return
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def detect_periodicity(face, u_lb, u_ub, v_lb, v_ub):
    TWO_PI = 2.0 * np.pi
    u_span = u_ub - u_lb
    v_span = v_ub - v_lb

    u_periodic = getattr(face.surface, 'u_periodic', False)
    v_periodic = getattr(face.surface, 'v_periodic', False)
    stype = getattr(face.surface, 'shape_name', '')
    print_if_verbose(f"  Surface type: {stype}, initial periodicity flags: u={u_periodic} v={v_periodic}")
    if stype in ('Cylinder', 'Cone', 'Sphere'):
        u_periodic = abs(u_span - TWO_PI) < PERIODIC_TOL or u_span > TWO_PI + PERIODIC_TOL
    if stype == 'Torus':
        u_periodic = abs(u_span - TWO_PI) < PERIODIC_TOL or u_span > TWO_PI + PERIODIC_TOL
        v_periodic = abs(v_span - TWO_PI) < PERIODIC_TOL or v_span > TWO_PI + PERIODIC_TOL
    if stype == 'Revolution':
        u_periodic = abs(u_span - TWO_PI) < PERIODIC_TOL or u_span > TWO_PI + PERIODIC_TOL
        curve = face.surface.curve
        curve_type = getattr(curve, 'shape_name', '')
        if curve_type in ('Circle', 'Ellipse'):
            v_periodic = abs(v_span - TWO_PI) < PERIODIC_TOL or v_span > TWO_PI + PERIODIC_TOL
        elif hasattr(curve, 'closed'):
            v_periodic = curve.closed
    if stype == 'Extrusion':
        # u drives the curve, v is linear extrusion direction (never periodic)
        curve = face.surface.curve
        curve_type = getattr(curve, 'shape_name', '')
        if curve_type in ('Circle', 'Ellipse'):
            u_periodic = abs(u_span - TWO_PI) < PERIODIC_TOL or u_span > TWO_PI + PERIODIC_TOL
    if stype == 'Offset':
        # Periodicity is inherited from the underlying base surface
        base = face.surface.surface
        base_type = getattr(base, 'shape_name', '')
        if base_type in ('Cylinder', 'Cone', 'Sphere'):
            u_periodic = abs(u_span - TWO_PI) < PERIODIC_TOL or u_span > TWO_PI + PERIODIC_TOL
        elif base_type == 'Torus':
            u_periodic = abs(u_span - TWO_PI) < PERIODIC_TOL or u_span > TWO_PI + PERIODIC_TOL
            v_periodic = abs(v_span - TWO_PI) < PERIODIC_TOL or v_span > TWO_PI + PERIODIC_TOL
        elif base_type == 'Revolution':
            u_periodic = abs(u_span - TWO_PI) < PERIODIC_TOL or u_span > TWO_PI + PERIODIC_TOL
            curve = base.curve
            curve_type = getattr(curve, 'shape_name', '')
            if curve_type in ('Circle', 'Ellipse'):
                v_periodic = abs(v_span - TWO_PI) < PERIODIC_TOL or v_span > TWO_PI + PERIODIC_TOL
            elif hasattr(curve, 'closed'):
                v_periodic = curve.closed
        elif base_type == 'Extrusion':
            curve = base.curve
            curve_type = getattr(curve, 'shape_name', '')
            if curve_type in ('Circle', 'Ellipse'):
                u_periodic = abs(u_span - TWO_PI) < PERIODIC_TOL or u_span > TWO_PI + PERIODIC_TOL
        else:
            u_periodic = getattr(base, 'u_periodic', False)
            v_periodic = getattr(base, 'v_periodic', False)

    return u_periodic, v_periodic
# ---------------------------------------------------------------------------
# Sub-mesh combination
# ---------------------------------------------------------------------------
def combine_sub_meshes(sub_meshes):
    """Combine multiple (V, F, factor) tuples into a single mesh.

    Offsets face indices for each sub-mesh beyond the first.

    Returns
    -------
    V_combined : ndarray (nv_total, 2)
    F_combined : ndarray (nf_total, 3)
    final_factor : float (from last sub-mesh)
    """
    V_list = []
    F_list = []
    vertex_offset = 0
    final_factor = None
    for V_sd, F_sd, factor_sd in sub_meshes:
        V_arr = np.asarray(V_sd)
        F_arr = np.asarray(F_sd)
        V_list.append(V_arr)
        F_list.append(F_arr + vertex_offset)
        vertex_offset += len(V_arr)
        final_factor = factor_sd
    V_combined = np.vstack(V_list)
    F_combined = np.vstack(F_list)
    return V_combined, F_combined, final_factor


# ---------------------------------------------------------------------------
# Grid initialization
# ---------------------------------------------------------------------------
def init_grid_mesh(
    singularities,
    num_u_segments: int,
    num_v_segments: int,
    u_lower_bound: float,
    u_upper_bound: float,
    v_lower_bound: float,
    v_upper_bound: float,
    strip_width_ratio: float = DEFAULT_STRIP_WIDTH_RATIO,
) -> Tuple[np.ndarray, np.ndarray, dict, list]:
    """Create a structured UV grid mesh, optionally with singularity strip fans.

    Direct port of the C++ ``init_grid_mesh`` in TriMeshCADMeshing.cpp.

    Parameters
    ----------
    singularities : dict of singualrityid to singularity objects, where singularity objects have attributes:
         - ``first2d``: (u, v) coordinates of the singularity in the parametric domain.
         - ``uiso``: boolean, True if the singularity is u-isolated, False if v-isolated.
         - ``point2d``: (u, v) coordinates of a point on the singular iso-line.
         - ``point3d``: (x, y, z) coordinates of the singularity in 3D space.
    strip_width_ratio : width of the fan region as a fraction of domain size.

    Returns
    -------
    uvs : ndarray, shape (nv, 2)
        Vertex UV coordinates.
    F : ndarray, shape (nf, 3), dtype int
        Triangle connectivity (0-based).
    sing_info : dict of dict
        Per-singularity bookkeeping (skipped singularities are left out):
        ``center_vertex_id`` (-1 here, set when the fans are stitched) and
        ``boundary_side`` ('left'/'right'/'bottom'/'top', None for an
        interior singularity).
    sub_domains : list of (ul, uu, vl, vu) tuples
        Bounds of the produced grid(s): two for an interior singularity, one
        otherwise, shrunk by the boundary-singularity strips.
    """
    stride = num_v_segments + 1
    grid_verts = (num_u_segments + 1) * stride
    grid_tris = 2 * num_u_segments * num_v_segments

    # --- classify singularities, compute sizes --------------------------------
    extra_verts = 0
    extra_tris = 0
    non_bnd_idx = None  # index of the non-boundary singularity (at most one)

    # Per-singularity classification
    is_boundary = {}
    is_at_min = {}
    n_segs = {}

    skipped_singularities = set()
    for idx, sing in singularities.items():
        at_bnd = False
        at_min = False
        start = np.asarray(sing['first2d'], dtype=float)

        if sing['uiso']:
            u_val = start[0]
            if u_val < u_lower_bound - GEOMETRY_TOL or u_val > u_upper_bound + GEOMETRY_TOL:
                print_if_verbose(f"  Singularity {idx} at u={u_val:.6f} outside domain "
                                 f"[{u_lower_bound:.6f}, {u_upper_bound:.6f}], skipping")
                skipped_singularities.add(idx)
                continue
            if abs(u_val - u_lower_bound) < GEOMETRY_TOL:
                at_bnd = True
                at_min = True
            elif abs(u_val - u_upper_bound) < GEOMETRY_TOL:
                at_bnd = True
                at_min = False
            ns = num_v_segments
        else:
            v_val = start[1]
            if v_val < v_lower_bound - GEOMETRY_TOL or v_val > v_upper_bound + GEOMETRY_TOL:
                print_if_verbose(f"  Singularity {idx} at v={v_val:.6f} outside domain "
                                 f"[{v_lower_bound:.6f}, {v_upper_bound:.6f}], skipping")
                skipped_singularities.add(idx)
                continue
            if abs(v_val - v_lower_bound) < GEOMETRY_TOL:
                at_bnd = True
                at_min = True
            elif abs(v_val - v_upper_bound) < GEOMETRY_TOL:
                at_bnd = True
                at_min = False
            ns = num_u_segments

        is_boundary[idx] = at_bnd
        is_at_min[idx] = at_min
        n_segs[idx] = ns

        if not at_bnd:
            non_bnd_idx = idx
            extra_verts += grid_verts  # second grid
            extra_tris += grid_tris   # second grid (no fans)
        # boundary singularities: no extra verts/tris (fans stitched after MMG)

    total_verts = grid_verts + extra_verts
    total_tris = grid_tris + extra_tris
    F = np.zeros((total_tris, 3), dtype=int)
    uvs = np.zeros((total_verts, 2))
    fo = 0  # face offset (current write position in F)
    # Bookkeeping per singularity (center_vertex_id filled during stitching)
    sing_info = {}
    for idx, sing in singularities.items():
        if idx in skipped_singularities:
            continue
        side = None
        if is_boundary[idx]:
            if sing['uiso']:
                side = 'left' if is_at_min[idx] else 'right'
            else:
                side = 'bottom' if is_at_min[idx] else 'top'
        sing_info[idx] = {"center_vertex_id": -1, "boundary_side": side}

    # --- helper functions -----------------------------------------------------
    def fill_verts(base, u0, u1, v0, v1):
        for i in range(num_u_segments + 1):
            u = u0 + i / num_u_segments * (u1 - u0)
            for j in range(num_v_segments + 1):
                v = v0 + j / num_v_segments * (v1 - v0)
                uvs[base + i * stride + j, 0] = u
                uvs[base + i * stride + j, 1] = v

    def fill_tris(base):
        nonlocal fo
        for i in range(num_u_segments):
            for j in range(num_v_segments):
                a = base + i * stride + j
                b = base + (i + 1) * stride + j
                c = base + (i + 1) * stride + (j + 1)
                d = base + i * stride + (j + 1)
                F[fo] = [a, b, c]
                fo += 1
                F[fo] = [a, c, d]
                fo += 1

    # --- build grids ----------------------------------------------------------

    if non_bnd_idx is not None:
        sing = singularities[non_bnd_idx]
        start = np.asarray(sing['first2d'], dtype=float)

        g1 = 0           # grid-1 vertex base
        g2 = grid_verts   # grid-2 vertex base

        if sing['uiso']:
            u_val = start[0]
            gap = (u_upper_bound - u_lower_bound) * strip_width_ratio

            fill_verts(g1, u_lower_bound, u_val - gap, v_lower_bound, v_upper_bound)
            fill_tris(g1)
            fill_verts(g2, u_val + gap, u_upper_bound, v_lower_bound, v_upper_bound)
            fill_tris(g2)
        else:
            v_val = start[1]
            gap = (v_upper_bound - v_lower_bound) * strip_width_ratio

            fill_verts(g1, u_lower_bound, u_upper_bound, v_lower_bound, v_val - gap)
            fill_tris(g1)
            fill_verts(g2, u_lower_bound, u_upper_bound, v_val + gap, v_upper_bound)
            fill_tris(g2)
    else:
        # no non-boundary singularity – single grid
        fill_verts(0, u_lower_bound, u_upper_bound, v_lower_bound, v_upper_bound)
        fill_tris(0)

    # --- boundary singularity strips ------------------------------------------
    # Push grid boundary vertices inward to create a gap for later fan stitching.
    # No fan triangles or center vertices — those are added after MMG remeshing.
    for idx, sing in singularities.items():
        if idx == non_bnd_idx or idx in skipped_singularities:
            continue

        g1 = 0
        g2 = grid_verts if non_bnd_idx is not None else 0
        ns = n_segs[idx]
        if sing['uiso']:
            inward = (u_upper_bound - u_lower_bound) * strip_width_ratio

            if is_at_min[idx]:
                gb = g1
                for j in range(ns + 1):
                    uvs[gb + j, 0] += inward
            else:
                gb = g2
                col = num_u_segments * stride
                for j in range(ns + 1):
                    uvs[gb + col + j, 0] -= inward
        else:
            inward = (v_upper_bound - v_lower_bound) * strip_width_ratio

            if is_at_min[idx]:
                gb = g1
                for i in range(ns + 1):
                    uvs[gb + i * stride, 1] += inward
            else:
                gb = g2
                for i in range(ns + 1):
                    uvs[gb + i * stride + num_v_segments, 1] -= inward

    # --- compute sub-domain bounds matching the produced mesh -----------------
    if non_bnd_idx is not None:
        sing = singularities[non_bnd_idx]
        start = np.asarray(sing['first2d'], dtype=float)
        if sing['uiso']:
            u_val = start[0]
            gap = (u_upper_bound - u_lower_bound) * strip_width_ratio
            sub_domains = [
                [u_lower_bound, u_val - gap, v_lower_bound, v_upper_bound],
                [u_val + gap, u_upper_bound, v_lower_bound, v_upper_bound],
            ]
        else:
            v_val = start[1]
            gap = (v_upper_bound - v_lower_bound) * strip_width_ratio
            sub_domains = [
                [u_lower_bound, u_upper_bound, v_lower_bound, v_val - gap],
                [u_lower_bound, u_upper_bound, v_val + gap, v_upper_bound],
            ]
    else:
        sub_domains = [[u_lower_bound, u_upper_bound, v_lower_bound, v_upper_bound]]

    # Boundary-singularity inward push: g1 is sub_domains[0], g2 is sub_domains[-1]
    for idx, sing in singularities.items():
        if idx == non_bnd_idx or idx in skipped_singularities:
            continue
        if sing['uiso']:
            inward = (u_upper_bound - u_lower_bound) * strip_width_ratio
            if is_at_min[idx]:
                sub_domains[0][0] += inward
            else:
                sub_domains[-1][1] -= inward
        else:
            inward = (v_upper_bound - v_lower_bound) * strip_width_ratio
            if is_at_min[idx]:
                sub_domains[0][2] += inward
            else:
                sub_domains[-1][3] -= inward

    sub_domains = [tuple(sd) for sd in sub_domains]
    return uvs, F, sing_info, sub_domains

def create_initial_mesh(
    face,
    num_u_segments: int = 8,
    num_v_segments: int = 8,
    u_lower: Optional[float] = None,
    u_upper: Optional[float] = None,
    v_lower: Optional[float] = None,
    v_upper: Optional[float] = None,
    strip_width_ratio: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, dict, bool, list]:
    """Create the initial UV grid mesh on a CAD patch (no remeshing).

    Grid stage of the port of ``TriMeshCADMeshing::create_good_quality_initial_mesh``;
    the MMG remesh (``run_mmg_remesh_on_initial_mesh``) is done by the caller.
    If any bound is None, all four are taken from ``decide_trim_domain``.

    Parameters
    ----------
    face : abs face object.  Must have ``.has_singularity`` and ``.surface``
        (with ``.sample()``, ``.derivative()``, ``.trim_domain``,
        and ``.singularities`` when singularities exist).
    num_u_segments, num_v_segments : grid resolution (default 8x8).
    strip_width_ratio : width of singularity fan strips as fraction of domain.

    Returns
    -------
    uvs : (nv, 2) grid UV vertices.
    F : (nf, 3) grid triangle connectivity (0-based).
    sing_info : dict of per-singularity bookkeeping, see ``init_grid_mesh``
        (empty dict if no singularities).
    too_small : bool, True when the parametric domain is degenerate (only
        checked when the bounds come from ``decide_trim_domain``).
    sub_domains : list of (ul, uu, vl, vu) grid bounds, see ``init_grid_mesh``.
    """
    too_small = False
    # Get UV bounds
    if u_lower is None or u_upper is None or v_lower is None or v_upper is None:
        sub_domains, _, too_small, _ = decide_trim_domain(face)
        # Without target_edge_length, decide_trim_domain returns a single sub-domain
        u_lower, u_upper, v_lower, v_upper = sub_domains[0]

    # 1. Create initial grid mesh
    uvs, F, sing_info, sub_domains = init_grid_mesh(
        face.singularities,
        num_u_segments,
        num_v_segments,
        u_lower,
        u_upper,
        v_lower,
        v_upper,
        strip_width_ratio,
    )
    return uvs, F, sing_info, too_small, sub_domains

def project_point_onto_polyline(point, P, tree=None):
    """Find the closest point on polyline *P* to *point*.

    Uses a KDTree on the polyline vertices to locate the nearest vertex,
    then checks adjacent segments for the exact closest point.

    Parameters
    ----------
    point : (3,) array-like – query point.
    P : (n, 3) ndarray – polyline vertices.
    tree : cKDTree, optional – pre-built tree on *P* (avoids rebuilding).

    Returns
    -------
    seg_idx : int
        Index of the segment (0 .. len(P)-2).
    t : float
        Interpolation parameter on that segment, in [0, 1].
    closest : (3,) ndarray
        The closest point on the polyline.
    """
    if tree is None:
        tree = cKDTree(P)

    _, nearest_idx = tree.query(point)

    # Check segments adjacent to the nearest vertex
    candidates = []
    if nearest_idx > 0:
        candidates.append(nearest_idx - 1)
    if nearest_idx < len(P) - 1:
        candidates.append(nearest_idx)

    best_dist = np.inf
    best_seg = 0
    best_t = 0.0
    best_point = P[0]

    for seg_idx in candidates:
        a, b = P[seg_idx], P[seg_idx + 1]
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < GEOMETRY_TOL:
            # degenerate segment: the closer endpoint
            t = 0.0 if np.linalg.norm(point - a) <= np.linalg.norm(point - b) else 1.0
        else:
            t = float(np.clip(np.dot(point - a, ab) / ab_len_sq, 0.0, 1.0))
        proj = a + t * ab
        dist = np.linalg.norm(point - proj)
        if dist < best_dist:
            best_dist = dist
            best_seg = seg_idx
            best_t = t
            best_point = proj.copy()

    return best_seg, best_t, best_point


def blend_corner_gaps_by_t(t, target_start, target_end, C_start, C_end):
    """Displacement (1 - s) a_s + s a_e with s = (t - t_s)/(t_e - t_s).

    a_s = target_start - C(t_s), a_e = target_end - C(t_e): the corner gaps to the
    curve at the corner parameters. Added to curve points C(t) this gives the
    snap targets Γ(t) = C(t) + δ(t) exactly (the 1-D Laplacian with weights
    1/Δt, i.e. straight in t rather than in sample index).
    """
    t = np.asarray(t, dtype=float)
    s = (t - t[0]) / (t[-1] - t[0])
    return np.outer(1.0 - s, target_start - C_start) + np.outer(s, target_end - C_end)


def snap_edge_polyline_to_corners(P, target_start, target_end, t_values=None, curve=None):
    """Project canonical corner positions onto the polyline, trim, and smooth.

    Ported from C++ ``trimming_execute`` + ``solve_laplacian_for_displacement``
    in ``EdgeMeshCADMeshingOperations``.

    1. Project *target_start* / *target_end* onto the polyline (closest point).
    2. Extract the sub-polyline between the two projected points.
    3. Distribute the corner displacement over the interior vertices so that the
       endpoints match the target positions exactly.

    With t_values and curve (P = curve.sample(t_values)) the displacement is
    blended by curve parameter t, from the corner gaps to the curve at the
    corner parameters (blend_corner_gaps_by_t): the snapped polyline is then
    exactly the snap targets Γ(t) = C(t) + δ(t). Without them it falls back to
    blending by sample index (uniform 1-D Laplacian).

    Parameters
    ----------
    P : (n, 3) ndarray – polyline vertices, ordered along the curve.
    target_start : (3,) ndarray – canonical position of the start corner.
    target_end   : (3,) ndarray – canonical position of the end corner.
    t_values : (n,) ndarray, optional – curve parameters of P.
    curve : abs curve, optional – the curve P samples (needed for the t blend).

    Returns
    -------
    P_snapped : (m, 3) ndarray – trimmed and snapped polyline.
    t_trimmed : (m,) ndarray – only when t_values is given, aligned with
        P_snapped: inner samples keep their t, each projected end gets
        t_k + τ (t_{k+1} − t_k) from its segment k and fraction τ (the corner
        parameter).
    """
    if len(P) < 2:
        return P if t_values is None else (P, np.asarray(t_values, dtype=float))

    # --- 1. project corners onto the polyline (KDTree) ---
    tree = cKDTree(P)
    seg_s, t_s, proj_s = project_point_onto_polyline(target_start, P, tree)
    seg_e, t_e, proj_e = project_point_onto_polyline(target_end, P, tree)

    # Ensure start comes before end along the polyline
    if seg_s > seg_e or (seg_s == seg_e and t_s > t_e):
        seg_s, seg_e = seg_e, seg_s
        t_s, t_e = t_e, t_s
        proj_s, proj_e = proj_e, proj_s
        target_start, target_end = target_end, target_start

    # --- 2. extract sub-polyline [proj_s, interior vertices, proj_e] ---
    sub_points = [proj_s]
    # interior vertices strictly between the two projections: a projection at fraction 1 of
    # seg_s (or 0 of seg_e) is the vertex P[seg_s + 1] (or P[seg_e]) itself, so skip that vertex
    first = seg_s + 1 + (1 if t_s == 1.0 else 0)
    last = seg_e + 1 - (1 if t_e == 0.0 else 0)
    for i in range(first, last):
        sub_points.append(P[i].copy())
    sub_points.append(proj_e)

    P_sub = np.array(sub_points)
    m = len(P_sub)
    t_trimmed = None
    if t_values is not None:
        t_values = np.asarray(t_values, dtype=float)
        t_trimmed = np.concatenate([[t_values[seg_s] + t_s * (t_values[seg_s + 1] - t_values[seg_s])],
                                    t_values[first:last],
                                    [t_values[seg_e] + t_e * (t_values[seg_e + 1] - t_values[seg_e])]])
    if m < 2:
        P_ends = np.vstack([target_start, target_end])
        return P_ends if t_values is None else (P_ends, t_trimmed)

    # --- 3. distribute the corner displacement ---
    if t_trimmed is not None and curve is not None and t_trimmed[-1] != t_trimmed[0]:
        # by t: curve points at the trimmed parameters plus the blended corner gaps
        C = np.asarray(curve.sample(t_trimmed.reshape(-1, 1)), dtype=float).reshape(-1, 3)
        P_snapped = C + blend_corner_gaps_by_t(t_trimmed, target_start, target_end, C[0], C[-1])
    else:
        # by sample index (uniform 1-D Laplacian)
        d_start = target_start - P_sub[0]    # displacement at start
        d_end   = target_end   - P_sub[-1]   # displacement at end
        t_param = np.linspace(0.0, 1.0, m)
        P_snapped = P_sub + np.outer(1.0 - t_param, d_start) + np.outer(t_param, d_end)
    P_snapped[0]  = target_start   # exact
    P_snapped[-1] = target_end     # exact
    return P_snapped if t_values is None else (P_snapped, t_trimmed)


def create_feature_edge_geometry_data(part,
                                      corner_data: CornerData,
                                      target_error: float = 5e-2) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Create edge polylines and snap them to canonical corner positions.

    Parameters
    ----------
    part : ABS part object.
    corner_data : CornerData whose ``corner_id_to_position`` maps corner
        (vertex) id -> (3,) 3-D position, read from the input B-rep data.
    target_error : refinement target for ``refine_edge_mesh``.

    Returns
    -------
    edge_geometry_data : dict edge id -> (n, 3) snapped polyline (empty for
        degenerate edges).
    edge_t_values : dict edge id -> curve parameters of the polyline samples
        (only for edges meshed by ``refine_edge_mesh``).
    """
    edge_geometry_data = {}
    edge_t_values = {}
    for i, edge in enumerate(part.Solid.edges):
        curve = edge.curve3d
        print_if_verbose(f"  Edge {i} ({type(curve).__name__}) ")
        P = np.empty((0, 3))
        ctype = type(curve).__name__
        if ctype != "Other":
            if ctype == "Circle" and curve.radius < GEOMETRY_TOL:
                edge_geometry_data[edge.id] = P
                continue
            if ctype == "Line":
                t_values, P = refine_edge_mesh(curve,
                                            target_error=target_error,
                                            initial_segments=2)
            else:
                is_closed = edge.start_vertex == edge.end_vertex
                t_values, P = refine_edge_mesh(curve,
                                            target_error=target_error,
                                            initial_segments=4 if is_closed else 3)
            edge_t_values[edge.id] = t_values
        edge_geometry_data[edge.id] = P

    # Snap each polyline to the canonical corner positions.
    # Open edges: project corners onto polyline (KDTree), trim to the
    # sub-polyline between the two projected points, then solve a 1-D
    # Laplacian to distribute the displacement smoothly.
    # Closed edges (start_vertex == end_vertex): skip projection/trim
    # (the polyline already covers the full curve) but still solve the
    # Laplacian to snap both endpoints to the canonical corner position.
    for edge in part.Solid.edges:
        P = edge_geometry_data.get(edge.id)
        if P is None or len(P) == 0:
            continue
        t_values = edge_t_values.get(edge.id)
        if edge.start_vertex == edge.end_vertex:
            # Closed curve – no projection/trim, only the corner snap (by t when available)
            target = corner_data.corner_id_to_position[edge.start_vertex]
            if t_values is not None and len(t_values) == len(P) and t_values[-1] != t_values[0]:
                P_snapped = P + blend_corner_gaps_by_t(t_values, target, target, P[0], P[-1])
            else:
                n = len(P)
                t_param = np.linspace(0.0, 1.0, n)
                P_snapped = P + np.outer(1.0 - t_param, target - P[0]) + np.outer(t_param, target - P[-1])
            P_snapped[0]  = target
            P_snapped[-1] = target
            edge_geometry_data[edge.id] = P_snapped
            continue
        target_start = corner_data.corner_id_to_position[edge.start_vertex]
        target_end = corner_data.corner_id_to_position[edge.end_vertex]
        if t_values is not None and len(t_values) == len(P):
            # keep edge_t_values aligned with the trimmed polyline (corner parameters at the ends)
            edge_geometry_data[edge.id], edge_t_values[edge.id] = snap_edge_polyline_to_corners(
                P, target_start, target_end, t_values=t_values, curve=edge.curve3d
            )
        else:
            edge_geometry_data[edge.id] = snap_edge_polyline_to_corners(
                P, target_start, target_end
            )

    return edge_geometry_data, edge_t_values

def initial_mesh_simple_processing(V_uv, F,
                                   face,
                                   sub_domains,
                                   u_periodic,
                                   v_periodic,
                                   sing_info):
    """Process mesh through alignment, stitching, and fan stitch (no embedding).

    Parameters
    ----------
    sub_domains : list of (ul, uu, vl, vu) sub-domain bounds.
        Their overall bounds are used for periodic edge detection.
        Gap-facing boundaries are used to find interior singularity edges.
    """
    # Use full domain bounds (first sub-domain for outer boundary detection)
    u_lower, u_upper, v_lower, v_upper = sub_domains[0][0], sub_domains[-1][1], sub_domains[0][2], sub_domains[0][3]
    # For u-iso split, u bounds span all sub-domains
    # For v-iso split, v bounds span all sub-domains
    if len(sub_domains) > 1:
        # Determine overall bounds from all sub-domains
        u_lower = min(sd[0] for sd in sub_domains)
        u_upper = max(sd[1] for sd in sub_domains)
        v_lower = min(sd[2] for sd in sub_domains)
        v_upper = max(sd[3] for sd in sub_domains)

    V_uv = V_uv.tolist() if isinstance(V_uv, np.ndarray) else V_uv
    F = F.tolist() if isinstance(F, np.ndarray) else F
    connectivity = construct_connectivity_data(V_uv, F)
    if u_periodic or v_periodic:
        (u_periodic_edges, v_periodic_edges) = embed_and_align_seam_edges(
            V_uv, connectivity, face, u_lower, u_upper, v_lower, v_upper, u_periodic, v_periodic)
    else:
        u_periodic_edges = []
        v_periodic_edges = []

    bnd_edges = _get_boundary_edges(F)
    sing_edges = {}
    for idx, info in sing_info.items():
        side = info.get('boundary_side')
        sing = face.singularities[idx]
        pos = np.asarray(sing['first2d'], dtype=float)

        if side is not None:
            # Boundary singularity: edges at the outer boundary
            axis = 0 if side in ('left', 'right') else 1
            target_val = u_lower if side == 'left' else u_upper if side == 'right' else v_lower if side == 'bottom' else v_upper
            tol = BOUNDARY_TOL * (u_upper - u_lower if axis == 0 else v_upper - v_lower)
            sing_edges[idx] = [
                e for e in bnd_edges
                if abs(V_uv[e[0]][axis] - target_val) < tol
                and abs(V_uv[e[1]][axis] - target_val) < tol
            ]
        else:
            # Interior singularity: a sub-domain's edge faces the singularity if the
            # singularity lies at or beyond that edge. Pick the gap-facing side per
            # sub-domain by position rather than tolerance — the adaptive strip ratio
            # can be as large as 0.2*span, which a fixed tolerance would miss.
            axis = 0 if sing['uiso'] else 1
            raw = pos[axis]
            tol = BOUNDARY_TOL * (u_upper - u_lower if axis == 0 else v_upper - v_lower)
            gap_edges = []
            for sd in sub_domains:
                lb = sd[axis * 2]
                ub = sd[axis * 2 + 1]
                if raw >= ub - SAME_VERTEX_TOL:
                    target = ub
                elif raw <= lb + SAME_VERTEX_TOL:
                    target = lb
                else:
                    continue
                gap_edges.extend([
                    e for e in bnd_edges
                    if abs(V_uv[e[0]][axis] - target) < tol
                    and abs(V_uv[e[1]][axis] - target) < tol
                ])
            sing_edges[idx] = gap_edges
    split_singularity_edges(V_uv, connectivity, sing_edges)

    V_3d = face.surface.sample(np.array(V_uv))
    V_3d_list = V_3d.tolist()
    (merged_u_vertices,
        merged_v_vertices,
        stitched_to_crux_vid,
        merged_u_sub_chains,
        merged_v_sub_chains) = stitch_periodic_edges_with_split(
        V_3d_list, connectivity, u_periodic_edges, v_periodic_edges, singularity_edges=sing_edges)
    F_3d_list = connectivity.F

    if face.singularities is not None and len(face.singularities) > 0:
        V_3d_list, F_3d, fan_sing_info = stitch_singularity_fans_3d(
            V_3d_list,
            F_3d_list,
            face,
            sing_edges,
        )
        # Merge: keep boundary_side from original sing_info, add center_vertex_id from fan stitch
        for idx, fan_info in fan_sing_info.items():
            if idx in sing_info:
                sing_info[idx]["center_vertex_id"] = fan_info["center_vertex_id"]
            else:
                sing_info[idx] = fan_info
    else:
        F_3d = connectivity.F

    surface_merged_edges = build_surface_merged_edges_with_apex(
        merged_u_sub_chains, merged_v_sub_chains, face, sing_info)
    return V_3d_list, F_3d, merged_u_vertices, merged_v_vertices, sing_info, surface_merged_edges

class CollapsedInteriorIsoLineError(BaseException):
    """A face without singularity records has an iso-line strictly inside its
    parametric domain that collapses to a point in 3-D. The interior
    singularity path (strip + two sub-domains + fans) assumes a periodic face
    and closed gap loops; on a bounded face it yields a figure-eight boundary,
    and no fallback produces a usable mesh. Derived from BaseException on
    purpose so neither the per-face ``except Exception`` fallback nor main's
    per-patch retry can swallow it: the run must end here and the input be
    looked at."""


class _PosteriorSingularityDetected(Exception):
    """Raised inside the manifold loop when the lifted base mesh reveals a
    collapsed iso-line on a face without singularity records. Carries the
    synthesized records; the caller re-runs the face through the singularity
    path with them."""
    def __init__(self, records):
        super().__init__(f"{len(records)} collapsed iso-line(s) detected")
        self.records = records


def _iso_line_record(surface, uiso, iso_val, run_lo, run_hi, n_samples=65):
    """Build a singularity record (exporter schema) for the iso-line
    u=iso_val (uiso) or v=iso_val, spanning [run_lo, run_hi] in the other
    coordinate. Returns (record, chain_length_3d)."""
    run = np.linspace(run_lo, run_hi, n_samples)
    uv = np.c_[np.full(n_samples, iso_val), run] if uiso else np.c_[run, np.full(n_samples, iso_val)]
    P = np.asarray(surface.sample(uv), float)
    chain = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    apex = P.mean(axis=0)
    rec = {'first2d': uv[0].copy(), 'firstpar': float(run[0]),
           'last2d': uv[-1].copy(), 'lastpar': float(run[-1]),
           'point2d': uv[n_samples // 2].copy(), 'point3d': apex,
           'precision': float(np.max(np.linalg.norm(P - apex, axis=1))),
           'rank': 1, 'uiso': bool(uiso)}
    return rec, chain


def detect_collapsed_iso_lines(face, V_uv, F, bounds, u_periodic, v_periodic,
                               model_diag=None):
    """Posterior degeneracy check on the lifted base mesh.

    Purely geometric: no pcurves, no loop topology. For each of the four
    domain-boundary iso-lines, the mesh vertices on that line are chained in
    parametric order and the chain's 3-D length is summed. A side whose lifted
    length is below ``config.collapsed_iso_line_rel_tol * model_diag`` is an
    apex; a singularity record is synthesized for it with the domain bound as
    its coordinate, so downstream classification is exact.

    Interior iso-lines are only diagnosed, never synthesized: interior mesh
    edges shorter than the tolerance seed candidate lines, and a candidate
    whose full line collapses raises ``CollapsedInteriorIsoLineError``, which
    ends the run (see that class).

    Parameters
    ----------
    V_uv, F : base mesh (after the first mmg pass), full unpadded domain.
    bounds : (u_lower, u_upper, v_lower, v_upper).
    u_periodic, v_periodic : seam sides are never degenerate lines; skipped.
    model_diag : part bbox diagonal. Falls back to the lifted face bbox
        diagonal when None (main.py passes the part value).

    Returns
    -------
    dict of synthesized records keyed like the exporter ('000', '001', ...),
    empty when nothing collapsed.

    Raises
    ------
    CollapsedInteriorIsoLineError : a collapsed iso-line strictly inside the
        domain.
    """
    import utils.config as _config
    V_uv = np.asarray(V_uv, float)
    F = np.asarray(F, int)
    u_lower, u_upper, v_lower, v_upper = bounds
    u_span, v_span = u_upper - u_lower, v_upper - v_lower
    P = np.asarray(face.surface.sample(V_uv), float)
    if model_diag is None:
        model_diag = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    tol_3d = _config.collapsed_iso_line_rel_tol * model_diag
    on_tol = 1e-7 * max(u_span, v_span)

    found = []   # (uiso, iso_val, chain, where)
    # --- boundary sides: chain the mesh vertices on each side ---
    sides = []
    if not u_periodic:
        sides += [(True, u_lower), (True, u_upper)]
    if not v_periodic:
        sides += [(False, v_lower), (False, v_upper)]
    on_side = np.zeros(len(V_uv), bool)
    for uiso, val in sides:
        col = 0 if uiso else 1
        ids = np.where(np.abs(V_uv[:, col] - val) < on_tol)[0]
        if len(ids) < 2:
            continue
        on_side[ids] = True
        ids = ids[np.argsort(V_uv[ids, 1 - col])]
        chain = float(np.linalg.norm(np.diff(P[ids], axis=0), axis=1).sum())
        print_if_verbose(f"  posterior check: side {'u' if uiso else 'v'}={val:.6g} "
                         f"has {len(ids)} vertices, lifted chain length {chain:.3e} "
                         f"(tol {tol_3d:.3e})")
        if chain < tol_3d:
            found.append((uiso, val, chain, 'boundary'))

    # --- interior lines: diagnose only. Seeded by short interior edges,
    # confirmed by sampling the surface along the full line; a hit ends the
    # run (no fallback can mesh it, see CollapsedInteriorIsoLineError).
    E = igl.edges(F)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    short = E[(L < tol_3d) & ~(on_side[E[:, 0]] & on_side[E[:, 1]])]
    if len(short):
        mid = 0.5 * (V_uv[short[:, 0]] + V_uv[short[:, 1]])
        for uiso, col, lo, hi, span in ((True, 0, u_lower, u_upper, u_span),
                                        (False, 1, v_lower, v_upper, v_span)):
            vals = np.sort(mid[:, col])
            # cluster candidate iso values; one representative per cluster
            reps = [vals[0]]
            for x in vals[1:]:
                if x - reps[-1] > 1e-3 * span:
                    reps.append(x)
            for x in reps:
                if x - lo < on_tol or hi - x < on_tol:
                    continue  # boundary line, handled above
                run_lo, run_hi = (v_lower, v_upper) if uiso else (u_lower, u_upper)
                _, chain = _iso_line_record(face.surface, uiso, x, run_lo, run_hi)
                print_if_verbose(f"  posterior check: interior candidate "
                                 f"{'u' if uiso else 'v'}={x:.6g} lifted length {chain:.3e}")
                if chain < tol_3d:
                    raise CollapsedInteriorIsoLineError(
                        f"face {face.id}: interior iso-line "
                        f"{'u' if uiso else 'v'}={x:.9g} (domain "
                        f"[{lo:.6g}, {hi:.6g}]) collapses to a point in 3-D "
                        f"(lifted length {chain:.3e} < tol {tol_3d:.3e}) but the "
                        f"face has no singularity record for it. Interior "
                        f"singularities cannot be synthesized posteriorly; "
                        f"ending the run.")

    records = {}
    for uiso, val, chain, where in found:
        run_lo, run_hi = (v_lower, v_upper) if uiso else (u_lower, u_upper)
        rec, _ = _iso_line_record(face.surface, uiso, val, run_lo, run_hi)
        key = f"{len(records):03d}"
        records[key] = rec
        print_if_verbose(f"  posterior singularity #{key}: {where} iso-line "
                         f"{'u' if uiso else 'v'}={val:.6g} collapses to "
                         f"{np.round(rec['point3d'], 6).tolist()} (lifted length {chain:.2e}, "
                         f"tol {tol_3d:.2e})")
    return records


def _run_initial_meshing_for_face(
    face,
    edge_geometry_data: dict,
    target_error: Optional[float] = 5e-2,
    mmg2d_path: str = "mmg2d_O3",
    num_u_segments: int = 8,
    num_v_segments: int = 8,
    prev_V_uv = None, prev_F = None,
    prev_sub_domains = None,
    attempt = 0,
    feature_conflict_refine: bool = True,
    model_diag: Optional[float] = None,
    _posterior_singularity_done: bool = False,
) -> tuple:
    """Run the full initial meshing pipeline for one CAD face.

    Mirrors the C++ ``meshing_patch`` function: creates a UV grid, refines it
    with MMG, inserts feature edges, aligns periodic seam edges, and stitches
    singularity fans.

    Parameters
    ----------
    face : abs Face object.
    edge_geometry_data : dict mapping edge id -> (nv, 3) 3-D polyline array,
        as returned by ``create_feature_edge_geometry_data``.
    mmg2d_path : path to the mmg2d executable.
    num_u_segments, num_v_segments : initial grid resolution.
    prev_V_uv, prev_F : UV mesh from previous attempt, used on retry
        (attempt > 0) to skip mmg remeshing and upsample instead.
    model_diag : part bbox diagonal, scale for the posterior collapsed
        iso-line detection (see ``detect_collapsed_iso_lines``).
    _posterior_singularity_done : internal; set on the single re-run after a
        posterior detection so it cannot recurse again.

    Returns
    -------
    V_3d : list of (x, y, z) vertex positions of the stitched 3-D mesh.
    F : (nf, 3) triangle connectivity (0-based) of the stitched 3-D mesh.
    merged_u_vertices, merged_v_vertices : vertex ids merged along the u / v
        seam (empty when the surface is not periodic in that direction).
    sing_info : dict mapping singularity index -> {"boundary_side": ...,
        "center_vertex_id": int}, the center vertex set by
        ``stitch_singularity_fans_3d`` (empty dict when no singularities).
    surface_merged_edges : seam edges from ``build_surface_merged_edges_with_apex``.
    V_uv_fallback, F_fallback, sub_domains : the UV mesh and sub-domain bounds
        to pass back as prev_V_uv / prev_F / prev_sub_domains on the next
        attempt; None on retries (attempt > 0) and on too-small domains.
    """
    # Keep the caller's target for the posterior re-run below: the curvature
    # cap floors at target/256, so feeding the capped value back in would
    # divide by 256 again on singular faces (the target collapses by orders of
    # magnitude and mmg runs away).
    target_error_in = target_error
    # Curvature cap: the effective error target may not exceed a fraction of
    # the surface's minimum principal-curvature radius, or the background mesh
    # leaves the surface's reach and projections of nearby features alias.
    if target_error is not None:
        capped = curvature_capped_error(face.surface, target_error)
        if capped < target_error:
            print_if_verbose(
                f"  curvature cap: target_error {target_error:.3e} -> "
                f"{capped:.3e} (R_min={capped / CURVATURE_CAP_BETA:.3e})")
            target_error = capped

    uvs, F, sing_info, too_small, sub_domains_initial = create_initial_mesh(
        face, num_u_segments=num_u_segments, num_v_segments=num_v_segments,
        u_lower=None, u_upper=None, v_lower=None, v_upper=None
    )
    singularity_sides = {info['boundary_side']: int(idx)
                         for idx, info in sing_info.items()
                         if info.get('boundary_side') is not None}

    # Full unpadded domain bounds for periodicity detection (a surface property,
    # independent of how the mesh is split into sub-domains).
    u_lower = min(sd[0] for sd in sub_domains_initial)
    u_upper = max(sd[1] for sd in sub_domains_initial)
    v_lower = min(sd[2] for sd in sub_domains_initial)
    v_upper = max(sd[3] for sd in sub_domains_initial)
    u_periodic, v_periodic = detect_periodicity(face, u_lower, u_upper, v_lower, v_upper)

    if attempt > 0:
        base_V = prev_V_uv if prev_V_uv is not None else uvs
        base_F = prev_F if prev_F is not None else F
        base_V, base_F = igl.upsample(np.asarray(base_V), np.asarray(base_F), attempt)
        # If we were handed sub_domains from the previous attempt, they describe
        # base_V's actual structure (adaptive gap). Otherwise fall back to the
        # initial-grid sub_domains (static gap) which match `uvs`.
        sd_for_simple = prev_sub_domains if (prev_V_uv is not None and prev_sub_domains is not None) else sub_domains_initial
        result = initial_mesh_simple_processing(
            base_V, base_F, face, sd_for_simple, u_periodic, v_periodic, sing_info)
        return (*result, None, None, None)

    if too_small:
        result = initial_mesh_simple_processing(
            uvs, F, face, sub_domains_initial, u_periodic, v_periodic, sing_info)
        return (*result, None, None, None)

    # Defaults so the except path below has coherent state if any of the
    # pre-embedding steps (initial remesh, sub-domain remesh, seam
    # stitching) raise before V_uv_fallback / F_fallback / final_factor /
    # sub_domains have been computed.
    V_uv_fallback = uvs.copy() if isinstance(uvs, np.ndarray) else np.array(uvs)
    F_fallback = F.copy() if isinstance(F, np.ndarray) else np.array(F)
    final_factor = None
    sub_domains = sub_domains_initial
    padding_ratio = None

    manifold_refine_times = 0
    import utils.config as _config
    manifold_refine_max = _config.manifold_refine_max
    # Preserve the original grid connectivity so each retry re-remeshes the same
    # input (mirrors how `uvs` is held constant across iterations). `F` itself is
    # clobbered into the remeshed / list-converted mesh below and must not be fed
    # back into remesh_with_error_and_edge_metric.
    F_grid = F
    try:
        while True:
            print_if_verbose(f". =============== manifold attempt #{manifold_refine_times}: target_error={target_error * 0.5**manifold_refine_times:.3e}")
            V_uv, F, final_factor = remesh_with_error_and_edge_metric(
                face.surface, uvs, F_grid, target_error=target_error * 0.5**manifold_refine_times,
                hgrad=1.5,
                mmg2d_path=mmg2d_path, max_iterations=10,
            )
            # Posterior degeneracy check (faces without exporter records
            # only): if the lifted base mesh has a collapsed iso-line, hand
            # synthesized records to the caller, which re-runs the face once
            # through the regular singularity path.
            if (manifold_refine_times == 0 and not face.singularities
                    and not _posterior_singularity_done):
                synthesized = detect_collapsed_iso_lines(
                    face, V_uv, F, (u_lower, u_upper, v_lower, v_upper),
                    u_periodic, v_periodic, model_diag=model_diag)
                if synthesized:
                    raise _PosteriorSingularityDetected(synthesized)
            # get the current edge length in 3d as reference
            before_edge_lengths = igl.avg_edge_length(np.array(face.surface.sample(np.array(V_uv))), np.array(F))
            # adaptively refine to get the padding ratio
            sub_domains, padding_ratio, _, interior_singularity_idx = decide_trim_domain(face, target_edge_length=before_edge_lengths)
            print_if_verbose("   decide trim domain: sub_domains={} padding_ratio={:.3f}".format(
                sub_domains, padding_ratio if padding_ratio is not None else -1.
            ))
            if len(sub_domains) > 1 or padding_ratio is not None:
                # Per sub-domain: create grid, remesh, refine singularity boundary
                sub_meshes = []
                for sd_bounds in sub_domains:
                    sd_ul, sd_uu, sd_vl, sd_vu = sd_bounds
                    uvs_sd, F_sd, sing_info_sd, _, _ = create_initial_mesh(
                        face, num_u_segments=num_u_segments, num_v_segments=num_v_segments,
                        u_lower=sd_ul, u_upper=sd_uu, v_lower=sd_vl, v_upper=sd_vu,
                        strip_width_ratio=0.)  # strip is implicit in the sub-domain bounds
                    V_sd, F_sd, factor_sd = remesh_with_error_and_edge_metric(
                        face.surface, uvs_sd, F_sd, target_error=target_error * 0.5**manifold_refine_times,
                        hgrad=1.5,
                        mmg2d_path=mmg2d_path, max_iterations=10,
                    )
                    V_sd, F_sd = refine_for_singularity_boundary(V_sd, F_sd, face, sd_bounds)
                    sub_meshes.append((V_sd, F_sd, factor_sd))

                # Combine sub-meshes into a single mesh
                if len(sub_meshes) == 1:
                    V_uv, F, final_factor = sub_meshes[0]
                else:
                    V_uv, F, final_factor = combine_sub_meshes(sub_meshes)
            # Save pre-embed mesh for fallback (before list conversion / seam alignment)
            V_uv_fallback = V_uv.copy() if isinstance(V_uv, np.ndarray) else np.array(V_uv)
            F_fallback = F.copy() if isinstance(F, np.ndarray) else np.array(F)
            # convert V_uv and F to list if not yet
            V_uv = V_uv.tolist() if isinstance(V_uv, np.ndarray) else V_uv
            F = F.tolist() if isinstance(F, np.ndarray) else F
            connectivity = construct_connectivity_data(V_uv, F)
            print_if_verbose(f" detect peirodicity  {u_periodic}, {v_periodic}")
            (u_periodic_edges, v_periodic_edges) = embed_and_align_seam_edges(V_uv, connectivity, face, u_lower, u_upper, v_lower, v_upper, u_periodic, v_periodic)
            V_3d = face.surface.sample(np.array(V_uv))
            V_3d_list = V_3d.tolist()
            F_uv = [row[:] for row in connectivity.F] # keep track of the F_uv before seam stitching
            (merged_u_vertices,
            merged_v_vertices,
            stitched_to_crux_vid) = stitch_periodic_edges(V_3d_list, connectivity, u_periodic_edges, v_periodic_edges)
            F_3d_list = connectivity.F
            # check both edge manifold and validity. And no changes of topology 
            # (i.e. if no perioidc should only have one boundary loop, single perioidc 2 bnd loops, double periodic 0 bnd loops)

            from utils.mesh_utils import mesh_is_valid, topology_consistency
            manifold_ok = bool(igl.is_edge_manifold(np.array(F_3d_list)))
            valid_ok = bool(mesh_is_valid(F_3d_list))
            topology_ok = bool(topology_consistency(F_3d_list, u_periodic, v_periodic))
            if manifold_ok and valid_ok and topology_ok:
                break
            if manifold_ok and valid_ok:
                # Wrong boundary-loop count is a topology defect: refining
                # cannot change it, so retrying only burns time (hours and
                # millions of vertices on a bad face).
                print_if_verbose("  boundary loop count mismatch after stitching; "
                                 "refinement cannot fix topology, stopping manifold retries")
                break
            if manifold_refine_times + 1 >= manifold_refine_max:
                print_if_verbose(f"  giving up manifold retries after {manifold_refine_max} rounds "
                                 f"(manifold={manifold_ok}, valid={valid_ok}, topology={topology_ok})")
                break
            # Non-manifold edges survived stitching. Tighten the remesh
            # (target_error scales by 0.5**manifold_refine_times) so the next
            # pass has a real chance to resolve them.
            manifold_refine_times += 1

        V_3d_arr = fix_nonmanifold_after_stitch(V_uv, F_uv, F_3d_list, face.surface)
        V_3d_list = V_3d_arr.tolist()
        if not igl.is_edge_manifold(np.array(F_3d_list)):
            print_if_verbose("Non-manifold edges remain after initial remesh. ")
        with alarm_timeout(EMBED_TIMEOUT_SECONDS):
            V_mmg, F_mmg, tagged_edges = embed_loops_one_patch(V_uv,
                                                V_3d_list,
                                                F_uv,
                                                F_3d_list,
                                                face,
                                                edge_geometry_data,
                                                sub_domains,
                                                before_embedding_edge_length=before_edge_lengths,
                                                padding_ratio=padding_ratio,
                                                target_error=target_error * 0.5**manifold_refine_times,
                                                max_iterations=10,
                                                mmg_factor=final_factor,
                                                mmg2d_path=mmg2d_path,
                                                u_periodic=u_periodic,
                                                v_periodic=v_periodic,
                                                singularity_sides=singularity_sides,
                                                interior_singularity_idx=interior_singularity_idx,
                                                use_feature_conflict_refine=(
                                                    feature_conflict_refine
                                                    and len(sub_domains) <= 1
                                                    and padding_ratio is None))
        V_mmg_list = V_mmg.tolist()
        F_mmg_list = F_mmg.tolist()
        V_uv_fallback = np.array(V_mmg_list)
        F_fallback = np.array(F_mmg_list)
        connectivity = construct_connectivity_data(V_mmg_list, F_mmg_list)
        u_periodic_edges, v_periodic_edges = align_seam_edges_from_tags(
            V_mmg_list, connectivity,
            tagged_edges.get(TAG_U_PERIODIC_LB, []),
            tagged_edges.get(TAG_U_PERIODIC_UB, []),
            tagged_edges.get(TAG_V_PERIODIC_LB, []),
            tagged_edges.get(TAG_V_PERIODIC_UB, []),
        )
        F_uv = [row[:] for row in connectivity.F]
        sing_edges = {}
        for idx in (face.singularities or {}):
            tag = TAG_SINGULARITY_BASE + int(idx)
            if tag in tagged_edges:
                sing_edges[idx] = tagged_edges[tag]
        split_singularity_edges(V_mmg_list, connectivity, sing_edges)
        V_3d = face.surface.sample(np.array(V_mmg_list))
        V_3d_list = V_3d.tolist()
        (merged_u_vertices,
         merged_v_vertices,
        stitched_to_crux_vid,
        merged_u_sub_chains,
        merged_v_sub_chains) = stitch_periodic_edges_with_split(V_3d_list, connectivity, u_periodic_edges, v_periodic_edges, singularity_edges=sing_edges)
        F_3d_list = connectivity.F

        if face.singularities is not None and len(face.singularities) > 0:
            V_3d_list, F_3d, fan_sing_info = stitch_singularity_fans_3d(
                V_3d_list,
                F_3d_list,
                face,
                sing_edges,
            )
            for idx, fan_info in fan_sing_info.items():
                if idx in sing_info:
                    sing_info[idx]["center_vertex_id"] = fan_info["center_vertex_id"]
                else:
                    sing_info[idx] = fan_info
        else:
            F_3d = connectivity.F

        if not igl.is_edge_manifold(np.array(F_3d)):
            print_if_verbose("Non-manifold edges remain after embedding and stitching. ")
        surface_merged_edges = build_surface_merged_edges_with_apex(
            merged_u_sub_chains, merged_v_sub_chains, face, sing_info)
        return V_3d_list, F_3d, merged_u_vertices, merged_v_vertices, sing_info, surface_merged_edges, V_uv_fallback, F_fallback, sub_domains
    except _PosteriorSingularityDetected as det:
        face.singularities = det.records
        print_if_verbose(f"  re-running face {face.id} with {len(det.records)} "
                         f"synthesized singularity record(s)")
        return _run_initial_meshing_for_face(
            face, edge_geometry_data, target_error=target_error_in,
            mmg2d_path=mmg2d_path, num_u_segments=num_u_segments,
            num_v_segments=num_v_segments, prev_V_uv=prev_V_uv, prev_F=prev_F,
            prev_sub_domains=prev_sub_domains, attempt=attempt,
            feature_conflict_refine=feature_conflict_refine,
            model_diag=model_diag, _posterior_singularity_done=True)
    except Exception as e:
        warnings.warn(f"initial meshing failed ({type(e).__name__}: {e}), falling back to insert_feature_edges_uv on pre-embedding mesh")
        V_embedded, F_embedded = insert_feature_edges_uv(np.array(V_uv_fallback), np.array(F_fallback), face, edge_geometry_data, mmg2d_path=mmg2d_path, base_factor=final_factor)
        result = initial_mesh_simple_processing(
            V_embedded, F_embedded, face, sub_domains,
            u_periodic, v_periodic, sing_info)
        
        return (*result, V_uv_fallback, F_fallback, sub_domains)


def run_initial_meshing_for_face(face, *args, **kwargs):
    """``_run_initial_meshing_for_face`` with the triangles oriented like the face.

    The mesh is built in the surface's (u, v) orientation, so its normals follow
    the surface normal. A face with ``surface_orientation == False`` points the
    other way, so its triangles are reversed here, once, before any tracing.
    The uv fallback mesh (prev_F for the next attempt) is left in uv order.
    """
    V_3d, F_3d, *rest = _run_initial_meshing_for_face(face, *args, **kwargs)
    if not face.surface_orientation:
        if isinstance(F_3d, np.ndarray):
            F_3d = F_3d[:, [0, 2, 1]].copy()
        else:
            F_3d = [[f[0], f[2], f[1]] for f in F_3d]
    return (V_3d, F_3d, *rest)


def refine_for_singularity_boundary(V_uv, F_uv, face, sub_domain_bounds):
    """Refine mesh edges near singularity-facing boundaries of a sub-domain.

    Parameters
    ----------
    V_uv, F_uv : mesh vertices and faces.
    face : abs Face object with singularities.
    sub_domain_bounds : (ul, uu, vl, vu) tuple for this sub-domain.
        The function identifies which boundary of this sub-domain faces a
        singularity and ensures at least 3 edges exist there.
    """
    if not face.singularities:
        return V_uv, F_uv

    sd_ul, sd_uu, sd_vl, sd_vu = sub_domain_bounds

    sides_to_refine = set()
    for sing in face.singularities.values():
        start = np.asarray(sing['first2d'], dtype=float)

        if sing['uiso']:
            u_val = start[0]
            # Check if the singularity is adjacent to this sub-domain's u boundaries
            # Interior sing: sub-domain's uu or ul edge faces the gap
            if abs(u_val - sd_uu) < GEOMETRY_TOL * 10:
                # Singularity is just beyond this sub-domain's right edge
                sides_to_refine.add((0, sd_uu))
            elif abs(u_val - sd_ul) < GEOMETRY_TOL * 10:
                # Singularity is just beyond this sub-domain's left edge
                sides_to_refine.add((0, sd_ul))
            elif u_val > sd_uu:
                # Singularity is to the right — this sub-domain's right edge faces it
                sides_to_refine.add((0, sd_uu))
            elif u_val < sd_ul:
                # Singularity is to the left — this sub-domain's left edge faces it
                sides_to_refine.add((0, sd_ul))
            # Boundary singularity at this sub-domain's edge (padding already applied)
            elif abs(u_val - sd_ul) < GEOMETRY_TOL:
                sides_to_refine.add((0, sd_ul))
            elif abs(u_val - sd_uu) < GEOMETRY_TOL:
                sides_to_refine.add((0, sd_uu))
        else:
            v_val = start[1]
            if abs(v_val - sd_vu) < GEOMETRY_TOL * 10:
                sides_to_refine.add((1, sd_vu))
            elif abs(v_val - sd_vl) < GEOMETRY_TOL * 10:
                sides_to_refine.add((1, sd_vl))
            elif v_val > sd_vu:
                sides_to_refine.add((1, sd_vu))
            elif v_val < sd_vl:
                sides_to_refine.add((1, sd_vl))
            elif abs(v_val - sd_vl) < GEOMETRY_TOL:
                sides_to_refine.add((1, sd_vl))
            elif abs(v_val - sd_vu) < GEOMETRY_TOL:
                sides_to_refine.add((1, sd_vu))

    for axis, side_target in sides_to_refine:
        bnd_edges = _get_boundary_edges(F_uv)
        side_edges = [
            e for e in bnd_edges
            if abs(V_uv[e[0]][axis] - side_target) < SAME_VERTEX_TOL and abs(V_uv[e[1]][axis] - side_target) < SAME_VERTEX_TOL
        ]
        if len(side_edges) < 3:
            upsample_level = 3 - len(side_edges)
            print_if_verbose(f" upsampling boundary at axis={axis} target={side_target:.6f} with level {upsample_level}")
            V_uv, F_uv = igl.upsample(np.array(V_uv), np.array(F_uv), upsample_level)

    return V_uv, F_uv

def decide_trim_domain(face, sub_domains=None, target_edge_length=None):
    """
    Decide the trim domain for a face, returning a list of sub-domain bounds.

    Parameters
    ----------
    face : abs Face object.
    sub_domains : list of (ul, uu, vl, vu) tuples, optional.
        If provided, used as starting bounds (skips periodic clipping).
        Can be a single sub-domain or multiple (for interior singularity).
    target_edge_length : float, optional.
        When provided, computes adaptive singularity padding via bisection.

    Returns
    -------
    sub_domains : list of (u_lower, u_upper, v_lower, v_upper) tuples
    padding_ratio : float or None (for boundary singularities)
    too_small : bool
    interior_singularity_idx : int or None, the singularity the domain is
        split around
    """
    surface = face.surface
    padding_ratio = None
    interior_strip_ratio = None
    too_small = False

    if sub_domains is not None and len(sub_domains) > 0:
        # Use provided sub-domains as starting point
        # Derive global bounds from the sub-domain list
        u_lower = min(sd[0] for sd in sub_domains)
        u_upper = max(sd[1] for sd in sub_domains)
        v_lower = min(sd[2] for sd in sub_domains)
        v_upper = max(sd[3] for sd in sub_domains)
    else:
        # Get UV bounds from surface
        trim_domain = surface.trim_domain  # [[u_min, u_max], [v_min, v_max]]
        u_lower = float(trim_domain[0, 0])
        u_upper = float(trim_domain[0, 1])
        v_lower = float(trim_domain[1, 0])
        v_upper = float(trim_domain[1, 1])

        # For periodic surfaces whose domain spans more than one full period, clip
        # to exactly one period so the mesh only covers [lower, lower + 2π].
        TWO_PI = 2.0 * np.pi
        PERIODIC_TOL = 1e-3
        stype = getattr(surface, 'shape_name', '')
        if stype in ('Cylinder', 'Cone', 'Sphere', 'Revolution', 'Torus'):
            if u_upper - u_lower > TWO_PI + PERIODIC_TOL:
                u_upper = u_lower + TWO_PI
        if stype == 'Torus':
            if v_upper - v_lower > TWO_PI + PERIODIC_TOL:
                v_upper = v_lower + TWO_PI

        # Guard against degenerate parametric domains (Triangle does not support small meshes)
        if (abs(u_upper - u_lower) < TOPOLOGY_TOL or abs(v_upper - v_lower) < TOPOLOGY_TOL):
            warnings.warn(
                f"The parametric domain of face {face} is too small. "
                f"u size {u_upper - u_lower} v size {v_upper - v_lower}"
            )
            u_upper = u_upper + GEOMETRY_TOL
            v_upper = v_upper + GEOMETRY_TOL
            too_small = True

    if target_edge_length is not None:
        u_lower, u_upper, v_lower, v_upper, padding_ratio, interior_strip_ratio = singularity_adaptive_padding_ratio(
            face, target_edge_length, u_lower, u_upper, v_lower, v_upper
        )

    # Build sub-domain list
    interior_singularity_idx = None
    if interior_strip_ratio is not None and face.singularities:
        ratio_lo, ratio_hi = interior_strip_ratio
        # Find the interior singularity and split the domain
        for idx, sing in face.singularities.items():
            start = np.asarray(sing['first2d'], dtype=float)
            if sing['uiso']:
                u_val = start[0]
                at_min = abs(u_val - u_lower) < GEOMETRY_TOL
                at_max = abs(u_val - u_upper) < GEOMETRY_TOL
                if not at_min and not at_max:
                    span = u_upper - u_lower
                    sub_domains = [
                        (u_lower, u_val - ratio_lo * span, v_lower, v_upper),
                        (u_val + ratio_hi * span, u_upper, v_lower, v_upper),
                    ]
                    interior_singularity_idx = int(idx)
                    break
            else:
                v_val = start[1]
                at_min = abs(v_val - v_lower) < GEOMETRY_TOL
                at_max = abs(v_val - v_upper) < GEOMETRY_TOL
                if not at_min and not at_max:
                    span = v_upper - v_lower
                    sub_domains = [
                        (u_lower, u_upper, v_lower, v_val - ratio_lo * span),
                        (u_lower, u_upper, v_val + ratio_hi * span, v_upper),
                    ]
                    interior_singularity_idx = int(idx)
                    break
        else:
            sub_domains = [(u_lower, u_upper, v_lower, v_upper)]
    else:
        sub_domains = [(u_lower, u_upper, v_lower, v_upper)]

    return sub_domains, padding_ratio, too_small, interior_singularity_idx
    

def singularity_adaptive_padding_ratio(face,
                                       before_embedding_edge_length,
                                       u_l, u_u, v_l, v_u):
    """
    Compute adaptive padding around singularities.

    For boundary singularities: adjusts domain bounds inward.
    For interior singularities: computes a strip half-width ratio (gap size).

    Starts with an aggressive 20% padding ratio, then bisects until the lifted
    3D distance from a test point to the singularity apex is approximately
    equal to before_embedding_edge_length.

    Returns adjusted (u_l, u_u, v_l, v_u), the boundary padding ratio,
    and the interior strip ratio.
    """
    INITIAL_RATIO = 0.20
    MAX_BISECT_ITERS = 20
    REL_TOL = 0.1  # 10% relative tolerance

    adj_u_l = u_l
    adj_u_u = u_u
    adj_v_l = v_l
    adj_v_u = v_u

    surface = face.surface
    singularities = face.singularities
    ratio = None
    # Interior strip ratio is per-side (ratio_lo, ratio_hi) since the singularity
    # may sit closer to one boundary than the other.
    interior_strip_ratio = None

    def _lifted_distance(test_uv, apex_3d):
        """3D distance from a lifted UV point to the singularity apex."""
        pt_3d = surface.sample(np.array([test_uv]))[0]
        return np.linalg.norm(np.asarray(pt_3d, dtype=float) - apex_3d)

    def _bisect_ratio(test_fn, initial_ratio):
        """Bisect to find ratio where test_fn(ratio) ≈ before_embedding_edge_length."""
        d_init = test_fn(initial_ratio)
        if d_init <= before_embedding_edge_length:
            return initial_ratio
        lo, hi = 0.0, initial_ratio
        for _ in range(MAX_BISECT_ITERS):
            mid = (lo + hi) / 2.0
            d_mid = test_fn(mid)
            if d_mid < before_embedding_edge_length:
                lo = mid
            else:
                hi = mid
            if abs(d_mid - before_embedding_edge_length) < REL_TOL * before_embedding_edge_length:
                break
        return (lo + hi) / 2.0

    for idx, sing in singularities.items():
        start = np.asarray(sing['first2d'], dtype=float)
        apex_3d = np.asarray(sing['point3d'], dtype=float)

        if sing['uiso']:
            u_val = start[0]
            if u_val < u_l - GEOMETRY_TOL or u_val > u_u + GEOMETRY_TOL:
                continue

            at_min = abs(u_val - u_l) < GEOMETRY_TOL
            at_max = abs(u_val - u_u) < GEOMETRY_TOL

            domain_size = u_u - u_l
            v_mid = (v_l + v_u) / 2.0

            if not at_min and not at_max:
                # Interior singularity: bisect each side independently with the
                # natural cap = available space / domain_size. If the cap binds
                # (lifted distance at the boundary is already small enough), the
                # strip would otherwise consume the entire side -> fall back to
                # the default static strip width relative to available space.
                max_ratio_lo = (u_val - u_l) / domain_size
                max_ratio_hi = (u_u - u_val) / domain_size
                def test_fn_interior_u_lo(r):
                    return _lifted_distance([u_val - r * domain_size, v_mid], apex_3d)
                def test_fn_interior_u_hi(r):
                    return _lifted_distance([u_val + r * domain_size, v_mid], apex_3d)

                if test_fn_interior_u_lo(max_ratio_lo) <= before_embedding_edge_length:
                    ratio_lo = DEFAULT_STRIP_WIDTH_RATIO * max_ratio_lo
                else:
                    ratio_lo = _bisect_ratio(test_fn_interior_u_lo, min(INITIAL_RATIO, max_ratio_lo))
                if test_fn_interior_u_hi(max_ratio_hi) <= before_embedding_edge_length:
                    ratio_hi = DEFAULT_STRIP_WIDTH_RATIO * max_ratio_hi
                else:
                    ratio_hi = _bisect_ratio(test_fn_interior_u_hi, min(INITIAL_RATIO, max_ratio_hi))
                interior_strip_ratio = (ratio_lo, ratio_hi)
                print_if_verbose(f"Singularity {idx}: interior u strip ratios = ({ratio_lo:.4f}, {ratio_hi:.4f})")
            else:
                # Boundary singularity
                def test_fn_boundary_u(r, _at_min=at_min):
                    test_u = (u_l + r * domain_size) if _at_min else (u_u - r * domain_size)
                    return _lifted_distance([test_u, v_mid], apex_3d)

                ratio = _bisect_ratio(test_fn_boundary_u, INITIAL_RATIO)
                if at_min:
                    adj_u_l = u_l + ratio * domain_size
                else:
                    adj_u_u = u_u - ratio * domain_size
                print_if_verbose(f"Singularity {idx}: u adjusted padding ratio = {ratio:.3f}")

        else:  # v-iso singularity
            v_val = start[1]
            if v_val < v_l - GEOMETRY_TOL or v_val > v_u + GEOMETRY_TOL:
                continue

            at_min = abs(v_val - v_l) < GEOMETRY_TOL
            at_max = abs(v_val - v_u) < GEOMETRY_TOL

            domain_size = v_u - v_l
            u_mid = (u_l + u_u) / 2.0

            if not at_min and not at_max:
                # Interior singularity: bisect each side independently with the
                # natural cap = available space / domain_size. If the cap binds
                # (lifted distance at the boundary is already small enough), the
                # strip would otherwise consume the entire side -> fall back to
                # the default static strip width relative to available space.
                max_ratio_lo = (v_val - v_l) / domain_size
                max_ratio_hi = (v_u - v_val) / domain_size
                def test_fn_interior_v_lo(r):
                    return _lifted_distance([u_mid, v_val - r * domain_size], apex_3d)
                def test_fn_interior_v_hi(r):
                    return _lifted_distance([u_mid, v_val + r * domain_size], apex_3d)

                if test_fn_interior_v_lo(max_ratio_lo) <= before_embedding_edge_length:
                    ratio_lo = DEFAULT_STRIP_WIDTH_RATIO * max_ratio_lo
                else:
                    ratio_lo = _bisect_ratio(test_fn_interior_v_lo, min(INITIAL_RATIO, max_ratio_lo))
                if test_fn_interior_v_hi(max_ratio_hi) <= before_embedding_edge_length:
                    ratio_hi = DEFAULT_STRIP_WIDTH_RATIO * max_ratio_hi
                else:
                    ratio_hi = _bisect_ratio(test_fn_interior_v_hi, min(INITIAL_RATIO, max_ratio_hi))
                interior_strip_ratio = (ratio_lo, ratio_hi)
                print_if_verbose(f"Singularity {idx}: interior v strip ratios = ({ratio_lo:.4f}, {ratio_hi:.4f})")
            else:
                # Boundary singularity
                def test_fn_boundary_v(r, _at_min=at_min):
                    test_v = (v_l + r * domain_size) if _at_min else (v_u - r * domain_size)
                    return _lifted_distance([u_mid, test_v], apex_3d)

                ratio = _bisect_ratio(test_fn_boundary_v, INITIAL_RATIO)
                if at_min:
                    adj_v_l = v_l + ratio * domain_size
                else:
                    adj_v_u = v_u - ratio * domain_size
                print_if_verbose(f"Singularity {idx}: v adjusted padding ratio = {ratio:.3f}")

    return adj_u_l, adj_u_u, adj_v_l, adj_v_u, ratio, interior_strip_ratio

def split_singularity_edges(V, connectivity, sing_edges):
    # V, F and sig_edges shoul dbe vids in the same mesh.
    # eg. is V is 3d mesh, sing_edges should be indexing to the 3d mesh (as opposed to unstitched uv mesh )
    if sing_edges is None:
        return

    for idx, edges in sing_edges.items():
        # Repeatedly split all current edges once per pass until we have >= 3 edges.
        while len(edges) < 3:
            if not edges:
                break

            current_edges = list(edges)
            edges.clear()
            for e in current_edges:
                split_edge_in_mesh_overloaded(V, connectivity, e)
                new_vid = len(V) - 1
                edges.append((e[0], new_vid))
                edges.append((new_vid, e[1]))
                print_if_verbose(f"Updated singularity edge {e} to {(e[0], new_vid)} and {(new_vid, e[1])} in singularity index {idx} after splitting edge {e}")

def build_singularity_data_from_face(face, sing_info: dict):
    """Build a list of SingularityData from the meshed sing_info and ABS face.

    Parameters
    ----------
    face : abs Face object.  ``face.singularities`` is a dict mapping
        singularity index -> singularity object with a ``"point3d"`` key
        giving the 3-D position.
    sing_info : dict returned by ``run_initial_meshing_for_face``, mapping
        singularity index -> ``{"center_vertex_id": int}``.

    Returns
    -------
    list[SingularityData] if the face has singularities, else None.
    """
    if not face.singularities or not sing_info:
        return None
    result = []
    for idx, sing in face.singularities.items():
        center_vid = sing_info.get(idx, {}).get("center_vertex_id", -1)
        if center_vid < 0:
            continue
        position = np.asarray(sing["point3d"], dtype=float)
        result.append(SingularityData(singularity_vids=[center_vid], position=position))
    return result if result else None


def build_corner_data_from_abs_part(part) -> CornerData:
    """Build corner_id -> 3-D position mapping from ABS solid topology.

    Corner positions are the B-rep vertex positions (``part.vertices[0]``);
    ``create_feature_edge_geometry_data`` snaps the edge polylines to them.
    """
    corner_id_to_position = {}
    for id, vertex in enumerate(part.vertices[0]):
        corner_id_to_position[id] = np.asarray(vertex, dtype=float)
    return CornerData(corner_id_to_position)


def build_feature_edge_to_patches_from_abs_part(part) -> dict:
    """Build feature_edge_id -> list[face_id] adjacency from ABS solid topology.

    Each feature edge is recorded once per adjacent face (seam edges appear
    twice in the loop but are counted once per face).
    """
    feature_edge_to_patches = defaultdict(list)
    for face in part.Solid.faces:
        seen_edge_ids: set = set()
        for loop in face.loops:
            for he in loop.halfedges:
                eid = he.edge.id
                if eid not in seen_edge_ids:
                    feature_edge_to_patches[eid].append(face.id)
                    seen_edge_ids.add(eid)
    return dict(feature_edge_to_patches)


def build_patch_input_data_from_abs_face(
    face,
    edge_geometry_data: dict,
    V_3d: list,
    F: np.ndarray,
    u_periodic_edges: tuple,
    v_periodic_edges: tuple,
    surface_merged_edges: list = None,
) -> PatchInputData:
    """Package initial-meshing outputs and ABS topology into a PatchInputData.

    Parameters
    ----------
    face : abs Face object.  ``face.outer_loop`` gives the outer loop id.
    edge_geometry_data : dict edge.id -> (nv, 3) ndarray in True orientation.
    V_3d, F : mesh from ``run_initial_meshing_for_face``.
    u_periodic_edges, v_periodic_edges : merged seam vertex ids from the same function.
    """
    patch_loop_topology_data = {}
    patch_loop_geometry_data = {}

    # face.outer_loop is the outer (boundary) loop id.
    initial_loop_id = face.outer_loop

    for loop in face.loops:
        # Detect seam edges: same edge.id appearing more than once in the loop.
        edge_id_counts: dict = {}
        for he in loop.halfedges:
            eid = he.edge.id
            edge_id_counts[eid] = edge_id_counts.get(eid, 0) + 1
        seam_edge_ids = {eid for eid, cnt in edge_id_counts.items() if cnt > 1}

        # Build LoopTopologyData.
        loop_topology_data = LoopTopologyData()
        for he in loop.halfedges:
            P = edge_geometry_data.get(he.edge.id)
            is_degenerate = P is None or len(P) == 0
            if he.orientation_wrt_edge:
                halfedge = HalfEdge(
                    he.edge.id,
                    he.orientation_wrt_edge,
                    is_degenerate=is_degenerate,
                    start_corner_id=he.edge.start_vertex,
                    end_corner_id=he.edge.end_vertex,
                )
            else:
                halfedge = HalfEdge(
                    he.edge.id,
                    he.orientation_wrt_edge,
                    is_degenerate=is_degenerate,
                    start_corner_id=he.edge.end_vertex,
                    end_corner_id=he.edge.start_vertex,
                )
            if he.edge.id in seam_edge_ids:
                loop_topology_data.seam_ids_set.add(he.edge.id)
            loop_topology_data.halfedges.append(halfedge)
        patch_loop_topology_data[loop.id] = loop_topology_data

        # Build LoopGeometryData: feature_id -> 3-D polyline in True orientation.
        loop_geometry_data = LoopGeometryData()
        for halfedge in loop_topology_data.halfedges:
            fid = halfedge.feature_id
            if fid in loop_geometry_data.feature_id_poly_V_True_orientation:
                continue
            P = edge_geometry_data.get(fid, np.empty((0, 3)))
            loop_geometry_data.feature_id_poly_V_True_orientation[fid] = np.asarray(P)
        patch_loop_geometry_data[loop.id] = loop_geometry_data

    return PatchInputData(
        V=V_3d,
        F=F,
        initial_loop_id=initial_loop_id,
        patch_loop_topology_data=patch_loop_topology_data,
        patch_loop_geometry_data=patch_loop_geometry_data,
        u_periodic_edges=u_periodic_edges,
        v_periodic_edges=v_periodic_edges,
        surface_merged_edges=surface_merged_edges or [],
    )
