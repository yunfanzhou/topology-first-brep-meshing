import numpy as np
from typing import Optional
from operations.align import _get_boundary_edges
from operations.remesh_background import (
    mmg_remesh, compute_mmg_factor, compute_first_fundamental_form,
    compute_triangle_approximation_errors,
    refine_background_for_feature_conflicts, per_vertex_edge_length_3d,
    refine_cdt_with_size_field,
)
import pygeodesic.geodesic as geodesic
# The PyPI pygeodesic lacks surface-point tracing; without it every patch would
# silently fall back to a different mesher.
if not hasattr(geodesic.PyGeodesicAlgorithmExact,
               "geodesicDistanceWithFacesAndBarycentricFromSurfacePoints"):
    raise ImportError(
        "The installed pygeodesic is not the fork this pipeline needs. Install it with "
        "'pip install -r requirements.txt' (pygeodesic from "
        "https://github.com/yunfanzhou/pygeodesic, commit 7ce3fd8).")
import igl
from utils.print import print_if_verbose
SAME_VERTEX_TOL = 1e-8
STRIP_WIDTH_RATIO = 0.01
PARALLEL_COS_THRESH = 0.98
EDGE_LENGTH_TOL = 1e-6
# Seam-edge segments within this many eps_eff (3D) of a periodic side are the
# side itself: the traced seam drifts off it by the background mesh's chord sag,
# measured <= 1.5 eps_eff (Sphere, Torus, fillet tori at error_rel 1e-3..1e-2).
SEAM_TOL_EPS_FACTOR = 3.0
# Max samples a single collinear run may be resampled into. Guards against a
# tiny/NaN target_edge_length blowing n_seg up into an effective hang / OOM.
MAX_RESAMPLE_SEGMENTS = 100_000

TAG_REQUIRED = 1
TAG_U_PERIODIC_LB = 10
TAG_U_PERIODIC_UB = 11
TAG_V_PERIODIC_LB = 12
TAG_V_PERIODIC_UB = 13
TAG_SINGULARITY_BASE = 20
TAG_PLAIN_BOUNDARY = 30

def fix_nonmanifold_after_stitch(V_list, F_uv, F_stitched, surface):
    """Detect non-manifold edges in the stitched mesh and fix by splitting
    the corresponding UV edges, then rebuild the stitched mesh.

    Uses positional correspondence: F_uv[fid] and F_stitched[fid] have the
    same vertex layout, so the UV edge at positions (i, i+1) corresponds to
    the stitched edge at the same positions.

    Modifies V_list, F_uv, F_stitched in place.
    Returns updated V_3d array.
    """
    from utils.mesh_utils import compute_EF, create_edge

    edge_to_faces = compute_EF(F_stitched)
    nonmanifold = {e: fids for e, fids in edge_to_faces.items() if len(fids) > 2}
    if not nonmanifold:
        return surface.sample(np.array(V_list))

    uv_edges_to_split = set()
    for (a3d, b3d), fids in nonmanifold.items():
        for fid in fids:
            sf = F_stitched[fid]
            uf = F_uv[fid]
            for i in range(3):
                j = (i + 1) % 3
                if create_edge(sf[i], sf[j]) == create_edge(a3d, b3d):
                    uv_edges_to_split.add(create_edge(uf[i], uf[j]))

    for (va, vb) in uv_edges_to_split:
        print_if_verbose(f"Splitting UV edge {(va, vb)} to fix non-manifold stitched edge")
        mid = [0.5 * (V_list[va][k] + V_list[vb][k]) for k in range(len(V_list[va]))]
        mid_vid = len(V_list)
        V_list.append(mid)
        new_uv = []
        new_st = []
        for fi in range(len(F_uv)):
            uf = F_uv[fi]
            local = None
            for i in range(3):
                if create_edge(uf[i], uf[(i + 1) % 3]) == create_edge(va, vb):
                    local = i
                    break
            if local is None:
                continue
            u0, u1, u2 = uf[local], uf[(local + 1) % 3], uf[(local + 2) % 3]
            sf = F_stitched[fi]
            s0, s1, s2 = sf[local], sf[(local + 1) % 3], sf[(local + 2) % 3]
            F_uv[fi] = [u0, mid_vid, u2]
            F_stitched[fi] = [s0, mid_vid, s2]
            new_uv.append([mid_vid, u1, u2])
            new_st.append([mid_vid, s1, s2])
        F_uv.extend(new_uv)
        F_stitched.extend(new_st)

    # Remove degenerate faces created by the splits
    keep = [i for i in range(len(F_stitched)) if len(set(F_stitched[i])) == 3]
    F_uv[:] = [F_uv[i] for i in keep]
    F_stitched[:] = [F_stitched[i] for i in keep]

    return surface.sample(np.array(V_list))


def convert_3d_geodesic_segments_to_2d(geodesic_return_data, V_uv, F):
    """Convert pygeodesic face/barycentric data to UV segments via barycentric interpolation."""
    _, path_points, face_ids, face_offsets, bary_start, bary_end = geodesic_return_data
    # face_offsets has length == len(path_points); segment i goes path[i]->path[i+1]
    # faces crossed by segment i: face_ids[face_offsets[i]:face_offsets[i+1]]
    uv_segments = []
    for i in range(len(face_offsets) - 1):
        for j in range(face_offsets[i], face_offsets[i + 1]):
            face_uv = V_uv[F[face_ids[j]]]  # (3, 2)
            uv_segments.append(np.stack([
                bary_start[j] @ face_uv,
                bary_end[j] @ face_uv,
            ]))          
    return uv_segments
def _bary_3d(pt, a, b, c):
    """Barycentric coordinates of pt w.r.t. triangle (a, b, c) in 3D (Cramer's rule).
    Falls back to nearest-vertex weights if the triangle is degenerate."""
    v0, v1, v2 = b - a, c - a, pt - a
    d00, d01 = np.dot(v0, v0), np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        # Degenerate triangle: snap to the closest vertex
        dists = [np.dot(pt - a, pt - a), np.dot(pt - b, pt - b), np.dot(pt - c, pt - c)]
        bary = np.zeros(3)
        bary[int(np.argmin(dists))] = 1.0
        return bary
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.array([1.0 - v - w, v, w])


def _project_point_to_mesh(pt, V_3d, F):
    """Return (vertex_id_or_None, fid, closest, bary) for pt projected onto V_3d/F.

    vertex_id_or_None is set when the point snaps to an existing vertex.
    """
    V_np = np.array(V_3d)
    F_np = np.array(F)
    _, I, C = igl.point_mesh_squared_distance(pt.reshape(1, 3), V_np, F_np)
    fid = int(I)
    closest = C
    f0, f1, f2 = F[fid]
    a, b, c = V_np[f0], V_np[f1], V_np[f2]
    bary = _bary_3d(closest, a, b, c)
    bary = np.clip(bary, 0.0, 1.0)
    bary /= bary.sum()
    snap_idx = int(np.argmax(bary))
    if bary[snap_idx] >= 1.0 - SAME_VERTEX_TOL:
        return [f0, f1, f2][snap_idx], fid, closest, bary
    return None, fid, closest, bary



def geodesic_uv_segments_one_feature_edge(singularities,
                                           polyline,
                                           V_3d: list,
                                           V_uv: list,
                                           F_3d: list,
                                           F_uv: list,
                                           sub_domains = None):
    """Trace geodesics for each consecutive pair of polyline points.

    V_3d, V_uv, F_3d, F_uv are the base mesh lists and are NOT modified.

    Singularity side-targets are derived from the `sub_domains` bounds
    (handles both boundary padding and interior strip gaps).
    """
    sides = set()
    if sub_domains is not None:
        for sing in singularities.values():
            axis = 0 if sing['uiso'] else 1
            pt = np.asarray(sing['point2d'], dtype=float)
            raw_target = float(pt[axis])
            for sd in sub_domains:
                lb = sd[axis * 2]
                ub = sd[axis * 2 + 1]
                if raw_target <= lb + SAME_VERTEX_TOL:
                    sides.add((axis, lb))
                elif raw_target >= ub - SAME_VERTEX_TOL:
                    sides.add((axis, ub))
    side_edges = []
    for axis, side_target in sides:
        bnd_edges = _get_boundary_edges(F_uv)
        side_edges.extend([
            e for e in bnd_edges
            if abs(V_uv[e[0]][axis] - side_target) < SAME_VERTEX_TOL and abs(V_uv[e[1]][axis] - side_target) < SAME_VERTEX_TOL
        ])
    def is_near_singularity(side_edges, bary, fid, F_uv):
        if not side_edges:
            return False
        tri = np.asarray(F_uv[fid], dtype=int)
        bary = np.asarray(bary, dtype=float)
        side_edge_set = {tuple(sorted((int(a), int(b)))) for a, b in side_edges}
        side_vids = {vid for edge in side_edge_set for vid in edge}

        if np.max(bary) >= 1.0 - SAME_VERTEX_TOL:
            vid = int(tri[int(np.argmax(bary))])
            return vid in side_vids

        for i in range(3):
            if bary[i] <= SAME_VERTEX_TOL:
                edge = tuple(sorted((int(tri[(i + 1) % 3]), int(tri[(i + 2) % 3]))))
                return edge in side_edge_set
        return False

    V_3d_np = np.array(V_3d, dtype=np.float64)
    F_3d_np = np.array(F_3d, dtype=np.int32)
    V_uv_np = np.array(V_uv, dtype=np.float64)
    F_uv_np = np.array(F_uv, dtype=np.int32)
    geoalg = geodesic.PyGeodesicAlgorithmExact(V_3d_np, F_3d_np)
    all_uv_segments = []
    cached_src = None
    for k in range(len(polyline) - 1):
        if cached_src is not None:
            vid_src, fid_src, closest_src, src_bary = cached_src
        else:
            vid_src, fid_src, closest_src, src_bary = _project_point_to_mesh(polyline[k], V_3d_np, F_3d_np)

        vid_tgt, fid_tgt, closest_tgt, tgt_bary = _project_point_to_mesh(polyline[k + 1], V_3d_np, F_3d_np)
        cached_src = (vid_tgt, fid_tgt, closest_tgt, tgt_bary)

        if is_near_singularity(side_edges, src_bary, fid_src, F_uv) or is_near_singularity(side_edges, tgt_bary, fid_tgt, F_uv):
            print_if_verbose(f"segment {k}: endpoint near singularity strip, skipping")
            continue

        if np.linalg.norm(closest_src - closest_tgt) < SAME_VERTEX_TOL:
            print_if_verbose(f"segment {k}: src==tgt, skipping")
            continue

        geo_data = geoalg.geodesicDistanceWithFacesAndBarycentricFromSurfacePoints(
            fid_src, closest_src,
            fid_tgt, closest_tgt,
            vid_src=vid_src,
            vid_tgt=vid_tgt,
        )
        all_uv_segments.extend(
            convert_3d_geodesic_segments_to_2d(geo_data, V_uv_np, F_uv_np)
        )
    return all_uv_segments  # list of (2, 2) UV segments



def resample_uv_segments(uv_segments, surface, target_edge_length):
    """Resample UV segments so 3D edge lengths match the pre-CDT mesh.

    Connected segments are chained into polylines.  Each chain is resampled
    at uniform 3D arc-length intervals.  Non-collinear bends (>~11°) are
    preserved.  Collinear runs of short segments get merged; long segments
    get split.
    """
    if not uv_segments:
        return uv_segments

    # Chain consecutive segments that share endpoints
    chains = []
    chain = [uv_segments[0][0].copy(), uv_segments[0][1].copy()]
    for seg in uv_segments[1:]:
        if np.linalg.norm(seg[0] - chain[-1]) < SAME_VERTEX_TOL:
            chain.append(seg[1].copy())
        else:
            chains.append(np.array(chain))
            chain = [seg[0].copy(), seg[1].copy()]
    chains.append(np.array(chain))

    # Resample each chain
    resampled = []
    for chain_uvs in chains:
        resampled.extend(resample_chain(chain_uvs, surface, target_edge_length))
    return resampled


def resample_chain(chain_uvs, surface, target_spacing):
    """Resample one polyline chain at uniform 3D arc-length intervals.

    Vertices where the 3D direction changes by more than ~11° are kept as
    corners.  Collinear runs between corners are resampled to target_spacing
    (merging short segments, splitting long ones).
    """
    n_pts = len(chain_uvs)
    if n_pts < 2:
        return []

    if not np.isfinite(target_spacing) or target_spacing <= SAME_VERTEX_TOL:
        raise ValueError(
            f"resample_chain: target_spacing must be a finite positive length, "
            f"got {target_spacing!r}. This usually means before_embedding_edge_length "
            f"was computed as ~0 or NaN upstream."
        )

    pts_3d = surface.sample(chain_uvs)
    seg_lens = np.linalg.norm(np.diff(pts_3d, axis=0), axis=1)
    total_len = seg_lens.sum()

    if total_len < target_spacing * 0.5:
        return [np.stack([chain_uvs[0], chain_uvs[-1]])]

    # Find corners: endpoints + significant direction changes
    corners = [0]
    for i in range(1, n_pts - 1):
        d_prev = pts_3d[i] - pts_3d[i - 1]
        d_next = pts_3d[i + 1] - pts_3d[i]
        lp, ln = np.linalg.norm(d_prev), np.linalg.norm(d_next)
        if lp < 1e-15 or ln < 1e-15:
            continue
        if np.dot(d_prev, d_next) / (lp * ln) < PARALLEL_COS_THRESH:
            corners.append(i)
    corners.append(n_pts - 1)

    # Resample each collinear run between consecutive corners
    out = []
    for ci in range(len(corners) - 1):
        si, ei = corners[ci], corners[ci + 1]
        sub_uvs = chain_uvs[si:ei + 1]
        sub_3d = pts_3d[si:ei + 1]
        sub_lens = np.linalg.norm(np.diff(sub_3d, axis=0), axis=1)
        sub_total = sub_lens.sum()

        n_seg = max(1, round(sub_total / target_spacing))
        if n_seg > MAX_RESAMPLE_SEGMENTS:
            print_if_verbose(
                f"resample_chain: clamping n_seg {n_seg} -> {MAX_RESAMPLE_SEGMENTS} "
                f"(sub_total={sub_total:.3e}, target_spacing={target_spacing:.3e})"
            )
            n_seg = MAX_RESAMPLE_SEGMENTS

        if n_seg == 1 or len(sub_uvs) == 2:
            out.append(np.stack([sub_uvs[0], sub_uvs[-1]]))
        else:
            cum = np.concatenate([[0.0], np.cumsum(sub_lens)])
            sample_t = np.linspace(0, sub_total, n_seg + 1)
            sample_uvs = []
            for t in sample_t:
                idx = min(int(np.searchsorted(cum, t, side='right')) - 1,
                          len(sub_uvs) - 2)
                sl = sub_lens[idx] if sub_lens[idx] > 0 else 1e-15
                alpha = np.clip((t - cum[idx]) / sl, 0.0, 1.0)
                sample_uvs.append((1 - alpha) * sub_uvs[idx] + alpha * sub_uvs[idx + 1])
            for i in range(len(sample_uvs) - 1):
                out.append(np.stack([sample_uvs[i], sample_uvs[i + 1]]))
    return out


def deduplicate_uv_segments(uv_segments, tol=SAME_VERTEX_TOL):
    """Remove duplicate and overlapping near-collinear UV segments.

    Two segments are duplicates if both endpoints match within tol.
    A shorter segment is redundant if it is a sub-segment of a longer
    collinear segment (both endpoints lie on the longer segment).
    Same pairwise tests as a double loop, one row of pairs at a time in numpy.
    """
    if len(uv_segments) <= 1:
        return uv_segments

    segs = [np.asarray(s, dtype=float) for s in uv_segments]
    A = np.array([s[0] for s in segs])
    B = np.array([s[1] for s in segs])
    n = len(segs)

    def dist(P, q):
        return np.linalg.norm(P - q, axis=1)

    # Step 1: remove exact duplicates (both endpoints match, in either order)
    keep = np.ones(n, dtype=bool)
    for i in range(n - 1):
        if not keep[i]:
            continue
        J = i + 1 + np.flatnonzero(keep[i + 1:])
        if len(J) == 0:
            continue
        fwd = (dist(A[J], A[i]) < tol) & (dist(B[J], B[i]) < tol)
        rev = (dist(B[J], A[i]) < tol) & (dist(A[J], B[i]) < tol)
        keep[J[fwd | rev]] = False

    unique = [s for s, k in zip(segs, keep) if k]
    UA, UB = A[keep], B[keep]

    # Step 2: remove sub-segments of longer collinear segments
    def on_segment(P, a, b):
        """Per row of P: lies on segment a→b within tol."""
        ab = b - a
        ab_len = np.linalg.norm(ab)
        if ab_len < tol:
            return dist(P, a) < tol
        t = ((P - a) @ ab) / (ab_len * ab_len)
        inside = (t >= -tol / ab_len) & (t <= 1 + tol / ab_len)
        proj = a + t[:, None] * ab
        return inside & (dist(P, proj) < tol)

    L = np.linalg.norm(UB - UA, axis=1)
    keep2 = np.ones(len(unique), dtype=bool)
    for i in range(len(unique)):
        if not keep2[i]:
            continue
        J = np.flatnonzero(keep2 & (L < L[i]))
        if len(J) == 0:
            continue
        inside = on_segment(UA[J], UA[i], UB[i]) & on_segment(UB[J], UA[i], UB[i])
        keep2[J[inside]] = False

    result = [s for s, k in zip(unique, keep2) if k]
    n_removed = len(uv_segments) - len(result)
    if n_removed > 0:
        print_if_verbose(f"[dedup] removed {n_removed} duplicate/sub-segments from {len(uv_segments)}")
    return result


def embed_loops_one_patch(V_uv: list,
                          V_3d: list,
                        F_uv: list,
                        F_3d: list,
                        face,
                        edge_geometry_data: dict,
                        sub_domains,
                        before_embedding_edge_length: float,
                        padding_ratio = STRIP_WIDTH_RATIO,
                        target_error: float = 5e-2,
                        max_iterations: int = 5,
                        mmg_factor: float = None,
                        mmg2d_path: Optional[str] = None,
                        u_periodic: bool = False,
                        v_periodic: bool = False,
                        singularity_sides: dict = None,
                        interior_singularity_idx=None,
                        use_feature_conflict_refine: bool = True):
    if max_iterations < 1:  # the remesh loop below must run at least once
        raise ValueError(f"embed_loops_one_patch needs max_iterations >= 1, got {max_iterations}")

    embedded_ids = set()
    uv_segments = []
    # Seam edges: used twice by this face (both sides of the periodic cut).
    halfedge_ids = [he.edge.id for loop in face.loops for he in loop.halfedges]
    seam_ids = {e for e in halfedge_ids if halfedge_ids.count(e) > 1}
    periodic_sides = []
    if u_periodic:
        periodic_sides += [(0, min(sd[0] for sd in sub_domains)), (0, max(sd[1] for sd in sub_domains))]
    if v_periodic:
        periodic_sides += [(1, min(sd[2] for sd in sub_domains)), (1, max(sd[3] for sd in sub_domains))]
    for loop in face.loops:
        for halfedge in loop.halfedges:
            edge_id = halfedge.edge.id
            if edge_id in embedded_ids:
                continue
            print_if_verbose(f" //////// embedding edge  {edge_id}")
            if edge_id not in edge_geometry_data:
                raise ValueError(f"Missing geometry data for edge {edge_id}")
            P = edge_geometry_data[edge_id]  # (n, 3)
            if P.shape[0] < 2:
                continue
            local_uv_segments = geodesic_uv_segments_one_feature_edge(face.singularities,
                                                                     P,
                                                                     V_3d,
                                                                     V_uv,
                                                                     F_3d,
                                                                     F_uv,
                                                                     sub_domains = sub_domains)
            # Snap to sub-domain bboxes before resampling so chain lengths are accurate.
            is_seam = edge_id in seam_ids
            cleaned_uv_segments = snap_segments_to_bbox(
                local_uv_segments, sub_domains,
                surface=face.surface,
                seam_sides=periodic_sides if is_seam else None,
                seam_tol_3d=SEAM_TOL_EPS_FACTOR * target_error if is_seam else None,
            )
            uv_segments.extend(cleaned_uv_segments)
            embedded_ids.add(edge_id)

    # Deduplicate overlapping segments from geodesic tracing before resampling.
    uv_segments = deduplicate_uv_segments(uv_segments)
    # Resample UV segments so their 3D edge lengths match the pre-CDT mesh.
    uv_segments = resample_uv_segments(uv_segments, face.surface, before_embedding_edge_length)
    # Conflict-driven size field: refine a throwaway copy of the pre-embed
    # mesh until distinct feature-curve pieces stop sharing triangles, then
    # keep only its per-vertex size as a lookup field. Applied inside the
    # post-CDT mmg loop below (the CDT itself takes no interior points), so
    # tracing gets one element strip per feature. Runs on the coarse mesh —
    # the expensive geodesic tracing above is untouched.
    size_field = None
    if use_feature_conflict_refine:
        try:
            V_bg, F_bg = refine_background_for_feature_conflicts(
                np.asarray(V_uv, dtype=float), np.asarray(F_uv, dtype=int),
                face, edge_geometry_data, mmg2d_path=mmg2d_path,
                periodic_sides=periodic_sides)
            from scipy.spatial import cKDTree
            # Looked up by 3D position: the two sides of a periodic seam are
            # the same surface points, so a query on either side sees the
            # refinement even when conflict refinement landed on one side only.
            V_bg_3d = face.surface.sample(np.asarray(V_bg, dtype=float))
            h_bg = per_vertex_edge_length_3d(V_bg_3d, np.asarray(F_bg, dtype=int))
            size_field = (cKDTree(V_bg_3d), h_bg)
        except TimeoutError:
            raise  # alarm_timeout fired: let it reach its caller
        except Exception as e:
            print_if_verbose(f"[conflict-refine] size field disabled for this patch: {e}")
    V_cdt, F_cdt, constrained_edges, bbox_edges_by_side = embed_geodesic_segments_in_uv(
        uv_segments, sub_domains,
        u_periodic=u_periodic,
        v_periodic=v_periodic,
        singularity_sides=singularity_sides,
        interior_singularity_idx=interior_singularity_idx,
    )
    bbox_tags = set(bbox_edges_by_side.keys())

    # Insertion-only Triangle refine of the CDT toward the conflict size
    # field: subdivide the embedded feature edges at 3D arc-length spacing
    # and conform their surroundings, BEFORE they become frozen MMG
    # RequiredEdges below. Without this the required edges keep their coarse
    # length while the size field refines everything around them, leaving
    # anisotropic strips along the features. Refine mode never collapses or
    # removes a constrained edge. All boundary edges are passed with their
    # tags — Triangle labels invented boundary segments with marker 1, which
    # would collide with TAG_REQUIRED.
    if size_field is not None and constrained_edges is not None and len(constrained_edges) > 0:
        try:
            segs_in, mks_in = [], []
            for e in constrained_edges:
                segs_in.append((int(e[0]), int(e[1])))
                mks_in.append(TAG_REQUIRED)
            for tag, arr in bbox_edges_by_side.items():
                for e in arr:
                    segs_in.append((int(e[0]), int(e[1])))
                    mks_in.append(int(tag))
            nv0, nf0 = len(V_cdt), len(F_cdt)
            V_cdt, F_cdt, segs_out, mks_out = refine_cdt_with_size_field(
                V_cdt, F_cdt, segs_in, mks_in, {TAG_REQUIRED},
                face.surface, size_field)
            constrained_edges = segs_out[mks_out == TAG_REQUIRED]
            bbox_edges_by_side = {
                int(t): segs_out[mks_out == int(t)]
                for t in np.unique(mks_out) if int(t) in bbox_tags}
            print_if_verbose(
                f"[conflict-refine] CDT refine: {nv0}v/{nf0}f -> "
                f"{len(V_cdt)}v/{len(F_cdt)}f, "
                f"{len(constrained_edges)} required edges")
        except TimeoutError:
            raise  # alarm_timeout fired: let it reach its caller
        except Exception as e:
            print_if_verbose(f"[conflict-refine] CDT refine pass skipped: {e}")

    req_edges = [(int(e[0]) + 1, int(e[1]) + 1) for e in constrained_edges] if constrained_edges is not None else []
    feat_edges = {}
    for tag, edges_arr in bbox_edges_by_side.items():
        feat_edges[tag] = [(int(e[0]) + 1, int(e[1]) + 1) for e in edges_arr]
    feat_edges = feat_edges if feat_edges else None

    V_cur, F_cur = V_cdt, F_cdt
    if mmg_factor is None:
        mmg_factor = compute_mmg_factor(face.surface, V_cur)
    resampled = False
    for it in range(max_iterations):
        print_if_verbose(f"After emebedding MMG iteration {it}, factor={mmg_factor:.3e}")
        EE, FF_m, GG = compute_first_fundamental_form(face.surface, V_cur)
        base_factor = mmg_factor
        if size_field is not None:
            # Cap the local target size by the conflict-refined field:
            # factor = 1/h^2, so max() takes the finer of the two. The
            # uniform base still governs everywhere the field is coarse.
            tree_bg, h_bg = size_field
            _, nn = tree_bg.query(face.surface.sample(np.asarray(V_cur, dtype=float)))
            factor_v = np.maximum(1.0 / np.maximum(h_bg[nn], 1e-12) ** 2,
                                  base_factor)
            EE, FF_m, GG = EE * factor_v, FF_m * factor_v, GG * factor_v
            base_factor = 1.0
        V_cur, F_cur, tagged_edges = mmg_remesh(
            V_cur, F_cur,
            EE=EE, FF_m=FF_m, GG=GG,
            metric_factor=base_factor,
            hgrad=-1,
            required_edges=req_edges,
            feature_edges=feat_edges,
            mmg2d_path=mmg2d_path,
            nosizreq=True,
        )
        req_edges = [(v0 + 1, v1 + 1) for v0, v1 in tagged_edges.get(TAG_REQUIRED, [])]
        feat_edges = {}
        for tag in bbox_tags:
            out = tagged_edges.get(tag, [])
            if out:
                feat_edges[tag] = [(v0 + 1, v1 + 1) for v0, v1 in out]
        feat_edges = feat_edges if feat_edges else None
        # Let degenerate-UV-triangle exceptions propagate to the caller in
        # initial_meshing.run_initial_meshing_for_face — its outer try/except
        # routes failures to the insert_feature_edges_uv fallback, which
        # doesn't need tagged_edges and starts from the pre-embedding mesh.
        tri_errors = compute_triangle_approximation_errors(face.surface, V_cur, F_cur)
        avg_error = float(np.average(tri_errors))
        if avg_error > target_error:
            if not resampled:
                # mmg only knows the metric at its input vertices. The CDT can be
                # sparse (a featureless face: just the domain corners), so the first
                # pass can miss with a correct size request. Rerun once at the same
                # factor, from this pass's output, before refining.
                resampled = True
                continue
            mmg_factor *= 4
        else:
            break
        
    return V_cur, F_cur, tagged_edges

def snap_segments_to_bbox(uv_segments,
                          sub_domains,
                          surface=None,
                          seam_sides=None, seam_tol_3d=None):
    """Drop segments that are very close to a sub-domain boundary or fall in a gap.

    A segment is dropped if both endpoints are within EDGE_LENGTH_TOL of the
    same edge of any sub-domain, or if either endpoint falls outside the union
    of sub-domains (i.e., into a gap or beyond the global bbox). Zero-length
    segments are also dropped.

    Seam edges (seam_sides given): the periodic sides already represent the
    seam, and the traced seam drifts off them by the chord sag of the
    background mesh (~eps_eff in 3D, not a fixed fraction of the span). A seam
    segment is also dropped when both endpoints are within seam_tol_3d, in 3D,
    of the same periodic side: |S(u, v) - S(side, v)|. A seam lying inside the
    patch is far from the sides and stays embedded.

    seam_sides : list of (axis, value) periodic domain sides, or None.
    seam_tol_3d : 3D distance bound for seam segments (with seam_sides).

    Returns
    -------
    filtered_segments : list of (2, 2) arrays
    """
    u_l_g = min(sd[0] for sd in sub_domains)
    u_u_g = max(sd[1] for sd in sub_domains)
    v_l_g = min(sd[2] for sd in sub_domains)
    v_u_g = max(sd[3] for sd in sub_domains)
    span = [u_u_g - u_l_g, v_u_g - v_l_g]

    boundaries = []
    for sd in sub_domains:
        boundaries.append((0, sd[0], span[1] * EDGE_LENGTH_TOL))
        boundaries.append((0, sd[1], span[1] * EDGE_LENGTH_TOL))
        boundaries.append((1, sd[2], span[0] * EDGE_LENGTH_TOL))
        boundaries.append((1, sd[3], span[0] * EDGE_LENGTH_TOL))

    def near_same_boundary(p0, p1):
        for axis, val, tol in boundaries:
            if abs(p0[axis] - val) < tol and abs(p1[axis] - val) < tol:
                return True
        return False

    def snap_to_boundary(p):
        for axis, val, tol in boundaries:
            if abs(p[axis] - val) < tol:
                p[axis] = val

    def outside_boundary(p):
        # Outside if not inside any sub-domain (catches gaps too)
        for sd in sub_domains:
            if (sd[0] - SAME_VERTEX_TOL <= p[0] <= sd[1] + SAME_VERTEX_TOL
                and sd[2] - SAME_VERTEX_TOL <= p[1] <= sd[3] + SAME_VERTEX_TOL):
                return False
        return True

    def near_same_seam_side(p0, p1):
        P = np.array([p0, p1], dtype=float)
        S_P = surface.sample(P)
        for axis, val in seam_sides:
            Q = P.copy()
            Q[:, axis] = val
            if np.all(np.linalg.norm(S_P - surface.sample(Q), axis=1) <= seam_tol_3d):
                return True
        return False

    check_seam = bool(seam_sides) and seam_tol_3d is not None and surface is not None
    filtered = []
    for seg in uv_segments:
        if np.linalg.norm(seg[0] - seg[1]) < EDGE_LENGTH_TOL:
            continue
        if near_same_boundary(seg[0], seg[1]):
            continue
        if check_seam and near_same_seam_side(seg[0], seg[1]):
            continue
        snap_to_boundary(seg[0])
        snap_to_boundary(seg[1])
        if outside_boundary(seg[0]) or outside_boundary(seg[1]):
            continue

        filtered.append(seg)
    return filtered


def merge_duplicate_cdt_vertices(V, F, edges, markers, tol):
    """Merge vertices closer than *tol*, update F/edges, drop degenerate faces/edges.

    Returns (V_new, F_new, edges_new, markers_new).
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    n = len(V)

    # Build remap: each vertex maps to the lowest-index vertex within tol
    remap = np.arange(n, dtype=np.int32)
    for i in range(n):
        if remap[i] != i:
            continue  # already merged into an earlier vertex
        diffs = np.abs(V[i + 1:] - V[i])
        dups = np.where((diffs[:, 0] < tol) & (diffs[:, 1] < tol))[0] + (i + 1)
        for j in dups:
            if remap[j] == j:  # not yet merged
                remap[j] = i

    n_merged = int(np.sum(remap != np.arange(n)))
    if n_merged == 0:
        return V, F, edges, markers

    print_if_verbose(f"[CDT] Merged {n_merged} duplicate vertices (tol={tol:.2e})")

    # Compact vertex array
    keep_mask = remap == np.arange(n)
    old_to_new = np.full(n, -1, dtype=np.int32)
    old_to_new[keep_mask] = np.arange(int(keep_mask.sum()), dtype=np.int32)
    # For merged vertices, follow the remap chain then translate
    for i in range(n):
        if not keep_mask[i]:
            old_to_new[i] = old_to_new[remap[i]]
    V_new = V[keep_mask]

    # Remap faces and drop degenerate (repeated-vertex) triangles
    F_new = old_to_new[F]
    valid_faces = (F_new[:, 0] != F_new[:, 1]) & (F_new[:, 1] != F_new[:, 2]) & (F_new[:, 2] != F_new[:, 0])
    n_degen = int((~valid_faces).sum())
    if n_degen > 0:
        print_if_verbose(f"[CDT] Removed {n_degen} degenerate faces after vertex merge")
    F_new = F_new[valid_faces]

    # Remap edges and drop collapsed ones
    if len(edges) > 0:
        edges = np.asarray(edges, dtype=np.int32)
        edges_new = old_to_new[edges]
        valid_edges = edges_new[:, 0] != edges_new[:, 1]
        edges_new = edges_new[valid_edges]
        markers = np.asarray(markers)[valid_edges]
    else:
        edges_new = edges

    return V_new, F_new, edges_new, markers


def _orient2d_sign(p, q, r):
    """Exact sign of orient2d(p, q, r): +1 / -1 / 0.

    Float fast path with a forward error bound; exact rational arithmetic
    (doubles are exactly representable as Fractions) when the float result
    cannot be certified. No tolerance anywhere.
    """
    acx, acy = p[0] - r[0], p[1] - r[1]
    bcx, bcy = q[0] - r[0], q[1] - r[1]
    det = acx * bcy - acy * bcx
    errbound = 3.3306690738754716e-16 * (abs(acx * bcy) + abs(acy * bcx))
    if det > errbound:
        return 1
    if det < -errbound:
        return -1
    from fractions import Fraction as Fr
    d = ((Fr(p[0]) - Fr(r[0])) * (Fr(q[1]) - Fr(r[1]))
         - (Fr(p[1]) - Fr(r[1])) * (Fr(q[0]) - Fr(r[0])))
    return (d > 0) - (d < 0)


def _on_segment_interior_exact(a, b, v):
    """Exact test: is v strictly inside segment [a, b]? Assumes collinear."""
    from fractions import Fraction as Fr
    ax, ay, bx, by, vx, vy = map(Fr, (a[0], a[1], b[0], b[1], v[0], v[1]))
    t_num = (vx - ax) * (bx - ax) + (vy - ay) * (by - ay)
    l2 = (bx - ax) ** 2 + (by - ay) ** 2
    return 0 < t_num < l2


def pslg_invalid_incidence(V, segments):
    """Detect input that violates Triangle's PSLG contract, exactly.

    Triangle allows segments to intersect only at shared endpoints.
    Independently traced feature edges can violate this — a vertex of one
    trace landing on another trace's chord, or two chords properly crossing
    (e.g. both traces riding the same background-mesh edge). On
    such input Triangle's segment-insertion walk (finddirection /
    scoutsegment) either errors or cycles forever, depending on which branch
    the exact-zero predicates select. The caller uses this tripwire to fail
    the embedding immediately (routing to the insert_feature_edges_uv
    fallback) instead of gambling on undefined behavior.

    All decisions use exact predicates (filtered orient2d + rational
    betweenness); the bbox prefilter uses exact float comparisons. Near-miss
    configurations that are not exactly incident are legal PSLG input and
    pass. Returns None, or a dict describing the first offending incidence.
    """
    V = np.asarray(V, dtype=float)
    S = np.asarray(segments, dtype=int)
    if len(S) == 0:
        return None
    A, B = V[S[:, 0]], V[S[:, 1]]
    lo, hi = np.minimum(A, B), np.maximum(A, B)

    for si in range(len(S)):
        js = np.arange(si + 1, len(S))
        if len(js) == 0:
            break
        sep = ((lo[js, 0] > hi[si, 0]) | (hi[js, 0] < lo[si, 0]) |
               (lo[js, 1] > hi[si, 1]) | (hi[js, 1] < lo[si, 1]))
        for sj in js[~sep]:
            shared = set(S[si]).intersection(S[sj])
            p1, q1 = A[si], B[si]
            p2, q2 = A[sj], B[sj]
            o1 = _orient2d_sign(p1, q1, p2)
            o2 = _orient2d_sign(p1, q1, q2)
            o3 = _orient2d_sign(p2, q2, p1)
            o4 = _orient2d_sign(p2, q2, q1)
            if shared:
                # sharing an endpoint is legal; still catch the case where
                # the far endpoint of one lies exactly inside the other
                for (o, seg_pts, v, vid) in ((o1, (p1, q1), p2, S[sj, 0]),
                                             (o2, (p1, q1), q2, S[sj, 1]),
                                             (o3, (p2, q2), p1, S[si, 0]),
                                             (o4, (p2, q2), q1, S[si, 1])):
                    if int(vid) not in shared and o == 0 and \
                            _on_segment_interior_exact(seg_pts[0], seg_pts[1], v):
                        return dict(kind='endpoint_in_interior',
                                    segments=(int(si), int(sj)), vertex=int(vid))
                continue
            if o1 * o2 < 0 and o3 * o4 < 0:
                return dict(kind='segments_cross', segments=(int(si), int(sj)),
                            seg_a=tuple(map(int, S[si])), seg_b=tuple(map(int, S[sj])))
            for (o, seg_pts, v, vid) in ((o1, (p1, q1), p2, S[sj, 0]),
                                         (o2, (p1, q1), q2, S[sj, 1]),
                                         (o3, (p2, q2), p1, S[si, 0]),
                                         (o4, (p2, q2), q1, S[si, 1])):
                if o == 0 and _on_segment_interior_exact(seg_pts[0], seg_pts[1], v):
                    return dict(kind='vertex_on_segment',
                                segments=(int(si), int(sj)), vertex=int(vid))
    return None


def embed_geodesic_segments_in_uv(
    uv_segments,
    sub_domains,
    u_periodic=False,
    v_periodic=False,
    singularity_sides=None,
    interior_singularity_idx=None,
):
    """Triangulate the union of sub-domain rectangles with required UV segments.

    sub_domains : list of (u_l, u_u, v_l, v_u) tuples. The triangulation covers
        their union. Gaps between sub-domains become holes (excluded from the mesh).
        Sides of each sub-domain that face a gap (i.e., not on the global outer
        bbox) are tagged with the interior singularity tag.
    interior_singularity_idx : int or None. Singularity index used to tag
        gap-facing sub-domain sides.
    """
    import triangle as tr
    if singularity_sides is None:
        singularity_sides = {}

    u_l_g = min(sd[0] for sd in sub_domains)
    u_u_g = max(sd[1] for sd in sub_domains)
    v_l_g = min(sd[2] for sd in sub_domains)
    v_u_g = max(sd[3] for sd in sub_domains)
    span = [u_u_g - u_l_g, v_u_g - v_l_g]
    merge_tol = EDGE_LENGTH_TOL * max(span[0], span[1])

    snap_vals_u = sorted({sd[0] for sd in sub_domains} | {sd[1] for sd in sub_domains})
    snap_vals_v = sorted({sd[2] for sd in sub_domains} | {sd[3] for sd in sub_domains})

    all_verts = []

    def find_or_add(pt):
        x, y = float(pt[0]), float(pt[1])
        for val in snap_vals_u:
            if abs(x - val) < SAME_VERTEX_TOL:
                x = val
                break
        for val in snap_vals_v:
            if abs(y - val) < SAME_VERTEX_TOL:
                y = val
                break
        for j, v in enumerate(all_verts):
            if abs(v[0] - x) < merge_tol and abs(v[1] - y) < merge_tol:
                return j
        idx = len(all_verts)
        all_verts.append([x, y])
        return idx

    # Pre-add sub-domain corners so they exist as canonical vertices
    sd_corner_ids = []
    for sd in sub_domains:
        sd_ul, sd_uu, sd_vl, sd_vu = sd
        c_bl = find_or_add([sd_ul, sd_vl])
        c_br = find_or_add([sd_uu, sd_vl])
        c_tr = find_or_add([sd_uu, sd_vu])
        c_tl = find_or_add([sd_ul, sd_vu])
        sd_corner_ids.append((c_bl, c_br, c_tr, c_tl))

    boundaries_list = []
    for sd in sub_domains:
        boundaries_list.append((0, sd[0], span[1] * EDGE_LENGTH_TOL))
        boundaries_list.append((0, sd[1], span[1] * EDGE_LENGTH_TOL))
        boundaries_list.append((1, sd[2], span[0] * EDGE_LENGTH_TOL))
        boundaries_list.append((1, sd[3], span[0] * EDGE_LENGTH_TOL))

    def near_same_boundary(p0, p1):
        for axis, val, tol in boundaries_list:
            if abs(p0[axis] - val) < tol and abs(p1[axis] - val) < tol:
                return True
        return False

    segs = []
    seg_markers = []
    for seg in uv_segments:
        i0 = find_or_add(seg[0])
        i1 = find_or_add(seg[1])

        if near_same_boundary(all_verts[i0], all_verts[i1]):
            continue
        if i0 != i1:
            segs.append([i0, i1])
            seg_markers.append(TAG_REQUIRED)

    def outer_side_tag(side):
        if side in singularity_sides:
            return TAG_SINGULARITY_BASE + singularity_sides[side]
        if side == 'left' and u_periodic: return TAG_U_PERIODIC_LB
        if side == 'right' and u_periodic: return TAG_U_PERIODIC_UB
        if side == 'bottom' and v_periodic: return TAG_V_PERIODIC_LB
        if side == 'top' and v_periodic: return TAG_V_PERIODIC_UB
        return TAG_PLAIN_BOUNDARY

    interior_tag = (TAG_SINGULARITY_BASE + interior_singularity_idx
                    if interior_singularity_idx is not None else TAG_PLAIN_BOUNDARY)

    def side_tag_for_subdomain(sd, side):
        if side == 'left':   is_outer = abs(sd[0] - u_l_g) < SAME_VERTEX_TOL
        elif side == 'right': is_outer = abs(sd[1] - u_u_g) < SAME_VERTEX_TOL
        elif side == 'bottom': is_outer = abs(sd[2] - v_l_g) < SAME_VERTEX_TOL
        else:                 is_outer = abs(sd[3] - v_u_g) < SAME_VERTEX_TOL
        return outer_side_tag(side) if is_outer else interior_tag

    def split_side_at_vertices(ca, cb):
        """Chain ca -> ... -> cb through every vertex lying exactly on the side.

        A feature vertex snapped onto a side (e.g. a loop tangent to its
        tight trim bbox) would otherwise sit in the interior of the
        corner-to-corner side segment, which violates Triangle's PSLG
        contract. Sides are axis-aligned and snapping writes the exact side
        value, so exact comparison matches pslg_invalid_incidence's test.
        """
        a, b = all_verts[ca], all_verts[cb]
        axis = 0 if a[0] == b[0] else 1   # coordinate constant along the side
        t = 1 - axis
        lo, hi = min(a[t], b[t]), max(a[t], b[t])
        on_side = [j for j, v in enumerate(all_verts)
                   if v[axis] == a[axis] and lo < v[t] < hi]
        on_side.sort(key=lambda j: all_verts[j][t], reverse=a[t] > b[t])
        chain = [ca] + on_side + [cb]
        return list(zip(chain[:-1], chain[1:]))

    bbox_tag_set = set()
    for sd, (c_bl, c_br, c_tr, c_tl) in zip(sub_domains, sd_corner_ids):
        for (ca, cb), side_key in [((c_bl, c_br), 'bottom'),
                                   ((c_br, c_tr), 'right'),
                                   ((c_tr, c_tl), 'top'),
                                   ((c_tl, c_bl), 'left')]:
            tag = side_tag_for_subdomain(sd, side_key)
            for i0, i1 in split_side_at_vertices(ca, cb):
                segs.append([i0, i1])
                seg_markers.append(tag)
            bbox_tag_set.add(tag)

    # One hole point per gap between any pair of non-overlapping sub-domains
    holes = []
    for i in range(len(sub_domains)):
        for j in range(i + 1, len(sub_domains)):
            sd_i, sd_j = sub_domains[i], sub_domains[j]
            if sd_i[1] < sd_j[0] - SAME_VERTEX_TOL:
                holes.append([0.5 * (sd_i[1] + sd_j[0]),
                              0.5 * (max(sd_i[2], sd_j[2]) + min(sd_i[3], sd_j[3]))])
            elif sd_j[1] < sd_i[0] - SAME_VERTEX_TOL:
                holes.append([0.5 * (sd_j[1] + sd_i[0]),
                              0.5 * (max(sd_i[2], sd_j[2]) + min(sd_i[3], sd_j[3]))])
            elif sd_i[3] < sd_j[2] - SAME_VERTEX_TOL:
                holes.append([0.5 * (max(sd_i[0], sd_j[0]) + min(sd_i[1], sd_j[1])),
                              0.5 * (sd_i[3] + sd_j[2])])
            elif sd_j[3] < sd_i[2] - SAME_VERTEX_TOL:
                holes.append([0.5 * (max(sd_i[0], sd_j[0]) + min(sd_i[1], sd_j[1])),
                              0.5 * (sd_j[3] + sd_i[2])])

    flags = 'pe'

    verts_arr = np.array(all_verts, dtype=np.float64)
    tri_input = {'vertices': verts_arr}
    if segs:
        tri_input['segments'] = np.array(segs, dtype=np.int32)
        tri_input['segment_markers'] = np.array(seg_markers, dtype=np.int32).reshape(-1, 1)
    if holes:
        tri_input['holes'] = np.array(holes, dtype=np.float64)

    if segs:
        bad = pslg_invalid_incidence(verts_arr, tri_input['segments'])
        if bad is not None:
            # Invalid PSLG: calling Triangle on it is undefined behavior
            # (error, garbage, or an infinite scoutsegment walk). Fail the
            # embedding now; the caller's except routes to the
            # insert_feature_edges_uv fallback, same as a Triangle failure.
            raise Exception(
                f"Embedding produced a PSLG that violates Triangle's input "
                f"contract (segments may only meet at shared endpoints): {bad}")

    import sys, os
    old_stderr_fd = os.dup(2)
    stderr_pipe_r, stderr_pipe_w = os.pipe()
    os.dup2(stderr_pipe_w, 2)
    os.close(stderr_pipe_w)
    try:
        result = tr.triangulate(
            tri_input,
            flags,
        )
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        triangle_stderr = os.read(stderr_pipe_r, 65536).decode(errors='replace')
        os.close(stderr_pipe_r)
        if triangle_stderr.strip():
            print(f"[triangle] stderr:\n{triangle_stderr}", file=sys.stderr)

    if 'triangles' not in result:
        raise Exception(
            f"Triangle returned no triangulation (keys={list(result.keys())}). "
            f"Triangle stderr: {triangle_stderr!r}."
        )

    V = result['vertices']
    F = result['triangles']
    raw_edges = np.asarray(result.get('edges', np.empty((0, 2), dtype=np.int32)), dtype=np.int32)
    raw_markers = np.asarray(result.get('edge_markers', np.empty(0, dtype=np.int32))).reshape(-1)

    # Merge duplicate Steiner points that Triangle inserts at near-intersections
    V, F, raw_edges, raw_markers = merge_duplicate_cdt_vertices(
        V, F, raw_edges, raw_markers, SAME_VERTEX_TOL,
    )

    constrained_edges = None
    bbox_edges_by_side = {}
    if len(raw_edges) > 0:
        mask1 = raw_markers == 1
        if np.any(mask1):
            constrained_edges = raw_edges[mask1]
        for tag in bbox_tag_set:
            mask = raw_markers == tag
            if np.any(mask):
                bbox_edges_by_side[tag] = raw_edges[mask]
    return V, F, constrained_edges, bbox_edges_by_side
