from operations.internal_operations import split_edge_in_mesh_overloaded
from utils.mesh_utils import create_edge
from utils.print import print_if_verbose
import igl
import utils.config as config
import numpy as np
BOUNDARY_TOL = 1e-3
SAME_VERTEX_TOL = 1e-8
def cumulative_lengths(connectivity, segment_vids):
    '''Helper function. Getting the cumulated length up until point i in the list of segment_vids'''
    cum_lengths = [0.0]
    for i in range(1, len(segment_vids)):
        length = connectivity.edge_set_to_length_mapping[create_edge(segment_vids[i - 1], segment_vids[i])]
        cum_lengths.append(cum_lengths[-1] + length)
    return cum_lengths
def align_stitching_halfedges_same_patch(V,
                                        connectivity,
                                        halfedge_tracing_data0,
                                        halfedge_tracing_data1):
    '''
    We require that the halfedge0 is in True orientation, and halfedge1 is in False orientation.
    '''
    traced_vids0 = halfedge_tracing_data0.traced_vids
    traced_vids1 = halfedge_tracing_data1.traced_vids[::-1]  # reverse the order
    segment_sizes0 = halfedge_tracing_data0.segment_upsample_size
    segment_sizes1 = halfedge_tracing_data1.segment_upsample_size[::-1]

    prev_index0 = 0
    prev_index1 = 0
    next_index0 = 0
    next_index1 = 0

    after_align_vids_A = []
    after_align_vids_B = []
    for i in range(len(segment_sizes0)):
        seg_size0 = segment_sizes0[i]
        seg_size1 = segment_sizes1[i]
        # 
        # |_,_,_|_,_|
        # 0 1 2 3 4 5
        # we want 0 to 3 including 3, so we want 0 to 4 for list slicing,
        # and since the upsample size is 2, so we want (0 + upsample_size + 1 + 1)

        next_index0 = prev_index0 + seg_size0 + 1 + 1
        next_index1 = prev_index1 + seg_size1 + 1 + 1
        traced_seg_vids0 = traced_vids0[prev_index0:next_index0]
        traced_seg_vids1 = traced_vids1[prev_index1:next_index1]
        (after_align_vids_A_one_segment, 
        after_align_vids_B_one_segment) = align_both_sides_of_one_segment_of_one_patch(V, 
                                                                                       connectivity, 
                                                                                       traced_seg_vids0, 
                                                                                       traced_seg_vids1)
        prev_index0 = next_index0 - 1
        prev_index1 = next_index1 - 1
        after_align_vids_A.extend(after_align_vids_A_one_segment[:-1])
        after_align_vids_B.extend(after_align_vids_B_one_segment[:-1])
        segment_sizes0[i] = len(after_align_vids_A_one_segment) - 2
        segment_sizes1[i] = len(after_align_vids_B_one_segment) - 2
        if i == len(segment_sizes0) - 1:
            after_align_vids_A.append(after_align_vids_A_one_segment[-1])
            after_align_vids_B.append(after_align_vids_B_one_segment[-1])

    # now we flip the order of halfedge1
    after_align_vids_B = after_align_vids_B[::-1]
    segment_sizes1 = segment_sizes1[::-1]

    halfedge_tracing_data0.traced_vids = after_align_vids_A
    halfedge_tracing_data1.traced_vids = after_align_vids_B
    halfedge_tracing_data0.segment_upsample_size = segment_sizes0
    halfedge_tracing_data1.segment_upsample_size = segment_sizes1

def align_both_sides_of_one_segment_of_one_patch(V, 
                                                 connectivity, 
                                                 segment_vids_A, 
                                                 segment_vids_B):
    """ Given the two segments of the feature edge, we are aligning so that they have equal number of vertices.
      But we are not stitching them to the same positions.
      We will use the ratio on each segment as parameters to align them. 
    """

    # Compute cumulative lengths
    cum_A = cumulative_lengths(connectivity, segment_vids_A)
    cum_B = cumulative_lengths(connectivity, segment_vids_B)
    total_A = cum_A[-1]
    total_B = cum_B[-1]

    # Normalize to parameter space [0,1]
    norm_param_A = [x / total_A for x in cum_A]
    norm_param_B = [x / total_B for x in cum_B]

    # just keep a queue of longest edges of the side to be split, And split the longest edge in halves.
    # insert the new vid to the correct place in after align 
    i_A = 1
    i_B = 1
    prev_vid_A = segment_vids_A[i_A - 1]
    prev_vid_B = segment_vids_B[i_B - 1]
    after_align_vids_A = [prev_vid_A]
    after_align_vids_B = [prev_vid_B]

    while True:
        if i_A == len(segment_vids_A) or i_B == len(segment_vids_B):
            break
        vid_A = segment_vids_A[i_A]
        param_A = norm_param_A[i_A]
        # find the distance between the two points
        vid_B = segment_vids_B[i_B]
        param_B = norm_param_B[i_B]
        if param_A < param_B:
            split_edge_in_mesh_overloaded(V,
                                          connectivity,
                                          (prev_vid_B, vid_B))
            
            prev_vid_B = len(V) - 1            
            prev_vid_A = vid_A
            i_A += 1
        elif param_A > param_B:
            split_edge_in_mesh_overloaded(V,
                                          connectivity,
                                          (prev_vid_A, vid_A))
            
            prev_vid_A = len(V) - 1
            prev_vid_B = vid_B
            i_B += 1
        else:
            # the two edges are equal, we can skip this edge
            prev_vid_A = vid_A
            prev_vid_B = vid_B
            i_A += 1
            i_B += 1

        after_align_vids_A.append(prev_vid_A)
        after_align_vids_B.append(prev_vid_B)
        if len(after_align_vids_A) != len(after_align_vids_B):
            raise ValueError(f"Error: the number of vertices on both sides are not equal after align. {len(after_align_vids_A)} != {len(after_align_vids_B)}")
    return after_align_vids_A, after_align_vids_B  

def align_one_segment_of_patches(Vs,
                                 connectivities,
                                 segments: list[list[int]]):
    ''' Align one segment of multiple patches together.
    Note we could also handle more than two patches stitches at one edge case.
    The input model is not a manifold in this case.
    Input:
        segments: list of list of vids corresponding to same edge segment embedded on different patches.
    '''
    ALIGN_SNAP_TOL_REL = 1e-2  # tolerance in normalized [0,1] parameter space for snapping cross-segment vertices
    # get cumulated lengths for all segments
    norm_param_list = []
    for i in range(len(segments)):
        cum_length = cumulative_lengths(connectivities[i], segments[i])
        total_length = cum_length[-1]
        norm_param = [x / total_length for k, x in enumerate(cum_length) if k != 0 and k != len(cum_length) - 1]
        norm_param_list.append(norm_param)

    # flat sorted list of (norm_param, segment_index, vid_index_in_norm_param_list)
    flat = [(norm_param_list[i][j], i, j)
            for i, nps in enumerate(norm_param_list)
            for j in range(len(nps))]
    flat.sort(key=lambda x: x[0])

    # Group consecutive close entries from DIFFERENT segments.
    # Within one group each segment may appear at most once (no same-segment merging).
    groups = []  # list of list of (p, seg_i, vid_j)
    if flat:
        current_group = [flat[0]]
        current_segs = {flat[0][1]}
        for k in range(1, len(flat)):
            p, seg_i, vid_j = flat[k]
            if p - current_group[0][0] <= ALIGN_SNAP_TOL_REL and seg_i not in current_segs:
                current_group.append(flat[k])
                current_segs.add(seg_i)
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [flat[k]]
                current_segs = {seg_i}
        if len(current_group) >= 2:
            groups.append(current_group)

    # Compute centroid for each group and snap all member vertices to it.
    group_membership = {}  # (seg_i, vid_j) -> group_index
    group_centroids = []
    for g_id, group in enumerate(groups):
        positions = [Vs[seg_i][segments[seg_i][vid_j + 1]] for _, seg_i, vid_j in group]
        centroid = np.mean(positions, axis=0)
        group_centroids.append(centroid)
        for _, seg_i, vid_j in group:
            group_membership[(seg_i, vid_j)] = g_id

    for (seg_i, vid_j), g_id in group_membership.items():
        Vs[seg_i][segments[seg_i][vid_j + 1]] = group_centroids[g_id]

    # Process each segment: walk the flat list in sorted order.
    # For own vertices: advance the pointer.
    # For other-segment vertices:
    #   - not in any group: inject their 3D position into the current mesh edge.
    #   - in a group that includes segment i's own vertex: skip (own vertex already snapped).
    #   - in a group that does NOT include segment i: inject centroid once for the group.
    after_align_segments = []
    for i, segment in enumerate(segments):
        connectivity = connectivities[i]
        V = Vs[i]
        prev_vid = segment[0]
        after_align_vids = [prev_vid]
        # own_ptr: how many own interior vertices have been processed.
        # segment[own_ptr + 1] is always the next unprocessed own vertex (or the last segment vertex).
        own_ptr = 0
        processed_groups = set()

        for p, seg_j, vid_k in flat:
            if seg_j == i:
                # Own interior vertex — advance
                vid = segment[vid_k + 1]
                after_align_vids.append(vid)
                prev_vid = vid
                own_ptr = vid_k + 1
            else:
                # Other segment's interior vertex
                next_own_vid = segment[own_ptr + 1]
                g_id = group_membership.get((seg_j, vid_k))

                if g_id is None:
                    # Not snapped — inject its current 3D position
                    new_point = Vs[seg_j][segments[seg_j][vid_k + 1]]
                    split_edge_in_mesh_overloaded(V, connectivity,
                                                  (prev_vid, next_own_vid),
                                                  new_point=new_point)
                    new_vid = len(V) - 1
                    after_align_vids.append(new_vid)
                    prev_vid = new_vid
                else:
                    group_segs = {s for _, s, _ in groups[g_id]}
                    if i in group_segs:
                        # Segment i has its own vertex in this group — skip injection;
                        # the own vertex will be appended when we reach it in the flat list.
                        pass
                    elif g_id not in processed_groups:
                        # Segment i not in group — inject centroid once
                        split_edge_in_mesh_overloaded(V, connectivity,
                                                      (prev_vid, next_own_vid),
                                                      new_point=group_centroids[g_id])
                        new_vid = len(V) - 1
                        after_align_vids.append(new_vid)
                        prev_vid = new_vid
                        processed_groups.add(g_id)
                    # else: centroid already injected for this group — skip

        after_align_segments.append(after_align_vids)
        Vs[i] = V
        connectivities[i] = connectivity
    return after_align_segments

def align_adjacent_patches(patch_to_V,
                           patch_to_connectivity,
                           patch_to_aggregated_tracing_data,
                           feature_edge_to_adjacent_patches):

    class PatchSegments:
        def __init__(self,
                     segments: list[list[int]] = None):
            self.segments = segments if segments is not None else []
    
    for feature_edge_id, adjacent_patch_ids in feature_edge_to_adjacent_patches.items():
        if len(adjacent_patch_ids) < 1:
            raise ValueError(f"Error: feature edge {feature_edge_id} has no patch attached")
        if len(adjacent_patch_ids) < 2:
            continue  # nothing to align
        if any(patch_to_aggregated_tracing_data[p].feature_id_to_segment_upsample_size.get(feature_edge_id) is None
               for p in adjacent_patch_ids):
            continue  # degenerate (zero-length) feature edge: a single corner vertex, nothing to align
        list_of_patch_segments = [] # list[list[list[int]]]
        connectivities = []
        Vs = []
        for patch_id in adjacent_patch_ids:
            print_if_verbose(f"aligning patch {patch_id} for feature edge {feature_edge_id}")
            connectivities.append(patch_to_connectivity[patch_id])
            Vs.append(patch_to_V[patch_id])

            if config.verbose:  # diagnostics only: they print nothing otherwise

                da = igl.doublearea(np.array(patch_to_V[patch_id]), np.array(patch_to_connectivity[patch_id].F))
                zero_area_faces = np.where(da < 1e-16 )[0]
                ia = igl.internal_angles(np.array(patch_to_V[patch_id]), np.array(patch_to_connectivity[patch_id].F))
                zero_min_angle_faces = np.where(ia.min(axis=1) < 1e-16)[0]
                if len(zero_min_angle_faces) > 0:
                    print_if_verbose(f" Warning: Patch {patch_id} has {len(zero_min_angle_faces)} zero-min-angle faces before alignment")
                if len(zero_area_faces) > 0:
                    print_if_verbose(f" Warning: Patch {patch_id} has {len(zero_area_faces)} zero-area faces before alignment")


            aggregated_tracing_data = patch_to_aggregated_tracing_data[patch_id]

            traced_vids = aggregated_tracing_data.feature_id_to_traced_vids[feature_edge_id]
            segment_upsample_sizes = aggregated_tracing_data.feature_id_to_segment_upsample_size[feature_edge_id]
            if aggregated_tracing_data.feature_id_to_orientation[feature_edge_id] == False:
                traced_vids = traced_vids[::-1]
                segment_upsample_sizes = segment_upsample_sizes[::-1]
            
            prev_index = 0
            next_index = 0
            patch_segments = PatchSegments()
            for i, seg_size in enumerate(segment_upsample_sizes):
                #
                # |_,_,_|_,_|
                # 0 1 2 3 4 5
                # we want 0 to 3 including 3, so we want 0 to 4 for list slicing,
                # and since the upsample size is 2, so we want (0 + upsample_size + 1 + 1)

                next_index = prev_index + seg_size + 1 + 1
                traced_seg_vids = traced_vids[prev_index:next_index]
                prev_index = next_index - 1
                patch_segments.segments.append(traced_seg_vids)
            list_of_patch_segments.append(patch_segments)
        # every patch_segments.segments should have same size
        segment_size = len(list_of_patch_segments[0].segments)
        for i in range(segment_size):
            segments_to_align = []
            for patch_segments in list_of_patch_segments:
                segments_to_align.append(patch_segments.segments[i])
            after_align_segments = align_one_segment_of_patches(Vs, 
                                                                connectivities,
                                                                segments_to_align)
            if i == segment_size - 1:
                for j, after_align_segment in enumerate(after_align_segments):
                    after_align_segment.append(list_of_patch_segments[j].segments[-1][-1])  # add the last point
            # update the aligned segments back to the patch_segments
            for j, patch_segments in enumerate(list_of_patch_segments):
                patch_segments.segments[i] = after_align_segments[j]
                
        # now all segments are aligned, we can update the tracing data back to the patches
        for k, patch_id in enumerate(adjacent_patch_ids):
            aggregated_tracing_data = patch_to_aggregated_tracing_data[patch_id]
            after_align_vids = []
            patch_segments = list_of_patch_segments[k]
            for segment in patch_segments.segments:
                after_align_vids.extend(segment)
            aggregated_tracing_data.feature_id_to_traced_vids[feature_edge_id] = after_align_vids


    for patch_id in patch_to_V.keys():
        V = np.array(patch_to_V[patch_id])
        connectivity = np.array(patch_to_connectivity[patch_id].F)
        da = igl.doublearea(V, connectivity)
        zero_area_faces = np.where(da < 1e-16 )[0]
        if len(zero_area_faces) > 0:
            print_if_verbose(f" Warning: Patch {patch_id} has {len(zero_area_faces)} zero-area faces after alignment")
        ia = igl.internal_angles(np.array(patch_to_V[patch_id]), np.array(patch_to_connectivity[patch_id].F))
        zero_min_angle_faces = np.where(ia.min(axis=1) < 1e-16)[0]
        if len(zero_min_angle_faces) > 0:
            print_if_verbose(f" Warning: Patch {patch_id} has {len(zero_min_angle_faces)} zero-min-angle faces after alignment")


def _get_boundary_edges(F):
    """Return set of boundary edges (edges appearing in exactly 1 triangle)."""
    edge_face_count = {}
    for tri in F:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = create_edge(a, b)
            edge_face_count[e] = edge_face_count.get(e, 0) + 1
    return {e for e, cnt in edge_face_count.items() if cnt == 1}


def _classify_periodic_edges(V, boundary_edges, periodic_index, lb, ub, tol):
    """Classify boundary edges as being on the lower (0) or upper (1) domain bound.

    For periodic_index=0 (u-periodic), checks whether both vertices of an edge
    have u ≈ u_lb or u ≈ u_ub.

    Returns dict mapping side_value (0 or 1) -> list of edges.
    """
    sides = {0: [], 1: []}
    for e in boundary_edges:
        v0, v1 = e
        coord0 = V[v0][periodic_index]
        coord1 = V[v1][periodic_index]
        if abs(coord0 - lb) < tol and abs(coord1 - lb) < tol:
            sides[0].append(e)
        elif abs(coord0 - ub) < tol and abs(coord1 - ub) < tol:
            sides[1].append(e)
    return sides


def _build_ordered_chain(V, edges, non_periodic_index):
    """Build ordered vertex chain from edges, sorted by ascending non-periodic coordinate.

    Legacy single-chain interface. If edges form multiple disconnected chains,
    returns only the first one (by non-periodic coordinate).
    Use _build_ordered_chains for multi-chain support.
    """
    chains = _build_ordered_chains(V, edges, non_periodic_index)
    if not chains:
        return []
    return chains[0]


def _build_ordered_chains(V, edges, non_periodic_index):
    """Build all ordered vertex chains from edges, sorted by ascending non-periodic coordinate.

    Returns a list of chains. Each chain is an ordered list of vertex ids.
    For a connected set of edges, returns a single chain.
    For disconnected edges (e.g., interior singularity split), returns multiple chains
    sorted by their starting non-periodic coordinate.
    """
    if not edges:
        return []
    # Build adjacency
    adj = {}
    for v0, v1 in edges:
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    visited = set()
    chains = []

    # Find all connected components and walk each as a chain
    for start_candidate in sorted(adj.keys(), key=lambda v: V[v][non_periodic_index]):
        if start_candidate in visited:
            continue

        # BFS to find all vertices in this component
        component = set()
        queue = [start_candidate]
        while queue:
            v = queue.pop()
            if v in component:
                continue
            component.add(v)
            for n in adj.get(v, []):
                if n not in component:
                    queue.append(n)

        # Start from endpoint with smallest non-periodic parameter
        endpoints = [v for v in component if len(adj.get(v, [])) == 1]
        if endpoints:
            start = min(endpoints, key=lambda v: V[v][non_periodic_index])
        else:
            # Closed loop
            start = min(component, key=lambda v: V[v][non_periodic_index])

        # Walk the chain
        chain = [start]
        prev = None
        current = start
        while True:
            next_v = None
            for n in adj.get(current, []):
                if n != prev:
                    next_v = n
                    break
            if next_v is None or next_v == start:
                break
            chain.append(next_v)
            prev = current
            current = next_v

        visited.update(chain)
        chains.append(chain)

    # Sort chains by starting non-periodic coordinate
    chains.sort(key=lambda c: V[c[0]][non_periodic_index])
    return chains


def _align_one_side_periodic(V, connectivity, chain, periodic_index, non_periodic_index, other_params):
    """Walk one side's boundary chain and split edges to inject other side's parameter values.

    For each parameter value in other_params that doesn't already exist on this
    side, split the boundary edge that spans that parameter and insert a new
    vertex at the target perpendicular coordinate.

    Returns the updated chain (with newly inserted vertices).
    """
    if len(chain) < 2 or not other_params:
        return chain

    result = [chain[0]]
    chain_idx = 0
    param_idx = 0

    while param_idx < len(other_params) and chain_idx < len(chain) - 1:
        target = other_params[param_idx]
        v_curr = result[-1]
        v_next = chain[chain_idx + 1]

        curr_p = V[v_curr][non_periodic_index]
        next_p = V[v_next][non_periodic_index]

        if curr_p >= target:
            # Already at or past this target parameter
            param_idx += 1
            continue

        if next_p <= target:
            # Target is at or beyond the next vertex — advance along the chain
            result.append(v_next)
            chain_idx += 1
            continue

        # Target is strictly between curr and next — split the edge
        new_uv = [0.0, 0.0]
        new_uv[periodic_index] = (V[v_curr][periodic_index] + V[v_next][periodic_index]) / 2
        new_uv[non_periodic_index] = target
        split_edge_in_mesh_overloaded(V, connectivity, (v_curr, v_next), new_point=new_uv)
        new_vid = len(V) - 1
        result.append(new_vid)
        # After split: edge (v_curr, v_next) → (v_curr, new_vid) + (new_vid, v_next)
        # Continue from new_vid; chain[chain_idx + 1] is still v_next.
        param_idx += 1

    # Append remaining chain vertices
    for i in range(chain_idx + 1, len(chain)):
        result.append(chain[i])

    return result


def _align_chain_pair(V, connectivity, chain0, chain1, periodic_index, non_periodic):
    """Align a single pair of corresponding periodic boundary chains.

    Uses a two-pointer walk to snap close vertices and inject missing ones,
    then cross-aligns both sides so they have equal length.

    Returns (aligned_chain0, aligned_chain1).
    """
    SNAP_REL = 0.1
    params1_to_inject = []
    params0_to_inject = []
    snap_updates = []

    i, j = 0, 0
    while i < len(chain0) - 2 and j < len(chain1) - 2:
        p0 = V[chain0[i + 1]][non_periodic]
        p1 = V[chain1[j + 1]][non_periodic]
        edge_len = min(
            abs(p0 - V[chain0[i]][non_periodic]),
            abs(p1 - V[chain1[j]][non_periodic]),
        )
        snap_tol = SNAP_REL * edge_len if edge_len > 0 else 0.0

        if abs(p0 - p1) <= snap_tol:
            mid = (p0 + p1) / 2.0
            ok0 = (i + 2 >= len(chain0) or mid <= V[chain0[i + 2]][non_periodic])
            ok1 = (j + 2 >= len(chain1) or mid <= V[chain1[j + 2]][non_periodic])
            if ok0 and ok1:
                snap_updates.append((chain0[i + 1], mid))
                snap_updates.append((chain1[j + 1], mid))
                i += 1
                j += 1
            elif p0 <= p1:
                params0_to_inject.append(p0)
                i += 1
            else:
                params1_to_inject.append(p1)
                j += 1
        elif p0 < p1:
            params0_to_inject.append(p0)
            i += 1
        else:
            params1_to_inject.append(p1)
            j += 1

    while i < len(chain0) - 2:
        params0_to_inject.append(V[chain0[i + 1]][non_periodic])
        i += 1
    while j < len(chain1) - 2:
        params1_to_inject.append(V[chain1[j + 1]][non_periodic])
        j += 1

    for vid, new_val in snap_updates:
        V[vid][non_periodic] = new_val

    chain0 = _align_one_side_periodic(
        V, connectivity, chain0, periodic_index, non_periodic, params1_to_inject)
    chain1 = _align_one_side_periodic(
        V, connectivity, chain1, periodic_index, non_periodic, params0_to_inject)

    if len(chain0) != len(chain1):
        print_if_verbose(f"Warning: seam chain lengths differ after alignment: "
                         f"{len(chain0)} vs {len(chain1)}")
        print_if_verbose(f"chain 0 {np.array(V)[chain0]}")
        print_if_verbose(f"chain 1 {np.array(V)[chain1]}")
        raise Exception(f"Error: seam chain lengths differ after alignment: "
                        f"{len(chain0)} vs {len(chain1)}")

    return list(chain0), list(chain1)


def align_seam_edges_from_tags(V, connectivity, u_lb_edges, u_ub_edges, v_lb_edges, v_ub_edges):
    """Align periodic seam edges using pre-classified edge lists from tags.

    Parameters
    ----------
    V : list of [u, v] positions (modified in-place).
    connectivity : ConnectivityMaintainance (modified in-place).
    u_lb_edges, u_ub_edges : list of (v0, v1) 0-based edge pairs for u-periodic seam sides.
    v_lb_edges, v_ub_edges : list of (v0, v1) 0-based edge pairs for v-periodic seam sides.

    Returns
    -------
    u_periodic_edges : list of (list_side0, list_side1) chain pairs.
    v_periodic_edges : list of (list_side0, list_side1) chain pairs.
    """
    u_periodic_edges = []
    v_periodic_edges = []

    for periodic_index, (lb_edges, ub_edges) in enumerate([
        (u_lb_edges, u_ub_edges),
        (v_lb_edges, v_ub_edges),
    ]):
        if not lb_edges and not ub_edges:
            continue

        non_periodic = 1 - periodic_index

        sides = {0: list(lb_edges), 1: list(ub_edges)}
        print_if_verbose(f"  Tag-based seam edges: {len(sides[0])} at lb, {len(sides[1])} at ub (periodic_index={periodic_index})")
        if not sides[0] or not sides[1]:
            continue

        chains0 = _build_ordered_chains(V, sides[0], non_periodic)
        chains1 = _build_ordered_chains(V, sides[1], non_periodic)

        if len(chains0) != len(chains1):
            print_if_verbose(f"  Warning: different number of chains on periodic sides: "
                             f"{len(chains0)} vs {len(chains1)} (periodic_index={periodic_index})")
            # Fallback: use single-chain behavior
            chains0 = [_build_ordered_chain(V, sides[0], non_periodic)]
            chains1 = [_build_ordered_chain(V, sides[1], non_periodic)]

        chain_pairs = []
        for k in range(len(chains0)):
            c0 = chains0[k]
            c1 = chains1[k]
            if len(c0) < 2 or len(c1) < 2:
                chain_pairs.append((list(c0), list(c1)))
                continue
            aligned_c0, aligned_c1 = _align_chain_pair(
                V, connectivity, c0, c1, periodic_index, non_periodic)
            chain_pairs.append((aligned_c0, aligned_c1))

        if periodic_index == 0:
            u_periodic_edges = chain_pairs
        else:
            v_periodic_edges = chain_pairs

    return u_periodic_edges, v_periodic_edges


def embed_and_align_seam_edges(V, connectivity, face, u_lb, u_ub, v_lb, v_ub,
                               u_periodic, v_periodic):
    """Align boundary vertices on both sides of periodic seams.

    Parameters
    ----------
    V : list of [u, v] positions (modified in-place).
    connectivity : ConnectivityMaintainance (modified in-place).
    face : abs Face object with ``.surface`` and ``.loops``.
    u_periodic, v_periodic : bool
        Periodicity flags from detect_periodicity().

    Returns
    -------
    u_periodic_edges : list of (list_side0, list_side1) chain pairs.
    v_periodic_edges : list of (list_side0, list_side1) chain pairs.
    """
    if not u_periodic and not v_periodic:
        return [], []

    TWO_PI = 2.0 * np.pi
    u_span = u_ub - u_lb
    v_span = v_ub - v_lb

    u_periodic_edges = []
    v_periodic_edges = []

    stype = getattr(face.surface, 'shape_name', '')
    if stype in ('Cylinder', 'Cone', 'Sphere', 'Revolution', 'Torus'):
        u_ub_eff = u_lb + TWO_PI if u_span > TWO_PI + BOUNDARY_TOL else u_ub
        v_ub_eff = v_lb + TWO_PI if v_span > TWO_PI + BOUNDARY_TOL else v_ub
    else:
        u_ub_eff = u_ub
        v_ub_eff = v_ub

    for periodic_index, (is_periodic, lb, ub) in enumerate([
        (u_periodic, u_lb, u_ub_eff),
        (v_periodic, v_lb, v_ub_eff),
    ]):
        if not is_periodic:
            continue

        non_periodic = 1 - periodic_index

        # Find and classify boundary edges for this periodic direction
        bnd_edges = _get_boundary_edges(connectivity.F)
        sides = _classify_periodic_edges(V, bnd_edges, periodic_index, lb, ub, SAME_VERTEX_TOL)

        if not sides[0] or not sides[1]:
            continue

        # Build ordered vertex chains on each side (may be multiple if interior singularity)
        chains0 = _build_ordered_chains(V, sides[0], non_periodic)
        chains1 = _build_ordered_chains(V, sides[1], non_periodic)

        if len(chains0) != len(chains1):
            print_if_verbose(f"  Warning: different number of chains on periodic sides: "
                             f"{len(chains0)} vs {len(chains1)} (periodic_index={periodic_index})")
            # Fallback: use single-chain behavior
            chains0 = [_build_ordered_chain(V, sides[0], non_periodic)]
            chains1 = [_build_ordered_chain(V, sides[1], non_periodic)]

        chain_pairs = []
        for k in range(len(chains0)):
            c0 = chains0[k]
            c1 = chains1[k]
            if len(c0) < 2 or len(c1) < 2:
                chain_pairs.append((list(c0), list(c1)))
                continue
            aligned_c0, aligned_c1 = _align_chain_pair(
                V, connectivity, c0, c1, periodic_index, non_periodic)
            chain_pairs.append((aligned_c0, aligned_c1))

        if periodic_index == 0:
            u_periodic_edges = chain_pairs
        else:
            v_periodic_edges = chain_pairs

    return u_periodic_edges, v_periodic_edges