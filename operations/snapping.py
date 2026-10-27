import numpy as np
import math
from utils.data_structures import LoopGeometryData, PatchTracingData
from utils.print import print_if_verbose
from scipy.sparse import diags, coo_matrix
from scipy.sparse.linalg import splu

def upsample_feature_edge_polyline_V_wrt_arclen(feature_edge_to_polyline_V, 
                                                feature_edge_per_segment_vertex_cnt, 
                                                feature_edge_vids_on_patch, 
                                                patch_V):

    # return: the upsampled polylineV  for current patch
    # get length of each edge on the patch for each segment 
    edge_lengths_per_segment_feature_edge = []
    traced_polyline_segment_length = [] # same size as the number of segments
    index = 0
    for idx, upsample_vertex_cnt in enumerate(feature_edge_per_segment_vertex_cnt):
        segment_lengths = []
        polyline_segment_length = 0
        # ._ o _ o _ .
        # 0  1   2   3
        # segment size = 2
        for i in range(upsample_vertex_cnt + 1):
            vid1 = feature_edge_vids_on_patch[index + i]
            vid2 = feature_edge_vids_on_patch[index + i + 1]
            if vid1 == vid2:
                raise ValueError(f"Duplicate vertex id {vid1} in feature_edge_vids_on_patch at segment {idx}, index {i}")
            length = np.linalg.norm(np.array(patch_V[vid1]) - np.array(patch_V[vid2]))
            polyline_segment_length += length
            segment_lengths.append(length)
        
        if polyline_segment_length <= 0:
            print_if_verbose(f"    !!!! Warning: polyline segment {idx} length is zero,")

        traced_polyline_segment_length.append(polyline_segment_length)
        edge_lengths_per_segment_feature_edge.append(segment_lengths)
        index += upsample_vertex_cnt + 1  # +1 to get to the end of the segment
    upsampled_feature_edge_polyline_V = []
    for i, segment_vertex_cnt in enumerate(feature_edge_per_segment_vertex_cnt):
        left_pos = feature_edge_to_polyline_V[i]
        right_pos = feature_edge_to_polyline_V[i + 1]

        cummulative_edge_length = 0
        # now we linearly interpolate the points on the edge wrt the length of the edge
        upsampled_feature_edge_polyline_V.append(left_pos)
        if len(edge_lengths_per_segment_feature_edge[i]) == 1:
            # no point on segment
            continue
        for j, edge_length in enumerate(edge_lengths_per_segment_feature_edge[i]): 
            if j == len(edge_lengths_per_segment_feature_edge[i]) - 1:
                # last segment has the right vertex at the right_pos of the original segment 
                break
            # get the ratio of the edge length to the segment length
            cummulative_edge_length += edge_length
            if not traced_polyline_segment_length[i] > 0:
                t = (j + 1) / len(edge_lengths_per_segment_feature_edge[i])
            else:
                t = cummulative_edge_length / traced_polyline_segment_length[i]
            # get the position on the edge
            new_pos = (1 - t) * left_pos + t * right_pos
            upsampled_feature_edge_polyline_V.append(new_pos)

    upsampled_feature_edge_polyline_V.append(feature_edge_to_polyline_V[-1])
    return upsampled_feature_edge_polyline_V

def get_boundary_index_and_positions(feature_edge_vids, feature_edge_to_polyline_V):
    # Construct the boundary condition for the mesh
    # Input:
    # feature_edge_vids: the vertex ids of the feature edges
    # feature_edge_to_polyline_V: the vertices of the feature edges
    # Output:
    # bi: the boundary indices
    # bp: the target positions of the boundary vertices bi (points of polyline_V)
    bc_d = {}
    bi = []
    bp = []
    # since feature_edge_vids could contain duplicate vertices but feature_edge_to_polyline_V does not, we need to iterate over the unique vertices in feature_edge_vids
    for i, vid in enumerate(feature_edge_vids):
        bc_d[vid] = feature_edge_to_polyline_V[i] 

    bi = list(bc_d.keys())
    bp = list(bc_d.values())
    return bi, bp

def get_patch_boundary_condition(V,
                                 patch_loop_geometry_data: dict[int, LoopGeometryData],
                                 patch_tracing_data: PatchTracingData):
    patch_bi = list()
    patch_bp = list()
    initial_loop_id = patch_tracing_data.initial_loop_tracing_data.initial_loop_id
    for loop_id, loop_geometry_data in patch_loop_geometry_data.items():
        if loop_id == initial_loop_id:
            get_loop_boundary_condition(V,
                                        loop_geometry_data,
                                        patch_tracing_data.initial_loop_tracing_data.loop_tracing_data,
                                        patch_bi,
                                        patch_bp)
        else:
            get_loop_boundary_condition(V, 
                                        loop_geometry_data, 
                                        patch_tracing_data.remaining_loops_tracing_data[loop_id], 
                                        patch_bi, 
                                        patch_bp)
    # A corner is listed by every traced feature ending at it: keep one constraint per vertex,
    # otherwise the harmonic solve counts the corner's displacement once per copy.
    patch_bp = np.asarray(patch_bp, dtype=np.float64).reshape(-1, 3)
    bi, first, inverse = np.unique(np.asarray(patch_bi, dtype=np.int64), return_index=True, return_inverse=True)
    mismatch = np.any(patch_bp != patch_bp[first][inverse], axis=1)
    if np.any(mismatch):
        vid = patch_bi[int(np.argmax(mismatch))]
        raise ValueError(f"Boundary vertex {vid} has different target positions on different traced feature edges")
    return bi, patch_bp[first]

def get_loop_boundary_condition(V,
                                loop_geometry_data,
                                loop_tracing_data,
                                patch_bi: list,
                                patch_bp: list):
    for feature_id, traced_vids in loop_tracing_data.feature_id_to_edge_tracing_data.items():
        if len(loop_geometry_data.feature_id_poly_V_True_orientation[feature_id]) == 0:
            continue
        orientation = loop_tracing_data.feature_id_to_edge_orientation[feature_id]
        segment_sizes = loop_tracing_data.feature_id_to_segment_upsample_size[feature_id]
        if not orientation:
            traced_vids = traced_vids[::-1]  # reverse the order
            segment_sizes = segment_sizes[::-1]
        original_feature_polyline_V_in_True = loop_geometry_data.feature_id_poly_V_True_orientation[feature_id]
        upsampled_feature_polyline_V = upsample_feature_edge_polyline_V_wrt_arclen(original_feature_polyline_V_in_True, 
                                                                                   segment_sizes, 
                                                                                   traced_vids, 
                                                                                   V)
        bi, bp = get_boundary_index_and_positions(traced_vids, 
                                                  upsampled_feature_polyline_V)
        patch_bi.extend(bi)
        patch_bp.extend(bp)

def adjacency_matrix(F, n_vertices):
    """
    Create a symmetric adjacency matrix from face list F.
    """
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    src = F.reshape(-1)
    dst = F[:, [1, 2, 0]].reshape(-1)
    I = np.concatenate([src, dst])
    J = np.concatenate([dst, src])
    A = coo_matrix((np.ones(len(I)), (I, J)), shape=(n_vertices, n_vertices))
    return A.tocsr()


def compute_uniform_laplacian(F, n_vertices):
    A = adjacency_matrix(F, n_vertices)
    d = A.sum(axis=1).A1
    D = diags(d)
    return D - A


def harmonic(V, F, b, bc):
    """
    Uniform harmonic interpolant with boundary constraints.

    Inputs:
        V (n, 3): vertex positions
        F (m, 3): face indices
        b (k,): boundary indices
        bc (k, 3): boundary values

    Returns:
        W (n, 3): solution
    """
    n = V.shape[0]
    L = compute_uniform_laplacian(F, n)

    # Q = -L (first-order harmonic)
    Q = -L.copy()

    # Solve min xᵀ Q x s.t. x[b] = bc
    all_idx = np.arange(n)
    bi = np.array(b)
    ii = np.setdiff1d(all_idx, bi)

    # Slice rows then columns: Q[ii[:, None], ii] would broadcast the index
    # arrays to a dense (n_interior, n_interior) int matrix (~1 TiB at 500k
    # vertices) before extracting; sequential slicing stays O(nnz).
    Q = Q.tocsr()
    Qrows = Q[ii]
    Qii = Qrows[:, ii].tocsc()
    Qib = Qrows[:, bi].tocsr()

    W = np.zeros((n, bc.shape[1]))
    if len(ii):
        # factor Q_ii once, solve all columns
        W[ii] = splu(Qii).solve(np.asarray(-(Qib @ bc), dtype=float).reshape(len(ii), -1))
    W[bi] = bc

    return W

def snapping_for_patch(V, F, bi, bp, summary, ignore_errors=False):
    # Input validation
    # Ensure bi is within valid range
    if len(bi) > 0 and (np.min(bi) < 0 or np.max(bi) >= len(V)):
        raise ValueError("Boundary indices out of range")
    # Calculate boundary conditions
    bc = np.array([bp[i] - V[vid] for i, vid in enumerate(bi)], dtype=np.float64)
    # get the max and avg of bc to monitor the error caused by tracing
    squared_norms = np.sum(bc**2, axis=1)  # Compute squared norm for each row
    max_bc = np.max(squared_norms)
    avg_bc = np.mean(squared_norms)
    summary["max_error"] = math.sqrt(max_bc)
    summary["avg_error"] = math.sqrt(avg_bc)
    summary["number_of_boundary_vertices"] = len(bi)
    print_if_verbose(f"Max tracing error: {math.sqrt(max_bc)}")
    print_if_verbose(f"Avg tracing error: {math.sqrt(avg_bc)}")
    if max_bc > 1:
        if ignore_errors:
            print_if_verbose("!!! Warning: Tracing error too large, but ignoring due to ignore_errors=True")
        else:
            raise ValueError(f"Tracing error too large: {math.sqrt(max_bc)}")
    try:
        V_copy = np.ascontiguousarray(V)
        F_copy = np.ascontiguousarray(F)
        bi_copy = np.ascontiguousarray(bi, dtype=np.int32)
        bc_copy = np.ascontiguousarray(bc, dtype=np.float64)
        # (no gc.collect() here: a full collection walks every live object and
        # dominated the snap stage on large models)
        D = harmonic(V_copy, F_copy, bi_copy, bc_copy)
        newV = V_copy + D
        V = [pos for pos in newV]
        # Update boundary vertices
        for i, vid in enumerate(bi):
            V[vid] = bp[i]
        
        return V
        
    except Exception as e:
        print_if_verbose(f"Error in harmonic field computation: {str(e)}")
        print_if_verbose(f"Shapes - V: {np.shape(V)}, F: {np.shape(F)}, bi: {np.shape(bi)}, bc: {np.shape(bc)}")
        raise RuntimeError(f"Failed to compute harmonic field: {str(e)}")

def solve_one_patch_linear_system(V,
                                  F,
                                  patch_tracing_data,
                                  patch_loop_geometry_data,
                                  summary,
                                  ignore_errors=False):
    bi, bp = get_patch_boundary_condition(V,
                                          patch_loop_geometry_data,
                                          patch_tracing_data)
    newV = snapping_for_patch(V, F, bi, bp, summary=summary, ignore_errors=ignore_errors)
    return newV