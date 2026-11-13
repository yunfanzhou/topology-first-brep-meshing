import numpy as np
import utils.dijkstra as dijkstra
from operations.internal_operations import embed_point_in_triangle
from utils.mesh_utils import adjacency_list
from utils.print import print_if_verbose
def project_feature_edge(V, F, feature_edge_polyline_V, feature_edge_orientation=True, myk=8):
    '''Project the feature edge to the mesh.
    Input: feature_edge_polyline_V, feature_edge_orientation
    Output:  #points by k (myk, at most #F) ndarray of the closest face ids (assuming regular mesh, this is one ring of triangles)
            Always a matrix: when there is only one vertex in polyline_V, it has one row.
    '''
    feature_V_oriented = feature_edge_polyline_V
    if feature_edge_orientation:
        pass
    else:
        feature_V_oriented = feature_edge_polyline_V[::-1]
    V_np = np.array(V)
    F_np = np.array(F)
    tree, _ = build_barycenter_kdtree(V_np, F_np)
    actual_k = min(myk, len(F_np))
    _, face_indices = tree.query(feature_V_oriented, k=actual_k)

    if isinstance(face_indices, np.ndarray) and face_indices.ndim == 1:
        face_indices = face_indices.reshape(1, -1)
    return face_indices 

def project_feature_edge_in_same_component(V,
                                        F_np,
                                        c,
                                        feature_edge_id_to_polyline_V,
                                        traced_id_set):
    '''Project the feature edge to the mesh, but only within the same connected component.
    We will do a majority vote among the closest faces for all the points of the feature edge. And pick the faces that are in the same component as the majority.
    '''
    # stack all the polyline V together
    curve_poly_V = []
    for feature_edge_id, polyline_V in feature_edge_id_to_polyline_V.items():
        if feature_edge_id in traced_id_set:
            continue
        curve_poly_V.extend(polyline_V)
    if len(curve_poly_V) == 0:
        # only degenerate (empty-polyline) feature edges are left: nothing to project
        return None, None
    V_np = np.array(V)
    tree, _ = build_barycenter_kdtree(V_np, F_np)
    actual_k = min(8, len(F_np))
    _, face_indices = tree.query(curve_poly_V, k=actual_k)
    if isinstance(face_indices, np.ndarray) and face_indices.ndim == 1:
        face_indices = face_indices.reshape(1, -1)

    # components 'c' is a list of components, each component is a list of face indices
    # build a set for each candidate component for quick membership test
    comp_sets = [set(comp) for comp in c]

    # Count how many times faces from each candidate component appear among the k nearest for all points
    comp_counts = [0] * len(comp_sets)
    for row in face_indices:
        for fid in np.atleast_1d(row):
            for ci, comp_set in enumerate(comp_sets):
                if fid in comp_set:
                    comp_counts[ci] += 1

    # choose the component with the maximum count (majority)
    if len(comp_counts) == 0:
        # nothing to do
        return face_indices, None
    chosen_idx = int(np.argmax(comp_counts))
    chosen_set_fids = comp_sets[chosen_idx]
    chosen_fids_list = list(chosen_set_fids)
    tree, _ = build_barycenter_kdtree(V_np, F_np[chosen_fids_list])
    actual_k2 = min(8, len(chosen_fids_list))
    _, face_indices_filtered = tree.query(curve_poly_V, k=actual_k2)
    if isinstance(face_indices_filtered, np.ndarray) and face_indices_filtered.ndim == 1:
        face_indices_filtered = face_indices_filtered.reshape(1, -1)
    return face_indices_filtered, chosen_set_fids

from scipy.spatial import KDTree
def build_barycenter_kdtree(V, F):
    V_np = np.array(V)
    F_np = np.array(F)
    barycenters = np.mean(V_np[F_np], axis=1)  # shape: (n_faces, 3)
    tree = KDTree(barycenters)
    return tree, barycenters

def trace_two_vertices(V, F, 
                       start_vid, 
                       end_vid,
                       tracing_prohibited_vids_set, 
                       adj_list=None, 
                       edge_lengths=None):
    # Trace a path between two vertices
    # Input:
    # V: vertices of the mesh
    # F: faces of the mesh
    # start_vid: the starting vertex id
    # end_vid: the ending vertex id
    # tracing_prohibited_vids_set: the set of vertex ids that cannot be traversed
    # adj_list: the vertex adjacency list (computed from F if None)
    # edge_lengths: dict mapping create_edge(v0, v1) to the edge length (required)
    # Output:
    # trace_vids: the list of vertex ids that are traced ([] if there is no path)
    # distance: the length of the traced path (inf if there is no path)
    if edge_lengths is None:
        raise TypeError("trace_two_vertices requires edge_lengths, a dict mapping each edge to its length")
    if adj_list is None:
        adj_list = adjacency_list(F)
    trace_vids, distance = dijkstra.dijkstra_shortest_path_to_single_target(edge_lengths, adj_list, start_vid, end_vid, tracing_prohibited_vids_set)
    return trace_vids, distance


def trace_to_find_best_candidate(V,
                                 connectivity,
                                 prev_vid,
                                 point,
                                 patch_regular_space_data,
                                 candidate_fids,
                                 off_boundary):
    current_vid, _ = embed_point_in_triangle(V,
                                             connectivity,
                                             point,
                                             patch_regular_space_data.frozen_edges_set,
                                             patch_regular_space_data.prohibited_vids_set,
                                             candidate_fids=candidate_fids,
                                             off_boundary=off_boundary,
                                             prev_vid=prev_vid,
                                             traced_edge_set=patch_regular_space_data.traced_edge_set)
    path_vids, dist = trace_two_vertices(V,
                                         connectivity.F,
                                         prev_vid,
                                         current_vid,
                                         patch_regular_space_data.prohibited_vids_set,
                                         connectivity.adj_list,
                                         connectivity.edge_set_to_length_mapping)
    if dist < float('inf'):
        return current_vid, path_vids, dist
    print_if_verbose("No valid path found among closest candidates ")
    raise ValueError("No valid path found among closest candidates")
