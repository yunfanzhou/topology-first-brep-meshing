import numpy as np
import copy 
from collections import defaultdict
import igl
from utils.print import print_if_verbose
def update_edge_length_and_adjacency_list(V, newv_index, edge, edge_adjacent_faces, adjacency_list, edge_mapping_to_length, adj_list_copy_dict=None, modified_edge_set=None):
    """
    Updates connectivity data structures after splitting an edge.

    This function assumes a new vertex `newv_index` has been created to split `edge`,
    and `edge_adjacent_faces` contains the new faces resulting from the split.
    """
    v0, v1 = int(edge[0]), int(edge[1])
    newv_index = int(newv_index)

    # --- 1. Identify all vertices involved ---
    vids_opposite_to_edge = set()
    for face in edge_adjacent_faces:
        if face is None or face[0] == -1:
            continue
        # Find the vertex in the face that is not part of the original edge
        opposite_v_set = set(face) - {v0, v1}
        if opposite_v_set:
            vids_opposite_to_edge.add(int(opposite_v_set.pop()))

    # --- 2. Backup adjacency lists if requested (BEFORE any changes) ---
    if adj_list_copy_dict is not None:
        nodes_to_backup = [v0, v1] + list(vids_opposite_to_edge)
        for vid in nodes_to_backup:
            if vid not in adj_list_copy_dict:
                adj_list_copy_dict[vid] = copy.deepcopy(list(adjacency_list[vid]))

    # --- 3. Update adjacency lists ---
    # Ensure adjacency list is large enough for the new vertex
    if len(adjacency_list) <= newv_index:
        adjacency_list.extend([set() for _ in range(newv_index - len(adjacency_list) + 1)])

    # Remove old edge connectivity
    adjacency_list[v0].discard(v1)
    adjacency_list[v1].discard(v0)

    # Add new connections from original edge vertices to the new vertex
    adjacency_list[v0].add(newv_index)
    adjacency_list[newv_index].add(v0)
    adjacency_list[v1].add(newv_index)
    adjacency_list[newv_index].add(v1)

    # Add new connections from opposite vertices to the new vertex
    for vid in vids_opposite_to_edge:
        adjacency_list[vid].add(newv_index)
        adjacency_list[newv_index].add(vid)

    # --- 4. Update edge lengths and modified set ---
    # Define normalized edges
    original_edge_norm = create_edge(v0, v1)
    new_edge1_norm = create_edge(v0, newv_index)
    new_edge2_norm = create_edge(v1, newv_index)

    if original_edge_norm is None:
        raise ValueError(f"Edge {edge} to split is invalid (vertices are the same).")

    # Update modified set
    if modified_edge_set is not None:
        modified_edge_set.add(original_edge_norm)
        modified_edge_set.add(new_edge1_norm)
        modified_edge_set.add(new_edge2_norm)

    # Update edge lengths for the split edge: the new vertex is not necessarily the midpoint
    old_len = edge_mapping_to_length.pop(original_edge_norm, None)
    if old_len is None:
        print_if_verbose(f"Warning: edge {original_edge_norm} not found in edge_mapping_to_length.")

    edge_mapping_to_length[new_edge1_norm] = np.linalg.norm(np.array(V[v0]) - np.array(V[newv_index]))
    edge_mapping_to_length[new_edge2_norm] = np.linalg.norm(np.array(V[v1]) - np.array(V[newv_index]))

    # Add lengths for new edges connecting to opposite vertices
    for vid in vids_opposite_to_edge:
        new_opposite_edge = create_edge(vid, newv_index)
        if new_opposite_edge:
            if modified_edge_set is not None:
                modified_edge_set.add(new_opposite_edge)
            edge_mapping_to_length[new_opposite_edge] = np.linalg.norm(np.array(V[vid]) - np.array(V[newv_index]))

    return adjacency_list, edge_mapping_to_length
def update_adjacency_list( newv_index, edge, edge_adjacent_faces, adjacency_list, adj_list_copy_dict = None):
    # Efficiently update the adjacency list
    """
    Updates connectivity data structures after splitting an edge.

    This function assumes a new vertex `newv_index` has been created to split `edge`,
    and `edge_adjacent_faces` contains the new faces resulting from the split.
    """
    v0, v1 = int(edge[0]), int(edge[1])
    newv_index = int(newv_index)

    # --- 1. Identify all vertices involved ---
    vids_opposite_to_edge = set()
    for face in edge_adjacent_faces:
        if face is None or face[0] == -1:
            continue
        # Find the vertex in the face that is not part of the original edge
        opposite_v_set = set(face) - {v0, v1}
        if opposite_v_set:
            vids_opposite_to_edge.add(int(opposite_v_set.pop()))

    # --- 2. Backup adjacency lists if requested (BEFORE any changes) ---
    if adj_list_copy_dict is not None:
        nodes_to_backup = [v0, v1] + list(vids_opposite_to_edge)
        for vid in nodes_to_backup:
            if vid not in adj_list_copy_dict:
                adj_list_copy_dict[vid] = copy.deepcopy(adjacency_list[vid])
    # --- 3. Update adjacency lists ---
    # Ensure adjacency list is large enough for the new vertex
    if len(adjacency_list) <= newv_index:
        adjacency_list.extend([set() for _ in range(newv_index - len(adjacency_list) + 1)])

    # Remove old edge connectivity
    adjacency_list[v0].discard(v1)
    adjacency_list[v1].discard(v0)

    # Add new connections from original edge vertices to the new vertex
    adjacency_list[v0].add(newv_index)
    adjacency_list[newv_index].add(v0)
    adjacency_list[v1].add(newv_index)
    adjacency_list[newv_index].add(v1)

    # Add new connections from opposite vertices to the new vertex
    for vid in vids_opposite_to_edge:
        adjacency_list[vid].add(newv_index)
        adjacency_list[newv_index].add(vid)

    return adjacency_list

def edge_length_mapping(V, F):
    """(min vid, max vid) -> 3D length for every edge of F (degenerate edges skipped)."""
    if len(F) == 0:
        return {}
    F_np = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    V_np = np.asarray(V, dtype=float)
    a = F_np[:, [1, 2, 0]].reshape(-1)
    b = F_np[:, [2, 0, 1]].reshape(-1)
    keep = a != b
    a, b = a[keep], b[keep]
    lengths = np.linalg.norm(V_np[a] - V_np[b], axis=1)
    lo, hi = np.minimum(a, b).tolist(), np.maximum(a, b).tolist()
    return dict(zip(zip(lo, hi), lengths.tolist()))

def adjacency_list(x, max_coeff=None):
    x_np = np.array(x)
    if max_coeff is None:
        max_coeff = np.max(x_np) + 1
    l = [set() for _ in range(max_coeff)]

    for a, b, c in x_np:
        l[a].add(b); l[b].add(a)
        l[b].add(c); l[c].add(b)
        l[c].add(a); l[a].add(c)

    return l
def compute_VF(F):
    # compute a VF table to reduce runtime 
    vertex_to_faces = defaultdict(list)
    for idx, face in enumerate(F):
        for v in face:
            vertex_to_faces[v].append(idx)
    return vertex_to_faces
def compute_EF(F):
    # Compute the edge to face mapping
    # Input: F
    # Output: edge_to_face
    edge_to_face = {}
    for f, face in enumerate(F):
        for i in range(3):
            edge = create_edge(face[i], face[(i + 1) % 3])
            if edge not in edge_to_face:
                edge_to_face[edge] = []
            edge_to_face[edge].append(f)
    return edge_to_face

def create_edge(left, right):
    # Create an edge from two vertices
    # Input: left, right
    # Output: edge
    if left == right:
        return None
    return (min(left, right), max(left, right))


def mesh_is_valid(F):
    # check if the mesh is valid, i.e., no triangle is adjacent to itself
    # and all triagle have unique vertices 
    for i, face in enumerate(F):
        if len(set(face)) < 3:
            print_if_verbose(f"Invalid face {i}: {face} has duplicate vertices.")
            return False
    return True
def topology_consistency(F, u_periodic_edges, v_periodic_edges):
    """
    Check if the mesh topology is consistent with the periodic edges.
    For a mesh with no periodic edges, there should be exactly one boundary loop.
    For a mesh with one set of periodic edges, there should be exactly two boundary loops.
    For a mesh with two sets of periodic edges, there should be no boundary loops.
    """
    all_bnd_loops = igl.all_boundary_loop(np.array(F))
    num_bnd_loops = len(all_bnd_loops)
    
    if not u_periodic_edges and not v_periodic_edges:
        expected_bnd_loops = 1
    elif (u_periodic_edges and not v_periodic_edges) or (not u_periodic_edges and v_periodic_edges):
        expected_bnd_loops = 2
    else:
        expected_bnd_loops = 0

    if num_bnd_loops != expected_bnd_loops:
        print_if_verbose(f"Topology inconsistency: Expected {expected_bnd_loops} boundary loops, but found {num_bnd_loops}.")
        return False
    return True