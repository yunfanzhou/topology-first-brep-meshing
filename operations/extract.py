import numpy as np
import igl
from collections import deque as deque   
from utils.mesh_utils import create_edge
from utils.print import print_if_verbose

def traced_edge_separated_components(F, tracing_edge_sets, V=None):
    # Find connected components separated by tracing edges
    cc = marching_inside_loop(F, tracing_edge_sets)
    return cc

def marching_inside_loop(F, loop_edges):
    """Find connected components separated by a loop.
       F is np.array(connectivity.F)
    """ 
    F_np = F
    if type(F) is not np.ndarray:
        F_np = np.array(F)
    TT, _ = igl.triangle_triangle_adjacency(F_np)
    unseen = set(range(len(F_np)))
    connected_components = []
    while unseen:
        q = deque()
        seed = unseen.pop()  # Start BFS from any unseen face
        q.append(seed)
        connected_component = []

        while q:
            f = q.popleft()
            connected_component.append(f)

            # Add neighbors
            for i in range(3):
                neighbor = TT[f, i]
                if neighbor == -1 or neighbor not in unseen:
                    continue

                # Extract edge from the current face
                face = F[f]
            
                edge = create_edge(face[(i ) % 3], face[(i +1) % 3])
                # If the edge is part of the loop, do not cross it
                if edge in loop_edges:
                    continue

                # Add to queue & mark as visited
                q.append(neighbor)
                unseen.remove(neighbor)

        connected_components.append(connected_component)

    return connected_components

# ======================================
# ======================================

def components_contains_all_trace(F_np,
                                 traced_edge_sets,
                                 components):
    '''
    Find all components that contain all traced edges. That is, meets only condition 2.
    This function is called during tracing a loop. Because our shortcut tracing, 
    while tracing initial loop, the traced edges could form a closed loop on the mesh.
    Therefore, separate the mesh into components.
    We want to make sure the succeeding traced edges are all in the same component.
    '''
     # Note: they can be multiple components that contain all the edges.
    candidate_c = []
    for component in components:
        if all_traced_edges_in_component(traced_edge_sets, F_np[component]):
            candidate_c.append(component)

    if len(candidate_c) > 0:
        return candidate_c
    
    # if there is no such component, fall throw and return None
    return None

def components_fullfill_2_3_4(F_np,
                               loop_topology_data,
                               loop_tracing_data,
                               traced_edge_sets,
                               cc):
    ret_component = []
    for i, component in enumerate(cc):
        F_subset = F_np[component]
        if (all_traced_edges_in_component(traced_edge_sets, F_subset) and
            all_boundary_edges_are_traced_edges(traced_edge_sets, F_subset) and
            all_seam_edges_are_interior_edges(loop_topology_data, loop_tracing_data, F_subset)):
            ret_component.append(component)

    return ret_component

def components_fullfill_2_3(F_np,
                             traced_edge_sets,
                             cc):
    ret_component = []
    for component in cc:
        F_subset = F_np[component]
        if (all_traced_edges_in_component(traced_edge_sets, F_subset) and
            all_boundary_edges_are_traced_edges(traced_edge_sets, F_subset)):
            ret_component.append(component)
    return ret_component

def components_pass_2_4_fail_3(F_np,
                               loop_topology_data,
                               loop_tracing_data,
                               traced_edge_sets,
                               cc):
    '''Components that meet conditions 2 and 4 but fail condition 3, i.e. annulus
    "disks" whose only defect is one or more untraced boundary loops (holes).
    These are candidates for repair by stitching the untraced hole(s) closed.'''
    ret_component = []
    for component in cc:
        F_subset = F_np[component]
        if (all_traced_edges_in_component(traced_edge_sets, F_subset) and
            not all_boundary_edges_are_traced_edges(traced_edge_sets, F_subset) and
            all_seam_edges_are_interior_edges(loop_topology_data, loop_tracing_data, F_subset)):
            ret_component.append(component)
    return ret_component

def _half_edge_faces(faces):
    '''Map each directed half-edge (a, b) to its face index; None if a half-edge
    occurs twice (non-manifold edge or inconsistent orientation).'''
    he_face = {}
    for fi, f in enumerate(faces):
        for k in range(3):
            he = (int(f[k]), int(f[(k + 1) % 3]))
            if he in he_face:
                return None
            he_face[he] = fi
    return he_face


def stitch_untraced_boundary_loops(V, faces, traced_edge_sets):
    '''Fan-fill every boundary loop of `faces` that is entirely untraced (a hole)
    with a triangle fan whose apex is the 3D centroid of that loop's vertices. The
    traced loop (an all-traced boundary loop) is left as the disk's real boundary.

    Mutates V (appends one apex vertex per hole) and returns (new_faces, ok).
    `ok` is False - and V is left untouched - if any boundary loop is only
    partially traced (not a clean annulus), if a hole touches another boundary
    loop at a vertex (a pinch), or if the filled mesh is not an oriented
    edge-manifold mesh whose boundary edges are all traced. The caller then
    traces the loop again with the fallback tracer.
    '''
    print_if_verbose("    stitch: checking for untraced boundary loops to fill...")
    new_faces = [list(f) for f in faces]
    he_face = _half_edge_faces(faces)
    if he_face is None:
        return new_faces, False
    # igl.all_boundary_loop orders each loop along the existing half-edges (disk
    # interior on the left). Where a hole touches another loop at one vertex it
    # walks vertex adjacency into fragments whose steps are not boundary
    # half-edges; filling those would add faces on edges that do not exist.
    boundary_loops = [[int(v) for v in loop] for loop in igl.all_boundary_loop(np.array(faces))]
    n_boundary = sum(1 for he in he_face if (he[1], he[0]) not in he_face)
    if (sum(len(loop) for loop in boundary_loops) != n_boundary
            or any(len(loop) < 3 or (loop[i - 1], loop[i]) not in he_face or (loop[i], loop[i - 1]) in he_face
                   for loop in boundary_loops for i in range(len(loop)))):
        print_if_verbose("    stitch: boundary loops touch at a vertex; not stitching.")
        return new_faces, False
    holes = []
    for loop in boundary_loops:
        edges = [create_edge(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))]
        traced_count = sum(1 for e in edges if e in traced_edge_sets)
        if traced_count == len(edges):
            continue  # the traced loop: keep as the disk boundary
        if traced_count != 0:
            return new_faces, False  # mixed loop: not a clean hole
        holes.append(loop)
    n_V = len(V)
    for loop in holes:
        centroid = np.mean(np.array([V[v] for v in loop], dtype=float), axis=0)
        apex = len(V)
        V.append(centroid)
        # loop[i] -> loop[i+1] is an existing half-edge; the fill triangle
        # (loop[i+1], loop[i], apex) keeps the new faces' winding consistent.
        for i in range(len(loop)):
            a = loop[i]
            b = loop[(i + 1) % len(loop)]
            new_faces.append([b, a, apex])
    filled_he = _half_edge_faces(new_faces)
    if (filled_he is None
            or any(len(set(f)) < 3 for f in new_faces[len(faces):])
            or any(create_edge(a, b) not in traced_edge_sets
                   for (a, b) in filled_he if (b, a) not in filled_he)):
        print_if_verbose("    stitch: filled mesh is not manifold or has an untraced boundary edge; not stitching.")
        del V[n_V:]
        return [list(f) for f in faces], False
    return new_faces, True

# condition 2: all traced edges are in the same component
def all_traced_edges_in_component(traced_edge_sets,
                                  F_subset):
    # check all the traced edges in the set of edges

    component_edge_set = set()
    unordered_edges = igl.edges(F_subset)

    for edge in unordered_edges:
        component_edge_set.add(create_edge(edge[0], edge[1]))
    if traced_edge_sets.issubset(component_edge_set):
        return True
    return False

# condition 3: all boundary edges are traced edges
def all_boundary_edges_are_traced_edges(traced_edge_sets,
                                        F_subset):
    boundary_edges = igl.boundary_facets(F_subset)  # sorted vid pairs
    for bnd_edge in boundary_edges:
        bnd_edge_tuple = create_edge(bnd_edge[0], bnd_edge[1])
        if bnd_edge_tuple not in traced_edge_sets:
            return False
    return True

# condition 4: all the seam edges are interior edges
def all_seam_edges_are_interior_edges(loop_topology_data,
                                      loop_tracing_data,
                                      F_subset):
    # get all the edges of the component
    component_edges = set()
    unordered_edges = igl.edges(F_subset)

    for edge in unordered_edges:
        component_edges.add(create_edge(edge[0], edge[1]))
    # get all the boundary edges
    boundary_edges = set()
    unordered_edges = igl.boundary_facets(F_subset)
    for edge in unordered_edges:
        boundary_edges.add(create_edge(edge[0], edge[1]))

    #get the interior edges
    interior_edges = component_edges - boundary_edges

    # check all the seam edges
    for feature_id, traced_vids in loop_tracing_data.feature_id_to_edge_tracing_data.items():
        if feature_id not in loop_topology_data.seam_ids_set:
            continue
        
        for i in range(len(traced_vids) - 1):
            seam_edge = create_edge(traced_vids[i], traced_vids[i + 1])
            if seam_edge not in interior_edges:
                return False
    return True

def update_connectivity_after_F_change(connectivity,
                                       current_loop,
                                       clear_backup = False):
    '''
    Update the connectivity(adj_list, edge_set_to_length_mapping, VF) information after remove faces from F.
    '''
    import copy
    ev = set([create_edge(connectivity.F[f][i], connectivity.F[f][(i + 1) % 3]) for f in range(len(connectivity.F)) for i in range(3)])
    for vid in current_loop:
        # --- Backup before modification ---
        if not clear_backup:    
            if connectivity.adj_list_copy_dict is not None and vid not in connectivity.adj_list_copy_dict:
                connectivity.adj_list_copy_dict[vid] = copy.deepcopy(connectivity.adj_list[vid])

        neighbors = connectivity.adj_list[vid]
        to_keep = {v for v in neighbors if create_edge(vid, v) in ev}
        to_remove = neighbors - to_keep

        neighbors.difference_update(to_remove)
        # update adjacency list
        for v in to_remove:
            # --- Backup before modification ---
            if not clear_backup:
                if connectivity.adj_list_copy_dict is not None and v not in connectivity.adj_list_copy_dict:
                    connectivity.adj_list_copy_dict[v] = copy.deepcopy(connectivity.adj_list[v])

            if v < len(connectivity.adj_list) and vid in connectivity.adj_list[v]:
                connectivity.adj_list[v].remove(vid)
        # update edge_to_length_mapping
        for v in to_remove:
            edge = create_edge(vid, v)
            if edge in connectivity.edge_set_to_length_mapping:
                # --- Backup before modification ---
                if not clear_backup:
                    if connectivity.modified_edge_set is not None and edge not in connectivity.modified_edge_set:
                        connectivity.modified_edge_set.add(edge)
                del connectivity.edge_set_to_length_mapping[edge]
    if clear_backup:
        connectivity.adj_list_copy_dict = {}
        connectivity.modified_edge_set = set()
    # Rebuild VF mapping
    for vid in connectivity.VF:
        connectivity.VF[vid].clear()  # Clear existing face references

    for face_idx, face in enumerate(connectivity.F):
        for vid in face:
            if vid in connectivity.VF:
                connectivity.VF[vid].append(face_idx)
            else:
                connectivity.VF[vid] = [face_idx]
