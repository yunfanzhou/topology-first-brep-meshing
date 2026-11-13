from operations.internal_operations import split_edge_in_mesh, split_edge_in_mesh_overloaded
from utils.mesh_utils import adjacency_list, create_edge
import igl
import numpy as np  
from utils.print import dump_oversized_patch_state

def construct_patch_initial_regular_space_data(connectivity, patch_regular_space_data):
    # initial regular space include the boundary edges as frozen edge, and their vertices as the prohibited_vids
    boundary_loops = igl.all_boundary_loop(np.array(connectivity.F))
    prohibited_vids = set()
    for loop in boundary_loops:
        prohibited_vids.update(loop)
    frozen_edges = set()
    for loop in boundary_loops:
        for i in range(len(loop)):
            frozen_edges.add(create_edge(loop[i], loop[(i + 1) % len(loop)]))
    patch_regular_space_data.prohibited_vids_set = prohibited_vids
    patch_regular_space_data.frozen_edges_set = frozen_edges
    patch_regular_space_data.traced_edge_set = set()
    patch_regular_space_data.current_loop = []
    patch_regular_space_data.previous_loops = []
    return

def regular_space_using_PRSD(V,
                             connectivity,
                             patch_regular_space_data):
    # iterate over the previous loops
    for i, tracing_loop_vids in enumerate(patch_regular_space_data.previous_loops):
        regular_space_each_loop_using_PRSD(V,
                                           connectivity,
                                           patch_regular_space_data,
                                           tracing_loop_vids)
    # iterate the current loop
    regular_space_each_loop_using_PRSD(V,
                                       connectivity,
                                       patch_regular_space_data,
                                       patch_regular_space_data.current_loop)
    # There are points that is in prohibited vids but not in the loops (e.g. topology recovery edges in non_initial_loop_fallback)
    # we need to split those edges as well
    rest_of_prohibited = patch_regular_space_data.prohibited_vids_set - set(patch_regular_space_data.current_loop)
    for tracing_loop_vids in patch_regular_space_data.previous_loops:
        rest_of_prohibited = rest_of_prohibited - set(tracing_loop_vids)
    regular_space_each_loop_using_PRSD(V,
                                       connectivity,
                                       patch_regular_space_data,
                                       list(rest_of_prohibited))

    # Monitoring only: dump oversized state, never interfere with the run.
    dump_oversized_patch_state(V, connectivity.F, tag="regular_space_using_PRSD")
    return

def regular_space_dumb(V, connectivity,
                        patch_regular_space_data):
    valid_vids = set(int(v) for face in connectivity.F for v in face)
    for vid in valid_vids:
        neighbors = connectivity.adj_list[vid]
        for neighbor in list(neighbors):
            edge = create_edge(vid, neighbor)
            if patch_regular_space_data.frozen_edges_set is not None and edge in patch_regular_space_data.frozen_edges_set:
                continue
            if neighbor in patch_regular_space_data.prohibited_vids_set:
                # split the edge
                edge = [vid, neighbor]
                split_edge_in_mesh_overloaded(V, connectivity, edge)
    # Monitoring only: dump oversized state, never interfere with the run.
    dump_oversized_patch_state(V, connectivity.F, tag="regular_space_dumb")
    return
def regular_space_each_loop_using_PRSD(V,
                                       connectivity,
                                       patch_regular_space_data,
                                       loop : list[int]):
    # iterate over the current loop
    for i in range(len(loop)):
        current = loop[i]
        neighbors = connectivity.adj_list[current]
        for neighbor in list(neighbors):
            edge = create_edge(current, neighbor)
            if patch_regular_space_data.frozen_edges_set is not None and edge in patch_regular_space_data.frozen_edges_set:
                continue
            if neighbor in patch_regular_space_data.prohibited_vids_set:
                # split the edge
                edge = [current, neighbor]
                split_edge_in_mesh_overloaded(V, connectivity, edge)

def regular_space_internal_on_entire_loop(prohibited_set,
                           V, F, 
                           tracing_loop_vids,
                           VF,
                           adj_list=None,
                           edge_set_to_length_mapping=None,
                           frozen_edge_set = None,
                           adj_list_copy_dict = None,
                           modified_edge_set = None):
    newV, newF = V, F
    if adj_list is None:
        adj_list = adjacency_list(newF)
    
    for i, vid in enumerate(tracing_loop_vids):
        next = tracing_loop_vids[(i + 1) % len(tracing_loop_vids)]  
        prev = tracing_loop_vids[(i - 1) % len(tracing_loop_vids)]  
        neighbors = set(adj_list[vid])
        neighbors -= {next, prev}
        for neighbor in neighbors:
            edge = create_edge(vid, neighbor)
            if frozen_edge_set is not None and edge in frozen_edge_set:
                continue
            if neighbor in prohibited_set:
                newV, newF, adj_list, _ = split_edge_in_mesh(newV, newF,
                                                            edge, 
                                                            VF,
                                                            adjacency_list=adj_list, 
                                                            edge_mapping_to_length=edge_set_to_length_mapping, 
                                                            adj_list_copy_dict=adj_list_copy_dict, 
                                                            modified_edge_set=modified_edge_set) 
    return newV, newF, adj_list, edge_set_to_length_mapping


'''
There are three cases when we need to update the regular space data when tracing one loop:
    1. When two vertices of the polyline are just traced -> update_regular_space_data()
    2. When one loop was just traced, we moved the current loop to traced loop and set current_loop to emptry again
       -> update_regular_space_data_after_one_loop() 
    3. When we revert to a previous state for a fallback -> revert_regular_space_data()
    3.5 This is a special case of revert. When we align two traced halfedges, the old traced path is no longer valid.
         We will remove the traced edges from the corresponding edges set. -> revert_traced_halfedge_regular_space_data()
'''
def revert_regular_space_data(regular_space_data):
    ''' Reverting happens when the tracing triggers topology violation.
        We will remove the traced edges from the corresponding edges set in regular_space_data.
        We will also remove the traced vertices from the prohibited set.
    '''
    for i in range(len(regular_space_data.current_loop)):
        regular_space_data.prohibited_vids_set.discard(regular_space_data.current_loop[i])

    traced_edge_set_vids_set = set()
    for edge in regular_space_data.traced_edge_set:
        if edge in regular_space_data.frozen_edges_set:
            regular_space_data.frozen_edges_set.remove(edge)
        traced_edge_set_vids_set.update(edge)
    regular_space_data.traced_edge_set.clear()
    if len(regular_space_data.current_loop) > 0:
        regular_space_data.prohibited_vids_set.discard(regular_space_data.current_loop[-1])
    regular_space_data.prohibited_vids_set = regular_space_data.prohibited_vids_set - traced_edge_set_vids_set
    regular_space_data.current_loop = []
    return

def revert_regular_space_data_after_non_initial_loop(regular_space_data,
                                                     loop_tracing_data):
    ''' Reverting happens when the tracing triggers topology violation.
        We will remove the traced edges from the corresponding edges set in regular_space_data.
        We will also remove the traced vertices from the prohibited set.
    '''
    for i in range(len(regular_space_data.current_loop)):
        regular_space_data.prohibited_vids_set.discard(regular_space_data.current_loop[i])
    for feature_id, traced_vids in loop_tracing_data.feature_id_to_edge_tracing_data.items():
        for j in range(len(traced_vids)-1):
            edge = create_edge(traced_vids[j], traced_vids[j+1])
            if edge in regular_space_data.traced_edge_set:
                regular_space_data.traced_edge_set.remove(edge)
            if edge in regular_space_data.frozen_edges_set:
                regular_space_data.frozen_edges_set.remove(edge)
            regular_space_data.prohibited_vids_set.discard(traced_vids[j])
        regular_space_data.prohibited_vids_set.discard(traced_vids[-1])

    regular_space_data.current_loop = []
    return 

def update_regular_space_data_after_one_loop(connectivity, regular_space_data):
    ''' Here the regular_space.current_loop is complete. We move it to the previous loop.'''
    # remove the prohibited vids that's not in the current connectivity
    valid_vids = set(int(v) for face in connectivity.F for v in face)
    regular_space_data.prohibited_vids_set = regular_space_data.prohibited_vids_set & valid_vids
    regular_space_data.previous_loops.append(regular_space_data.current_loop)
    regular_space_data.current_loop = []
    return

def update_regular_space_data(traced_path: list[int],
                              regular_space_data):
    ''' The traced_path should be up to date at this function'''
    regular_space_data.prohibited_vids_set.update(traced_path)

    for i in range(len(traced_path) - 1):
        edge = create_edge(traced_path[i], traced_path[i+1])
        regular_space_data.traced_edge_set.add(edge)
        regular_space_data.frozen_edges_set.add(edge)
        
        regular_space_data.current_loop.append(traced_path[i])
    return

def noninitial_loop_fallback_update_regular_space_data(traced_path: list[int],
                                                      regular_space_data):
    ''' the traced_path are connecting internal loop to boundary, we only need to add the traced pathto prohibited vids set and frozen edges
    '''
    for vid in traced_path:
        regular_space_data.prohibited_vids_set.add(vid)
    for i in range(len(traced_path) - 1):
        edge = create_edge(traced_path[i], traced_path[i+1])
        regular_space_data.frozen_edges_set.add(edge)
    return

def revert_traced_halfedge_regular_space_data(outdated_halfedge_traced_vids: list[int],
                                              regular_space_data):
    ''' Reverting happens when we align two traced halfedges, the old traced path is no longer valid.
        We will remove the traced edges from the corresponding edges set in regular_space_data.
        We will also remove the traced vertices from the prohibited set.
        We also need to remove the traced vertices from the current loop.
    '''
    for j in range(len(outdated_halfedge_traced_vids)-1):
        edge = create_edge(outdated_halfedge_traced_vids[j], outdated_halfedge_traced_vids[j+1])
        regular_space_data.traced_edge_set.discard(edge)
        regular_space_data.frozen_edges_set.discard(edge)
    # Vertices still incident to a remaining traced edge belong to another, still-valid
    # halfedge (e.g. the corner inherited from the previous halfedge) and must stay prohibited.
    still_traced = {v for e in regular_space_data.traced_edge_set for v in e}
    set_outdated = set(outdated_halfedge_traced_vids) - still_traced
    for v in set_outdated:
        regular_space_data.prohibited_vids_set.discard(v)
    regular_space_data.current_loop = [v for v in regular_space_data.current_loop if v not in set_outdated]
    return

def revert_surface_merged_edges_regular_space_data(surface_merged_edges: list[list[int]],
                                                   regular_space_data):
    ''' Reverting happens when we stitch two vertices together, the edges between them are no longer valid.
        We will remove the edges between the two vertices from the corresponding edges set in regular_space_data.
        We will also remove the vertices from the prohibited set.
    '''
    for edge in surface_merged_edges:
        if edge in regular_space_data.traced_edge_set:
            regular_space_data.traced_edge_set.remove(edge)
        if edge in regular_space_data.frozen_edges_set:
            regular_space_data.frozen_edges_set.remove(edge)
        regular_space_data.prohibited_vids_set.discard(edge[0])
        regular_space_data.prohibited_vids_set.discard(edge[1])
    return

def construct_patch_regular_space_from_tracing_data(regular_space_data,
                                              patch_tracing_data):
    prohibited_vids_set = set()
    traced_edge_set = set()
    for feature_id, traced_vids in patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data.items():
        prohibited_vids_set.update(set(traced_vids))
        for i in range(len(traced_vids)-1):
            v_start = traced_vids[i]
            v_end = traced_vids[i+1]
            traced_edge_set.add(create_edge(v_start, v_end))
    for loop_id, loop_tracing_data in patch_tracing_data.remaining_loops_tracing_data.items():
        for feature_id, traced_vids in loop_tracing_data.feature_id_to_edge_tracing_data.items():
            prohibited_vids_set.update(set(traced_vids))
            for i in range(len(traced_vids)-1):
                v_start = traced_vids[i]
                v_end = traced_vids[i+1]
                traced_edge_set.add(create_edge(v_start, v_end))
    regular_space_data.prohibited_vids_set = prohibited_vids_set
    # Separate sets: freezing an edge must not mark it as traced
    regular_space_data.frozen_edges_set = set(traced_edge_set)  # Note: frozen_edges_set with 's'
    regular_space_data.traced_edge_set = traced_edge_set