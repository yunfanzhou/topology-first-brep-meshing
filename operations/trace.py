# This is a rewrite of the previous various insert_*.py files.
# We observed that the patch before insertion will have intended shape of the geometry of the patch 
# (not the topology, e.g. a disk could be embedded in a cylinder).
from copy import deepcopy
from operations.align import align_stitching_halfedges_same_patch
from operations.internal_operations import embed_point_in_triangle
from utils.data_structures import EdgeTracingData, LoopTracingData, NonInitialLoopTreatmentData, FallBackLoopTracingData
from operations.insert_trace_one_loop import project_feature_edge, trace_two_vertices, trace_to_find_best_candidate, project_feature_edge_in_same_component
from utils.mesh_utils import compute_VF, create_edge, adjacency_list, edge_length_mapping
from utils.regular_space import noninitial_loop_fallback_update_regular_space_data, regular_space_dumb, regular_space_using_PRSD, revert_regular_space_data_after_non_initial_loop
from utils.regular_space import revert_surface_merged_edges_regular_space_data, revert_traced_halfedge_regular_space_data, update_regular_space_data, update_regular_space_data_after_one_loop, revert_regular_space_data, construct_patch_initial_regular_space_data
from operations.extract import all_traced_edges_in_component, components_contains_all_trace, traced_edge_separated_components, components_fullfill_2_3_4, components_fullfill_2_3, components_pass_2_4_fail_3, stitch_untraced_boundary_loops, update_connectivity_after_F_change
from operations.stitch import split_between_two_stitching_vertices, stitch_halfedges_same_patch, stitch_halfcorners_same_patch, pre_split_halfcorners
import igl
from utils.dijkstra import dijkstra_shortest_path_between_loops
import numpy as np
from utils.print import print_if_verbose

MAX_TRACE_FACTOR = 100 # if the traced distance is 100x that of the L2 distance of the feature edge, we terminate the tracing and refine the base mesh

def boundary_already_prohibited(connectivity, patch_regular_space_data):
    boundary_loops = igl.all_boundary_loop(np.array(connectivity.F))
    for loop in boundary_loops:
        for v in loop:
            if v not in patch_regular_space_data.prohibited_vids_set:
                return False
    return True

def temporarily_prohibit_boundary(connectivity, patch_regular_space_data):
    # Add the current mesh boundary to the prohibited / frozen sets so a
    # retry trace cannot hug the boundary. Returns the additions so they can
    # be reverted afterwards.
    boundary_loops = igl.all_boundary_loop(np.array(connectivity.F))
    added_vids = set()
    added_edges = set()
    for loop in boundary_loops:
        for v in loop:
            if v not in patch_regular_space_data.prohibited_vids_set:
                patch_regular_space_data.prohibited_vids_set.add(v)
                added_vids.add(v)
        for j in range(len(loop)):
            edge = create_edge(loop[j], loop[(j + 1) % len(loop)])
            if edge not in patch_regular_space_data.frozen_edges_set:
                patch_regular_space_data.frozen_edges_set.add(edge)
                added_edges.add(edge)
    return added_vids, added_edges

def restore_boundary_prohibition(patch_regular_space_data, added_vids, added_edges):
    patch_regular_space_data.prohibited_vids_set -= added_vids
    patch_regular_space_data.frozen_edges_set -= added_edges

def check_degenerate_apex_features(loop_topology_data, initial_loop_tracing_data):
    """Called once every feature edge with a polyline is traced, so only the
    degenerate (singular apex) features are left untraced. Each one is fine only
    if its apex corner is mapped to a vertex (the singularity corner mapping,
    which holds the real apex vid, else corner_id_to_vid) that is the first or
    last vid of a feature edge traced in this loop; otherwise tracing went wrong
    (e.g. it traced from a substitute vertex next to the apex) and we raise so
    the retry path runs."""
    loop_tracing_data = initial_loop_tracing_data.loop_tracing_data
    traced_ends = set()
    for fid in loop_tracing_data.traced_id_set:
        vids = loop_tracing_data.feature_id_to_edge_tracing_data.get(fid)
        if vids:
            traced_ends.update((vids[0], vids[-1]))
    loop_id = initial_loop_tracing_data.initial_loop_id
    for halfedge in loop_topology_data.halfedges:
        if not halfedge.is_degenerate:
            continue
        corner_id = halfedge.start_corner_id
        apex_vid = initial_loop_tracing_data.singularity_corner_mapping.get(
            corner_id, loop_tracing_data.corner_id_to_vid.get(corner_id))
        if apex_vid is None:
            raise ValueError(f"Degenerate feature {halfedge.feature_id} in loop {loop_id}: apex corner "
                             f"{corner_id} is not mapped to a vertex")
        if apex_vid not in traced_ends:
            raise ValueError(f"Degenerate feature {halfedge.feature_id} in loop {loop_id}: apex corner "
                             f"{corner_id} (vid {apex_vid}) is not the end of any traced feature edge")


def detect_boundary_hug_after_halfedge(V,
                                       connectivity,
                                       patch_regular_space_data,
                                       feature_id_to_polyline_V,
                                       traced_id_set,
                                       loop_topology_data,
                                       initial_loop_tracing_data):
    # Two flavors of boundary-hug, both detected here:
    #   (a) After the trace, no component contains all traced edges — the
    #       just-finished halfedge scattered traces across components.
    #   (b) Components do contain all traced edges, but the projection
    #       majority of the remaining (untraced) feature edges sits in a
    #       different component than the one the current tracing lives in.
    # If every feature edge in the loop has already been traced, this was
    # the last halfedge and there's no point reverting to "fix" anything —
    # just fall through and let after_trace_initial_loop pick the right
    # component. Also guards against empty-curve KDTree queries in case (b).
    # Degenerate feature edges (a singular apex) have an empty polyline and are
    # never traced. Once every other feature edge is traced they must be
    # anchored at the end of a traced feature (check_degenerate_apex_features).
    if not any(fid not in traced_id_set and len(P) > 0 for fid, P in feature_id_to_polyline_V.items()):
        check_degenerate_apex_features(loop_topology_data, initial_loop_tracing_data)
        return False
    F_np = np.array(connectivity.F)
    cc = traced_edge_separated_components(F_np,
                                          patch_regular_space_data.traced_edge_set,
                                          V)
    if len(cc) <= 1:
        return False
    c = components_contains_all_trace(F_np,
                                      patch_regular_space_data.traced_edge_set,
                                      cc)
    if not c:
        # Case (a): no all-trace component — strong sign the trace went
        # along a boundary into the wrong side.
        return True
    _, projection_majority_set = project_feature_edge_in_same_component(V,
                                                                         F_np,
                                                                         cc,
                                                                         feature_id_to_polyline_V,
                                                                         traced_id_set)
    if projection_majority_set is None:
        return False
    # Case (b): components partition F, so two components are equal iff
    # they share any fid. Sample one fid per all-trace component to check
    # membership against the projection-majority set.
    for comp in c:
        if len(comp) == 0:
            continue
        if int(comp[0]) in projection_majority_set:
            return False
    return True

def lookup_corner_vid(corner_id, loop_tracing_data, shared_corner_to_vid):
    # Look up a corner's vid: prefer the current loop's map (within-loop
    # reuse, e.g. successive halfedges sharing a corner), then fall back to
    # the cross-loop shared map (corners shared with the initial loop or an
    # earlier non-initial loop), else None.
    if corner_id in loop_tracing_data.corner_id_to_vid:
        return loop_tracing_data.corner_id_to_vid[corner_id]
    if shared_corner_to_vid and corner_id in shared_corner_to_vid:
        return shared_corner_to_vid[corner_id]
    return None

def revert_one_halfedge_trace(V,
                              connectivity,
                              patch_regular_space_data,
                              loop_tracing_data,
                              halfedge,
                              snapshot):
    # Undo the trace state added by the just-finished halfedge.
    restore_connectivity(V,
                         connectivity,
                         snapshot["V_len"],
                         snapshot["F_copy"])
    revert_traced_halfedge_regular_space_data(snapshot["traced_vids"],
                                              patch_regular_space_data)
    feature_id = halfedge.feature_id
    loop_tracing_data.traced_id_set.discard(feature_id)
    loop_tracing_data.feature_id_to_edge_tracing_data.pop(feature_id, None)
    loop_tracing_data.feature_id_to_segment_upsample_size.pop(feature_id, None)
    loop_tracing_data.feature_id_to_edge_orientation.pop(feature_id, None)
    for corner_id in snapshot["corner_id_keys_added"]:
        loop_tracing_data.corner_id_to_vid.pop(corner_id, None)

def reset_initial_loop_tracing_data(initial_loop_tracing_data):
    """Clear the initial loop's tracing data before a retry. Singular corners
    keep their apex vid (singularity_corner_mapping survives rollbacks), so the
    retry traces from the apex instead of inserting a new corner vertex."""
    initial_loop_tracing_data.loop_tracing_data = LoopTracingData()
    for corner_id, apex_vid in initial_loop_tracing_data.singularity_corner_mapping.items():
        initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[corner_id] = apex_vid

def trace_one_feature_edge(V,
                           connectivity, 
                           patch_regular_space_data,
                           curve_poly_V,
                           edge_tracing_return_data,
                           first_corner_vid_pair, # the corners could have been inserted before
                           second_corner_vid_pair,
                           closest_fids_matrix = None,
                           project_off_boundary = True,
                           fall_back = False,
                           ignore_trace_geometry = False):    
    '''
    The curve_poly_V is a vector of vertices. The vertex positions are ordered in the same orientation as the halfedge in the loop.
    The closest_fids_matrix is a precomputed matrix with each row the closest face ids for each vertex in curve_poly_V. 
    The closest_fids_matrix is passed in by the initial shortcut tracing (and by the non-initial loop retry after a component split). 
    Because, we need to avoid closest faces in different components. 
    But in rollback tracing, the loop will never be closed before the last vertex. So we don't need to worry about it.
    '''
    # project the poly V on to the mesh 
    if closest_fids_matrix is None:
        closest_fids_matrix = project_feature_edge(V, 
                                                   connectivity.F, 
                                                   curve_poly_V)
    prev_vid = None
    current_vid = None
    cummulated_l2_distance = 0.0
    cummulated_traced_distance = 0.0
    for i, point in enumerate(curve_poly_V):
        traced_path = None
        if i == 0:
            # first point check if the corner id has a vid already
            if first_corner_vid_pair[1] is not None:
                prev_vid = first_corner_vid_pair[1]
                edge_tracing_return_data.start_vid = first_corner_vid_pair[1]
                # inherited corner vertex: must never be a snap target for later samples of this edge
                patch_regular_space_data.prohibited_vids_set.add(prev_vid)
            else:
                prev_vid, _ = embed_point_in_triangle(
                                                    V,
                                                    connectivity,
                                                    point,
                                                    patch_regular_space_data.frozen_edges_set,
                                                    patch_regular_space_data.prohibited_vids_set,
                                                    candidate_fids=closest_fids_matrix[0],
                                                    off_boundary=project_off_boundary,
                                                    prev_vid=None,  # first sample of the edge
                                                    traced_edge_set=patch_regular_space_data.traced_edge_set)
                first_corner_vid_pair[1] = prev_vid
                patch_regular_space_data.prohibited_vids_set.add(prev_vid) 
                # important to add the embedded vid to the prohibited set to avoid tracing through it in the later tracing of the same edge
            edge_tracing_return_data.start_vid = prev_vid
            continue

        if i == len(curve_poly_V) - 1:
            # last point                            
            # if we are doing the short cut initial loop tracing, the last corner and the first corner could be the same.
            # Even if the second_corner_vid_pair has None vid, when the function was called. We need to update it now.
            if second_corner_vid_pair[1] is not None:
                current_vid = second_corner_vid_pair[1]
                edge_tracing_return_data.end_vid = second_corner_vid_pair[1]
            else:
                if not fall_back and first_corner_vid_pair[0] == second_corner_vid_pair[0] and first_corner_vid_pair[1] is not None:
                    # this is the case when we are tracing the initial loop using the short cut method.
                    current_vid = first_corner_vid_pair[1]
                    edge_tracing_return_data.end_vid = first_corner_vid_pair[1]
                else:
                    current_vid = None
        # before tracing we need to run regular_space 
        regular_space_using_PRSD(V,
                                connectivity,
                                patch_regular_space_data)
        if current_vid is not None and prev_vid != current_vid:
            # in this case, the two vertices are set and we only need to trace.
            (traced_path, 
            traced_distance) = trace_two_vertices(V,
                                                  connectivity.F,
                                                  prev_vid, 
                                                  current_vid,
                                                  patch_regular_space_data.prohibited_vids_set,
                                                  connectivity.adj_list,
                                                  connectivity.edge_set_to_length_mapping)

        elif current_vid is None:
            # if the current_vid is None, we need to find the best inserted vertex by tracing the candidates, 
            # and chose the one with shortest geodesic distance
            closest_fids_for_point = project_feature_edge(V,
                                                        connectivity.F,
                                                        [point])[0]
            (current_vid, 
            traced_path, 
            traced_distance) = trace_to_find_best_candidate(V,
                                                            connectivity,
                                                            prev_vid,
                                                            point,
                                                            patch_regular_space_data,
                                                            closest_fids_for_point,
                                                            project_off_boundary)
        else:
            raise ValueError("Both prev_vid and current_vid are None in trace_one_feature_edge. Or prev_vid equals current_vid.")
        # if this the last vertex of the edge, update the last corner mapping
        if i == len(curve_poly_V) - 1:
            second_corner_vid_pair[1] = current_vid
        if traced_path is None or len(traced_path) == 0:
            print_if_verbose(f"Tracing failed for feature edge between vids {prev_vid} and current point {point}.")
            raise ValueError("Failed to find path between two vertices during feature edge tracing.")
        # update the cummulated distances
        cummulated_traced_distance += traced_distance
        if i > 0:
            cummulated_l2_distance += np.linalg.norm(point - curve_poly_V[i-1])
        SHORT_EDGE_THRESHOLD = 1e-6
        if cummulated_l2_distance < SHORT_EDGE_THRESHOLD:
            print_if_verbose(f"Skipping trace geometry check for feature edge due to short L2 distance of {cummulated_l2_distance}. Cummulated traced distance: {cummulated_traced_distance}")

        elif cummulated_traced_distance > MAX_TRACE_FACTOR * cummulated_l2_distance and not ignore_trace_geometry:
            print_if_verbose(f"Tracing terminated early for feature edge due to excessive traced distance of factor {MAX_TRACE_FACTOR}. "
                             f"Cummulated traced distance: {cummulated_traced_distance}, Cummulated L2 distance: {cummulated_l2_distance}")
            raise ValueError("Tracing terminated early due to excessive traced distance.")
        #update tracing loop data
        update_edge_tracing_data(traced_path, edge_tracing_return_data)
        #update regular space data
        update_regular_space_data(traced_path, patch_regular_space_data)
        # set prev_vid to current_vid, and current to None
        prev_vid = current_vid
        current_vid = None

    return

def trace_initial_loop_new(V,
                       connectivity,
                       patch_regular_space_data,
                       loop_topology_data,
                       loop_geometry_data,
                       surface_merged_edges,
                       initial_loop_tracing_data,
                       ignore_trace_geometry,
                       is_last_attempt=False):
    # add singularity corners
    if initial_loop_tracing_data.singularity_corner_mapping:
        for corner_vid in initial_loop_tracing_data.singularity_corner_mapping.values():
            patch_regular_space_data.prohibited_vids_set.add(corner_vid)
    # store the initial state before the loop begins
    before_tracing_V_len = len(V)
    before_tracing_F_copy = [face[:] for face in connectivity.F]
    try:
        initial_trace_failed = trace_initial_loop_with_heuristics(V, 
                                                        connectivity, 
                                                        patch_regular_space_data,
                                                        loop_topology_data,
                                                        loop_geometry_data,
                                                        initial_loop_tracing_data,
                                                        off_boundary=True,
                                                        ignore_trace_geometry=ignore_trace_geometry)
    except ValueError as e:
        print_if_verbose(f"Initial loop tracing raised error: {e}")
        initial_trace_failed = True
    if initial_trace_failed:
        print_if_verbose("Initial loop tracing failed again in first heuristic, soft fallback.")

        #restore the saved state
        restore_connectivity(V,
                             connectivity,
                             before_tracing_V_len,
                             before_tracing_F_copy)
        reset_initial_loop_tracing_data(initial_loop_tracing_data)
        # prepare the regular space data for fallback tracing 
        # 1. clear out the previous tracing data. 
        revert_regular_space_data(patch_regular_space_data)   
        # 1.5 making sure all the boundary edges are in the prohibited vids set
        construct_patch_initial_regular_space_data(connectivity, patch_regular_space_data)
        # 2. If the singularity corners are in the mesh, we need to add the vid to prohibited set
        for corner_id, apex_vid in initial_loop_tracing_data.singularity_corner_mapping.items():
            patch_regular_space_data.prohibited_vids_set.add(apex_vid)
        # try heuristic again but with borders prohibited
        second_heuristic_failed = trace_initial_loop_with_heuristics(V,  
                                                                    connectivity,
                                                                    patch_regular_space_data,
                                                                    loop_topology_data,
                                                                    loop_geometry_data,
                                                                    initial_loop_tracing_data,
                                                                    off_boundary=True,
                                                                    ignore_trace_geometry=ignore_trace_geometry)
        if second_heuristic_failed:
            print_if_verbose("Initial loop tracing failed again in second heuristic, revert to fall back tracing.")
            # if the result violate the topology we need to do the fall back trace of the initial loop)
            #restore the saved state
            restore_connectivity(V,
                                 connectivity,
                                 before_tracing_V_len,
                                 before_tracing_F_copy)
            reset_initial_loop_tracing_data(initial_loop_tracing_data)
            # prepare the regular space data for fallback tracing 
            # 1. clear out the previous tracing data. 
            revert_regular_space_data(patch_regular_space_data)   
            # 1.5 making sure all the boundary edges are in the prohibited vids set
            construct_patch_initial_regular_space_data(connectivity, patch_regular_space_data)
            # 2. If the singularity corners are in the mesh, we need to add the vid to prohibited set
            for corner_id, apex_vid in initial_loop_tracing_data.singularity_corner_mapping.items():
                patch_regular_space_data.prohibited_vids_set.add(apex_vid)
            # 3. Add the seam merged edges (if exists) to the frozen edge set
            for edge in surface_merged_edges:
                patch_regular_space_data.frozen_edges_set.add(edge)
                patch_regular_space_data.prohibited_vids_set.add(edge[0])
                patch_regular_space_data.prohibited_vids_set.add(edge[1])
            # finally do the fall back tracing
            trace_initial_loop_fallback(V,
                                    connectivity,
                                    patch_regular_space_data,
                                    loop_topology_data,
                                    loop_geometry_data,
                                    surface_merged_edges,
                                    initial_loop_tracing_data,
                                    ignore_trace_geometry,
                                    is_last_attempt=is_last_attempt)
    
    # After all tracing have finished now add the degenerate feature edge vids to the mapping
    # only degenerate feature edges have one vid. Others have at least 2 vids
    for i, halfedge in enumerate(loop_topology_data.halfedges):
        if not halfedge.is_degenerate:
            continue
        start_cid = halfedge.start_corner_id
        end_cid = halfedge.end_corner_id
        if start_cid != end_cid and start_cid is not None and end_cid is not None:
            raise ValueError(f"Degenerate feature edge halfedge {halfedge.feature_id} has different start and end corner ids.")
        corner_id = start_cid
        vid = initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[corner_id]
        initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data[halfedge.feature_id] = [vid]
        initial_loop_tracing_data.loop_tracing_data.feature_id_to_segment_upsample_size[halfedge.feature_id] = None
        initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_orientation[halfedge.feature_id] = None

        
    update_connectivity_after_F_change(connectivity,
                                       patch_regular_space_data.current_loop,
                                       clear_backup=True)
    # update regular space data after one loop after tracing
    update_regular_space_data_after_one_loop(connectivity, 
                                             patch_regular_space_data)


def trace_initial_loop_with_heuristics(V,
                       connectivity,
                       patch_regular_space_data,
                       loop_topology_data,
                       loop_geometry_data,
                       initial_loop_tracing_data,
                       off_boundary = False,
                       ignore_trace_geometry = True):
    initial_trace_failed = False
    # we need to add the singularity corner vids to the prohibited set for regular space before tracing
    if initial_loop_tracing_data.singularity_corner_mapping:
        for corner_id, apex_vid in initial_loop_tracing_data.singularity_corner_mapping.items():
            patch_regular_space_data.prohibited_vids_set.add(apex_vid)
    # add a check that the initial loop is a simple loop 
    for i, halfedge in enumerate(loop_topology_data.halfedges):
        prev_halfedge = loop_topology_data.halfedges[i-1]
        if halfedge.start_corner_id != prev_halfedge.end_corner_id:
            raise Exception(f"Initial loop is not a simple loop. Previous halfedge end corner {prev_halfedge.end_corner_id} does not match current halfedge start corner {halfedge.start_corner_id}.")

        if halfedge.feature_id in initial_loop_tracing_data.loop_tracing_data.traced_id_set:
            continue
        # get the polyline vertices for this halfedge
        curve_poly_V = loop_geometry_data.feature_id_poly_V_True_orientation[halfedge.feature_id]
        if not halfedge.orientation:
            curve_poly_V = curve_poly_V[::-1]
        if (len(curve_poly_V) == 0):
            continue
        F_np = np.array(connectivity.F)
        # before tracing each feature edge, check if the traced edges form more than one component. 
        # And update the connectivity with the component that contains all the traced edge
        # get the closest_fids_matrix for the current curve_poly_V
        closest_fids_matrix = None
        cc = traced_edge_separated_components(F_np, 
                                                patch_regular_space_data.traced_edge_set,
                                                V)
        if len(cc) > 1:
            c = components_contains_all_trace(F_np, 
                                                patch_regular_space_data.traced_edge_set, 
                                                cc)
            if not c:
                print_if_verbose("components contain all traced edges failed. Inital loop trace fallback")
                break
            elif len(c) == 1:
                # Update the connectivity with the component that contains all the traced edge
                connectivity.F = F_np[c[0]].tolist()
                update_connectivity_after_F_change(connectivity,
                                                   patch_regular_space_data.current_loop)
            else:
                print_if_verbose("Number of components contains all traced edges more than 1. Do majority vote")
                # more than one component contains all the traced edges
                # we need to check if the projected fids are all in the same components. 
                # We might need to update the connectivity to stay in the same component for tracing
                closest_fids_matrix, chosen_set_fids = project_feature_edge_in_same_component(V,
                                                                             F_np,
                                                                             c,
                                                                             loop_geometry_data.feature_id_poly_V_True_orientation,
                                                                             initial_loop_tracing_data.loop_tracing_data.traced_id_set)
                if chosen_set_fids is not None:
                    connectivity.F = F_np[list(chosen_set_fids)].tolist()
                    update_connectivity_after_F_change(connectivity,
                                                       patch_regular_space_data.current_loop)
                
        else:
        # if the traced_edges does not separate the mesh into components. 
        # do need to do anything
            pass

        # now with the updated connectivity we can trace the feature edge
        first_corner_vid = initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[halfedge.start_corner_id] if halfedge.start_corner_id in initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid else None
        second_corner_vid = initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[halfedge.end_corner_id] if halfedge.end_corner_id in initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid else None
        edge_tracing_return_data = EdgeTracingData()
        # since the polyline is always oriented in True, we want the corners to match the insertion order
        # flip the first and the second if the halfedge orientation is not True
        first_corner_pair = [halfedge.start_corner_id, first_corner_vid]
        second_corner_pair = [halfedge.end_corner_id, second_corner_vid]
        print_if_verbose(f"    Tracing halfedge {i}/{len(loop_topology_data.halfedges)}: {halfedge.feature_id} from corner {first_corner_pair[0]} to {second_corner_pair[0]}")
        connectivity.F_len_before_tracing = len(connectivity.F)
        # Snapshot for potential revert if this trace turns out to hug the boundary.
        pre_trace_corner_ids = set(initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid.keys())
        pre_trace_V_len = len(V)
        pre_trace_F_copy = [face[:] for face in connectivity.F]
        trace_one_feature_edge(V,
                               connectivity,
                               patch_regular_space_data,
                               curve_poly_V,
                               edge_tracing_return_data,
                               first_corner_pair,
                               second_corner_pair,
                               closest_fids_matrix,
                               project_off_boundary=off_boundary,
                               ignore_trace_geometry=ignore_trace_geometry)
        # update loop tracing data from the edge tracing result
        update_loop_tracing_data(edge_tracing_return_data,
                                 first_corner_pair,
                                 second_corner_pair,
                                 initial_loop_tracing_data.loop_tracing_data,
                                 halfedge)

        # If the just-finished trace ended up on the wrong side of a
        # component split (projection-majority of the remaining feature
        # edges is in a different component), the trace likely hugged the
        # boundary. Revert this halfedge and retry it with the current
        # mesh boundary added to the prohibited / frozen sets so the
        # tracer is forced inland.
        if (not boundary_already_prohibited(connectivity, patch_regular_space_data)
                and detect_boundary_hug_after_halfedge(V,
                                                       connectivity,
                                                       patch_regular_space_data,
                                                       loop_geometry_data.feature_id_poly_V_True_orientation,
                                                       initial_loop_tracing_data.loop_tracing_data.traced_id_set,
                                                       loop_topology_data,
                                                       initial_loop_tracing_data)):
            print_if_verbose(f"    Halfedge {halfedge.feature_id} appears to hug boundary; reverting and retrying with boundary prohibited.")
            snapshot = { 
                "V_len": pre_trace_V_len,
                "F_copy": pre_trace_F_copy,
                "traced_vids": list(edge_tracing_return_data.traced_vids),
                "corner_id_keys_added": (set(initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid.keys())
                                         - pre_trace_corner_ids),
            }
            revert_one_halfedge_trace(V,
                                      connectivity,
                                      patch_regular_space_data,
                                      initial_loop_tracing_data.loop_tracing_data,
                                      halfedge,
                                      snapshot)
            added_vids, added_edges = temporarily_prohibit_boundary(connectivity,
                                                                    patch_regular_space_data)
            try:
                first_corner_vid = initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[halfedge.start_corner_id] if halfedge.start_corner_id in initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid else None
                second_corner_vid = initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[halfedge.end_corner_id] if halfedge.end_corner_id in initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid else None
                first_corner_pair = [halfedge.start_corner_id, first_corner_vid]
                second_corner_pair = [halfedge.end_corner_id, second_corner_vid]
                edge_tracing_return_data = EdgeTracingData()
                connectivity.F_len_before_tracing = len(connectivity.F)
                trace_one_feature_edge(V,
                                       connectivity,
                                       patch_regular_space_data,
                                       curve_poly_V,
                                       edge_tracing_return_data,
                                       first_corner_pair,
                                       second_corner_pair,
                                       closest_fids_matrix,
                                       project_off_boundary=off_boundary,
                                       ignore_trace_geometry=ignore_trace_geometry)
                update_loop_tracing_data(edge_tracing_return_data,
                                         first_corner_pair,
                                         second_corner_pair,
                                         initial_loop_tracing_data.loop_tracing_data,
                                         halfedge)
            finally:
                restore_boundary_prohibition(patch_regular_space_data,
                                             added_vids,
                                             added_edges)

    # check the topology after the initial loop tracing
    if not initial_trace_failed and not after_trace_initial_loop(connectivity,
                                                                 loop_topology_data,
                                                                 initial_loop_tracing_data.loop_tracing_data,
                                                                 patch_regular_space_data,
                                                                 V):
        print_if_verbose("After trace initial loop topology check failed.")
        initial_trace_failed = True
    return initial_trace_failed     

def adopt_stitched_apex(initial_loop_tracing_data, apex_vids):
    """Hard fallback: the apex vid is non-traceable, so each seam side is traced
    from its own corner vertex, the half corners are stitched, and the apex
    itself ends up outside the kept disk. Map the apex corner to the stitched
    corner vertex (singularity_corner_mapping), so the apex check before the snap
    sees the seam ending at it.
    apex_vids: corner id -> apex vid before the fallback."""
    for corner_id, old_apex in apex_vids.items():
        new_vid = initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid.get(corner_id)
        if new_vid is None or new_vid == old_apex:
            continue
        initial_loop_tracing_data.singularity_corner_mapping[corner_id] = new_vid


def trace_initial_loop_fallback(V,
                                connectivity,
                                patch_regular_space_data,
                                loop_topology_data,
                                loop_geometry_data,
                                surface_merged_edges,
                                initial_loop_tracing_data,
                                ignore_trace_geometry,
                                is_last_attempt=False):
    print_if_verbose("  Tracing initial loop using fall back method.")
    apex_vids = dict(initial_loop_tracing_data.singularity_corner_mapping)
    fallback_loop_tracing_data = FallBackLoopTracingData() 

    # Count total number of non-degenerate halfedges
    traced_halfedge_count = sum(1 for h in loop_topology_data.halfedges if not h.is_degenerate)
    traced_halfedge_idx = 0
    for i, halfedge in enumerate(loop_topology_data.halfedges):
        if halfedge.is_degenerate:
            continue
        # before tracing each feature edge, check if the traced edges form more than one component. 
        # And update the connectivity with the component that contains all the traced edge
        
        F_np = np.array(connectivity.F)
        cc = traced_edge_separated_components(F_np, 
                                              patch_regular_space_data.traced_edge_set)
        if len(cc) > 1:
            c = components_contains_all_trace(F_np, 
                                              patch_regular_space_data.traced_edge_set, 
                                              cc)
            if not c:
                print_if_verbose(" Rollback trace: Traced edges not all in component")
                raise ValueError("Traced edges do not form a valid component in initial loop fallback.")
            elif len(c) == 1:
                # Update the connectivity with the component that contains all the traced edge
                connectivity.F = F_np[c[0]].tolist()
                update_connectivity_after_F_change(connectivity,
                                                   patch_regular_space_data.current_loop)
            else:
            # more than one component contains all the traced edges
            # do not need to update the connectivity
                pass
        else:
        # if the traced_edges does not separate the mesh into components. 
        # do need to do anything
            pass

        # now with the updated connectivity we can trace the feature edge
        # always trace the edge in the orientation of the halfedge
        curve_poly_V = loop_geometry_data.feature_id_poly_V_True_orientation[halfedge.feature_id]
        if not halfedge.orientation:
            curve_poly_V = curve_poly_V[::-1]

        # we no longer use the trimesh's singulairty vertex as part of trace. Instead, the singualrity vid is non-traceable, so the mesh is truly opened to a disk.
        first_corner_vid = fallback_loop_tracing_data.halfcorner_vids[-1] if fallback_loop_tracing_data.halfcorner_vids else None

        if traced_halfedge_idx == traced_halfedge_count - 1:
            second_corner_vid = fallback_loop_tracing_data.halfcorner_vids[0] if fallback_loop_tracing_data.halfcorner_vids else None
        else:
            second_corner_vid = None

        first_corner_pair = [halfedge.start_corner_id, first_corner_vid]
        second_corner_pair = [halfedge.end_corner_id, second_corner_vid]
        print_if_verbose(f"    Tracing halfedge {i} / {len(loop_topology_data.halfedges)}: {halfedge.feature_id} from corner {first_corner_pair[0]} to {second_corner_pair[0]}")
        edge_tracing_return_data = EdgeTracingData()
        if (len(curve_poly_V) == 0):
            print_if_verbose(f"Feature edge {halfedge.feature_id} is not degenerate but curve polyline is empty.")
            raise ValueError("Feature edge is not degenerate but curve polyline is empty.")
        connectivity.F_len_before_tracing = len(connectivity.F)
        if len(loop_topology_data.halfedges) == 1:
            # in the special case where the patch only have one halfedge, the halfedge will be a closed loop.
            #  So we use reference to the first corner pair for both start and end.
            # Once the start is inserted, that becomes the end vid
            trace_one_feature_edge(V,
                                connectivity,
                                patch_regular_space_data,
                                curve_poly_V,
                                edge_tracing_return_data,
                                first_corner_pair,
                                first_corner_pair,
                                closest_fids_matrix=None,
                                fall_back=True,
                                ignore_trace_geometry=ignore_trace_geometry)
        else:
            trace_one_feature_edge(V,
                                connectivity,
                                patch_regular_space_data,
                                curve_poly_V,
                                edge_tracing_return_data,
                                first_corner_pair,
                                second_corner_pair,
                                closest_fids_matrix=None,
                                fall_back=True,
                                ignore_trace_geometry=ignore_trace_geometry)
        # update loop tracing data from the edge tracing result
        fallback_loop_tracing_data.halfedges_tracing_data.append(edge_tracing_return_data)
        if len(fallback_loop_tracing_data.halfcorner_vids) == 0:
            # For the first halfedge, append the start vid
            fallback_loop_tracing_data.halfcorner_vids.append(first_corner_pair[1])
        # Always append the end vid
        fallback_loop_tracing_data.halfcorner_vids.append(second_corner_pair[1])
        traced_halfedge_idx += 1
    # get the component that is fully boundaried by trace
    if not after_trace_initial_loop_fallback(connectivity,
                                       patch_regular_space_data):
        print_if_verbose("After trace initial loop fallback topology check failed.")
        raise ValueError("Error: The initial loop fallback tracing failed.")

    # let's update the connectivity information before stitching seam halfedges
    update_connectivity_after_F_change(connectivity,
                                       patch_regular_space_data.current_loop,
                                       clear_backup=True)
    
    # now we need to merge the seam halfedges together
    # 0. find the halfedges that correspond to the same edge
    aligning_halfedge_pairs = dict[int, list[int, int]]()
    # mapping from feature id to halfedge edge_tracing_data
    # We use the convention that first one is in True orientation, second one is in False orientation
    halfedge_index = 0
    for i, halfedge in enumerate(loop_topology_data.halfedges):
        if halfedge.is_degenerate:
            continue
        if loop_topology_data.seam_ids_set.__contains__(halfedge.feature_id):
            if halfedge.feature_id not in aligning_halfedge_pairs:
                aligning_halfedge_pairs[halfedge.feature_id] = [None, None]  # Initialize with default values
            if halfedge.orientation:
                aligning_halfedge_pairs[halfedge.feature_id][0] = halfedge_index
            else:
                aligning_halfedge_pairs[halfedge.feature_id][1] = halfedge_index
        else:
            # if it's not a seam edge, we could directly update the orientation in the output data as the orientation
            initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_orientation[halfedge.feature_id] = halfedge.orientation
        halfedge_index += 1

    # 1. align the halfedges that correspond to the same feature edge.
    for feature_id, (true_halfedge_index, false_halfedge_index) in aligning_halfedge_pairs.items():
        halfedge_tracing_data0 = fallback_loop_tracing_data.halfedges_tracing_data[true_halfedge_index]
        halfedge_tracing_data1 = fallback_loop_tracing_data.halfedges_tracing_data[false_halfedge_index]
        # revert the regular space data for the two halfedges to keep regular space data up to date for later spliting stiching edges
        revert_traced_halfedge_regular_space_data(halfedge_tracing_data0.traced_vids,
                                                  patch_regular_space_data)
        align_stitching_halfedges_same_patch(V,
                                             connectivity,
                                             halfedge_tracing_data0,
                                             halfedge_tracing_data1)
        # since the upsample size is not changing after aligning, we update the upsample size in the loop tracing data here
        initial_loop_tracing_data.loop_tracing_data.feature_id_to_segment_upsample_size[feature_id] = halfedge_tracing_data0.segment_upsample_size
    # update the regular space data after aligning all the halfedges, to update the prohibited vids set and current loop
    patch_regular_space_data.current_loop = []
    for halfedge_tracing_data in fallback_loop_tracing_data.halfedges_tracing_data:
        update_regular_space_data(halfedge_tracing_data.traced_vids,
                                  patch_regular_space_data)
    # 2. split edges between stitching halfedges to ensure manifoldness after stitch
    #   We assume there is always a path between two stitching vertices. 
    #   The stitching decision is then made based on number of edge in the path.
    #   We need one round of regular space to ensure there is always a path between two stitching vertices.
    regular_space_using_PRSD(V,
                            connectivity,
                            patch_regular_space_data)

    # we also need to remove the seam edges in the prohibited list and the frozen edge set as they might block the tracing between two stitching vertices.
    # They might also block later non-initial tracing loops.
    revert_surface_merged_edges_regular_space_data(surface_merged_edges,
                                                   patch_regular_space_data)

    # adding the singularity vids back
    if initial_loop_tracing_data.singularity_corner_mapping:
        for corner_id, apex_vid in initial_loop_tracing_data.singularity_corner_mapping.items():
            patch_regular_space_data.prohibited_vids_set.add(apex_vid)

    for feature_id, (true_halfedge_index, false_halfedge_index) in aligning_halfedge_pairs.items():
        halfedge_tracing_data0 = fallback_loop_tracing_data.halfedges_tracing_data[true_halfedge_index]
        halfedge_tracing_data1 = fallback_loop_tracing_data.halfedges_tracing_data[false_halfedge_index]

        for i in range(len(halfedge_tracing_data0.traced_vids)):
            vid0 = halfedge_tracing_data0.traced_vids[i]
            vid1 = halfedge_tracing_data1.traced_vids[::-1][i]

            split_between_two_stitching_vertices(V,
                                                 connectivity,
                                                 vid0, 
                                                 vid1,
                                                 patch_regular_space_data.prohibited_vids_set)    


    # 3. stitch the halfedges together. Note we don't stitch the two end points 
    for feature_id, (true_halfedge_index, false_halfedge_index) in aligning_halfedge_pairs.items():
        halfedge_tracing_data0 = fallback_loop_tracing_data.halfedges_tracing_data[true_halfedge_index]
        halfedge_tracing_data1 = fallback_loop_tracing_data.halfedges_tracing_data[false_halfedge_index]
        
        # reverse the halfedge_tracing_data1 to be in the same orientation as halfedge_tracing_data0
        trace1 = halfedge_tracing_data1.traced_vids[::-1]  # reverse the False orientation
        # stitch and update connectivity halfedges without handling the corners
        trace_true_orientation = stitch_halfedges_same_patch(V,
                                                             connectivity,
                                                             halfedge_tracing_data0.traced_vids,
                                                             trace1)
        # zero-area faces after the stitch fail this attempt (tolerated on the last attempt)
        _da = igl.doublearea(np.array(V), np.array(connectivity.F))
        _zero = np.where(_da == 0)[0]
        if len(_zero) > 0:
            print_if_verbose(f"[stitch_halfedges_same_patch] ZERO-AREA introduced for feature_id={feature_id}: {len(_zero)} faces {_zero.tolist()}")
            if not is_last_attempt:
                raise ValueError(f"ZERO-AREA faces after stitch_halfedges_same_patch feature_id={feature_id}")
        # update the halfedge tracing data to the stitched one
        # but onte we still need to update the corner later after the halfcorner stitching
        initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data[feature_id] = trace_true_orientation
        initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_orientation[feature_id] = True

    # 4. stitch the halfcorners (we might have one halfcorner corresponding to more than two halfedges)
    # get the corner to vids mapping
    corner_id_to_halfcorner_vids = dict[int, list[int]]()
    index_to_halfedge_tracing_data = 0 
    # we need a index cnt because when tracing we skipped the degenerate halfedges
    for halfedge in loop_topology_data.halfedges:
        if halfedge.is_degenerate:
            continue
        start_corner_id = halfedge.start_corner_id
        halfcorner_start_vid = fallback_loop_tracing_data.halfcorner_vids[index_to_halfedge_tracing_data]
        if start_corner_id not in corner_id_to_halfcorner_vids:
            corner_id_to_halfcorner_vids[start_corner_id] = []
        corner_id_to_halfcorner_vids[start_corner_id].append(halfcorner_start_vid)

        index_to_halfedge_tracing_data += 1

    for corner_id, halfcorner_vids in corner_id_to_halfcorner_vids.items():
        if len(halfcorner_vids) > 1:
            pre_split_halfcorners(V, connectivity, halfcorner_vids,
                                  prohibited_vids=patch_regular_space_data.prohibited_vids_set)
            stitched_to_vid = stitch_halfcorners_same_patch(V,
                                                            connectivity,
                                                            halfcorner_vids)
            # zero-area faces after the stitch fail this attempt (tolerated on the last attempt)
            _da = igl.doublearea(np.array(V), np.array(connectivity.F))
            _zero = np.where(_da < 1e-16)[0]
            if len(_zero) > 0:
                print_if_verbose(f"[stitch_halfcorners_same_patch] ZERO-AREA introduced for corner_id={corner_id} halfcorner_vids={halfcorner_vids}: {len(_zero)} faces {_zero.tolist()}")
                if not is_last_attempt:
                    raise ValueError(f"ZERO-AREA faces after stitch_halfcorners_same_patch corner_id={corner_id}")
        # also update the loop tracing data corner id to vid mapping
            initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[corner_id] = stitched_to_vid
        else:
            initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[corner_id] = halfcorner_vids[0]
    adopt_stitched_apex(initial_loop_tracing_data, apex_vids)
    # 5. update the loop tracing data and regular_space_data
    revert_regular_space_data(patch_regular_space_data) # we revert regular_space_data first. 
    # Can then update the loop tracing edge data one by one while updating the tracing data 
    index_to_halfedge_tracing_data = 0 
    for halfedge in loop_topology_data.halfedges:
        if halfedge.is_degenerate:
            continue
        if halfedge.feature_id in initial_loop_tracing_data.loop_tracing_data.traced_id_set:
            # this halfedge has been traced before, we skip it
            index_to_halfedge_tracing_data += 1
            continue
        feature_id = halfedge.feature_id
        if feature_id in loop_topology_data.seam_ids_set :
            # we only check the True oriented edges for seam edges
            if not halfedge.orientation:
                index_to_halfedge_tracing_data += 1
                continue
        # check if the edge tracing data is updated
        traced_vid = []
        if initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data.__contains__(feature_id):
            traced_vid = initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data[feature_id][1:-1]  # remove the two end points, we will add them later
        else:
            traced_vid = fallback_loop_tracing_data.halfedges_tracing_data[index_to_halfedge_tracing_data].traced_vids[1:-1]  # remove the two end points, we will add them later
        # check if the corner mapping is updated
        if not initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid.__contains__(halfedge.start_corner_id):
            first_corner_id = halfedge.start_corner_id
            first_corner_vid = fallback_loop_tracing_data.halfedges_tracing_data[index_to_halfedge_tracing_data].start_vid
            initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[first_corner_id] = first_corner_vid
        if not initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid.__contains__(halfedge.end_corner_id):
            second_corner_id = halfedge.end_corner_id
            second_corner_vid = fallback_loop_tracing_data.halfedges_tracing_data[index_to_halfedge_tracing_data].end_vid
            initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[second_corner_id] = second_corner_vid
        # this halfedge has been stitched, we need to update the edge tracing data  and also handle the corner id
        first_corner_vid = initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[halfedge.start_corner_id]
        second_corner_vid = initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[halfedge.end_corner_id]
        traced_vid.insert(0, first_corner_vid)
        traced_vid.append(second_corner_vid)
        # update the edge tracing data in the loop tracing data
        initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data[feature_id] = traced_vid
        initial_loop_tracing_data.loop_tracing_data.feature_id_to_segment_upsample_size[feature_id] = fallback_loop_tracing_data.halfedges_tracing_data[index_to_halfedge_tracing_data].segment_upsample_size
        initial_loop_tracing_data.loop_tracing_data.traced_id_set.add(feature_id)
        index_to_halfedge_tracing_data += 1
        # update the regular_space_data after merging. 
        update_regular_space_data(traced_vid, patch_regular_space_data)
    return 


def trace_non_initial_loop(V,
                          connectivity,
                          patch_regular_space_data,
                          loop_topology_data,
                          loop_geometry_data,
                          loop_tracing_data,
                          patch_non_initial_loop_treatment_data,
                          ignore_trace_geometry,
                          shared_corner_to_vid=None):
    non_initaial_loop_fallback = False
    after_trace_non_initial_loop_success = True
    previous_loops_need_treatment = []

    # store the initial state before the loop begins
    before_tracing_V_len = len(V)
    before_tracing_F_copy = [face[:] for face in connectivity.F]
    visited_corner_ids = set()
    for i, halfedge in enumerate(loop_topology_data.halfedges):
        if halfedge.is_degenerate:
            continue
        prev_halfedge = loop_topology_data.halfedges[i-1]
        if halfedge.start_corner_id != prev_halfedge.end_corner_id:
            raise Exception(f"Non initial loop is not a simple loop. Previous halfedge end corner {prev_halfedge.end_corner_id} does not match current halfedge start corner {halfedge.start_corner_id}.")
        if halfedge.feature_id in loop_tracing_data.traced_id_set:
            # this is possible when the interior loop shares some edge with itself, or with other loops;
            # they would just need to be merged. Not implemented, since it does not occur in the dataset
            raise Exception(f"Non initial loop contains seam edge {halfedge.feature_id}")
        visited_corner_ids.add(halfedge.start_corner_id)

        # now with the updated connectivity we can trace the feature edge
        first_corner_vid = lookup_corner_vid(halfedge.start_corner_id, loop_tracing_data, shared_corner_to_vid)
        second_corner_vid = lookup_corner_vid(halfedge.end_corner_id, loop_tracing_data, shared_corner_to_vid)
        edge_tracing_return_data = EdgeTracingData()
        # since the polyline is always oriented in True, we want the corners to match the insertion order
        # flip the first and the second if the halfedge orientation is not True
        first_corner_pair = [halfedge.start_corner_id, first_corner_vid]
        second_corner_pair = [halfedge.end_corner_id, second_corner_vid]
        print_if_verbose(f"    Tracing halfedge {halfedge.feature_id} from corner {first_corner_pair[0]}: {first_corner_pair[1]} to {second_corner_pair[0]}: {second_corner_pair[1]}")
        # get the polyline vertices for this halfedge
        curve_poly_V = loop_geometry_data.feature_id_poly_V_True_orientation[halfedge.feature_id]
        if not halfedge.orientation:
            curve_poly_V = curve_poly_V[::-1]
        try:
            connectivity.F_len_before_tracing = len(connectivity.F)
            trace_one_feature_edge(V,
                                connectivity,
                                patch_regular_space_data,
                                curve_poly_V,
                                edge_tracing_return_data,
                                first_corner_pair,
                                second_corner_pair,
                                closest_fids_matrix = None,
                                ignore_trace_geometry=ignore_trace_geometry)
        except ValueError as e:
            print_if_verbose(f"     Non-initial loop tracing on halfedge {halfedge.feature_id} failed with error: {e}")
            if "due to excessive traced distance" in str(e):
                # tracing failed due to excessive traced distance
                # we directly raise the error here
                raise e
            if "No valid path found" in str(e):
                # tracing failed due to no valid path found
                # try to check if there are multiple components formed by traced edges
                F_np = np.array(connectivity.F)
                # before tracing each feature edge, check if the traced edges form more than one component. 
                # And update the connectivity with the component that contains all the traced edge
                # get the closest_fids_matrix for the current curve_poly_V
                closest_fids_matrix = None
                cc = traced_edge_separated_components(F_np, 
                                                        patch_regular_space_data.traced_edge_set,
                                                        V)
                if len(cc) > 1:
                    c = components_contains_all_trace(F_np, 
                                                        patch_regular_space_data.traced_edge_set, 
                                                        cc)
                    if not c:
                        print_if_verbose("components contain all traced edges failed doesn't exist")
                        raise ValueError("Traced edges do not form a valid component in non-initial loop tracing. There is no valid path found")
                    elif len(c) == 1:
                        # Update the connectivity with the component that contains all the traced edge
                        connectivity.F = F_np[c[0]].tolist()
                        update_connectivity_after_F_change(connectivity,
                                                        patch_regular_space_data.current_loop)
                    else:
                        print_if_verbose("Number of components contains all traced edges more than 1. Do majority vote")
                        # more than one component contains all the traced edges
                        # we need to check if the projected fids are all in the same components. 
                        # We might need to update the connectivity to stay in the same component for tracing
                        closest_fids_matrix, chosen_set_fids = project_feature_edge_in_same_component(V,
                                                                                    F_np,
                                                                                    c,
                                                                                    loop_geometry_data.feature_id_poly_V_True_orientation,
                                                                                    loop_tracing_data.traced_id_set)
                        if chosen_set_fids is not None:
                            connectivity.F = F_np[list(chosen_set_fids)].tolist()
                            update_connectivity_after_F_change(connectivity,
                                                            patch_regular_space_data.current_loop)
                    print_if_verbose(f"     Retry tracing halfedge {halfedge.feature_id} after updating connectivity.")
                    # revert the traced edge in regular space data before retracing
                    revert_traced_halfedge_regular_space_data(edge_tracing_return_data.traced_vids,
                                                              patch_regular_space_data)
                    first_corner_vid = lookup_corner_vid(halfedge.start_corner_id, loop_tracing_data, shared_corner_to_vid)
                    second_corner_vid = lookup_corner_vid(halfedge.end_corner_id, loop_tracing_data, shared_corner_to_vid)
                    first_corner_pair = [halfedge.start_corner_id, first_corner_vid]
                    second_corner_pair = [halfedge.end_corner_id, second_corner_vid]
                    # revert the traced edges from edge_tracing_return_data
                    edge_tracing_return_data = EdgeTracingData()
                    # retry tracing
                    connectivity.F_len_before_tracing = len(connectivity.F)
                    trace_one_feature_edge(V,
                                connectivity,
                                patch_regular_space_data,
                                curve_poly_V,
                                edge_tracing_return_data,
                                first_corner_pair,
                                second_corner_pair,
                                closest_fids_matrix = closest_fids_matrix,
                                ignore_trace_geometry=ignore_trace_geometry)
                else:
                # if the traced_edges does not separate the mesh into components.
                # this is a true problem
                    raise ValueError("Traced edges do not form a valid component in non-initial loop tracing. There is no valid path found")
            else:
                # Unrecognized ValueError: don't fall through and treat a failed
                # trace as a success (which would feed an empty/partial
                # edge_tracing_return_data into update_loop_tracing_data). Re-raise.
                raise e
        # update loop tracing data from the edge tracing result
        update_loop_tracing_data(edge_tracing_return_data,
                                 first_corner_pair,
                                 second_corner_pair,
                                 loop_tracing_data,
                                 halfedge)
        # If this halfedge closed a sub-loop (returned to a previously visited
        # corner), prune the carved-off interior immediately. This keeps the
        # invariant that connectivity.F is always the disk-side component, so
        # a non-initial "loop" composed of multiple closed sub-loops doesn't
        # accumulate >2 components by the end.
        if halfedge.end_corner_id in visited_corner_ids:
            prune_success, trapped_loops = prune_after_subloop_closure(V,
                                                                       connectivity,
                                                                       patch_regular_space_data)
            if not prune_success:
                after_trace_non_initial_loop_success = False
                previous_loops_need_treatment = trapped_loops
                break
        visited_corner_ids.add(halfedge.end_corner_id)
    patch_non_initial_loop_treatment_data.loops_need_treatment = previous_loops_need_treatment
    if not after_trace_non_initial_loop_success:
        if not previous_loops_need_treatment:
            # The closure produced an unexpected component structure but no
            # previous loops are trapped — the fallback would have nothing to
            # treat, so there's no recovery path.
            raise ValueError("Non-initial loop tracing failed: sub-loop closure produced unexpected component structure with no trapped previous loops.")
        non_initaial_loop_fallback = True
        restore_connectivity(V,
                             connectivity,
                             before_tracing_V_len,
                             before_tracing_F_copy)
        # prepare the regular space data for fallback tracing 
        # 1. clear out the previous tracing data. 
        revert_regular_space_data_after_non_initial_loop(patch_regular_space_data, loop_tracing_data)
        loop_tracing_data = LoopTracingData()
        trace_non_initial_loop_fallback(V,
                                        connectivity,
                                        patch_regular_space_data,
                                        loop_topology_data,
                                        loop_geometry_data,
                                        loop_tracing_data,
                                        patch_non_initial_loop_treatment_data=patch_non_initial_loop_treatment_data,
                                        ignore_trace_geometry=ignore_trace_geometry,
                                        shared_corner_to_vid=shared_corner_to_vid)
    # After all tracing have finished now add the degenerate feature edge vids to the mapping
    # only degenerate feature edges have one vid. Others have at least 2 vids
    for i, halfedge in enumerate(loop_topology_data.halfedges):
        if not halfedge.is_degenerate:
            continue
        start_cid = halfedge.start_corner_id
        end_cid = halfedge.end_corner_id
        if start_cid != end_cid and start_cid is not None and end_cid is not None:
            raise ValueError(f"Degenerate feature edge halfedge {halfedge.feature_id} has different start and end corner ids.")
        corner_id = start_cid
        vid = loop_tracing_data.corner_id_to_vid[corner_id]
        loop_tracing_data.feature_id_to_edge_tracing_data[halfedge.feature_id] = [vid]
        loop_tracing_data.feature_id_to_segment_upsample_size[halfedge.feature_id] = None
        loop_tracing_data.feature_id_to_edge_orientation[halfedge.feature_id] = None


    update_connectivity_after_F_change(connectivity,
                                       patch_regular_space_data.current_loop,
                                       clear_backup=True)
    # update regular space data after one loop after tracing
    update_regular_space_data_after_one_loop(connectivity, 
                                             patch_regular_space_data)    
    return loop_tracing_data, non_initaial_loop_fallback

def trace_non_initial_loop_fallback(V,
                                    connectivity,
                                    patch_regular_space_data,
                                    loop_topology_data,
                                    loop_geometry_data,
                                    loop_tracing_data,
                                    patch_non_initial_loop_treatment_data,
                                    ignore_trace_geometry,
                                    shared_corner_to_vid=None):
    # when a non initial loop tracing has failed, that is when the carved hole contains other traced loops inside.
    # for each of the previously traced non-initial loops, we need to connect them to the boundary of the surface first.
    # first run regular space to make sure there is always a path
    after_trace_non_initial_loop_success = True
    previous_loop_to_treat = []
    while len(patch_non_initial_loop_treatment_data.loops_need_treatment) > 0:
        print_if_verbose(f"     Tracing non-initial loop using fall back method with previous loops: {patch_non_initial_loop_treatment_data.loops_need_treatment}")
        previous_loops_to_treat = []
        still_need_to_treat_ids = []
        for i in patch_non_initial_loop_treatment_data.loops_need_treatment:
            if i in patch_non_initial_loop_treatment_data.already_treated_non_initial_loops:
                continue
            still_need_to_treat_ids.append(i)
            previous_loops_to_treat.append(patch_regular_space_data.previous_loops[i])
        print_if_verbose(f"      still need to treat loops {still_need_to_treat_ids} ")


        for i, previous_loop in enumerate(previous_loops_to_treat):
            # run regular space on just this loop to make sure there is always a path
            regular_space_using_PRSD(V,
                                        connectivity,
                                        patch_regular_space_data)
            
            # Connect the traced loops to the boundary of the surface. 
            traced_vids = dijkstra_shortest_path_between_loops(connectivity.edge_set_to_length_mapping,
                                                                connectivity.adj_list,
                                                                previous_loop,
                                                                patch_regular_space_data.previous_loops[0],  # the initial loop is always at index 0
                                                                patch_regular_space_data.prohibited_vids_set,
                                                                patch_regular_space_data.frozen_edges_set)
            if traced_vids is None or len(traced_vids) == 0:
                
                print_if_verbose("     Retry after running full version of regular space to find path to connect traced loop to boundary during non-initial loop fallback tracing.")
                regular_space_using_PRSD(V,
                                        connectivity,
                                        patch_regular_space_data)
                traced_vids = dijkstra_shortest_path_between_loops(connectivity.edge_set_to_length_mapping,
                                                                connectivity.adj_list,
                                                                previous_loop,
                                                                patch_regular_space_data.previous_loops[0],  # the initial loop is always at index 0
                                                                patch_regular_space_data.prohibited_vids_set,
                                                                patch_regular_space_data.frozen_edges_set)
            if traced_vids is None or len(traced_vids) == 0:
                print_if_verbose("     Retry after running full version of regular space to find path to connect traced loop to boundary during non-initial loop fallback tracing.")
                regular_space_dumb(V,
                                        connectivity,
                                        patch_regular_space_data)
                traced_vids = dijkstra_shortest_path_between_loops(connectivity.edge_set_to_length_mapping,
                                                                connectivity.adj_list,
                                                                previous_loop,
                                                                patch_regular_space_data.previous_loops[0],  # the initial loop is always at index 0
                                                                patch_regular_space_data.prohibited_vids_set,
                                                                patch_regular_space_data.frozen_edges_set)
            if traced_vids is None or len(traced_vids) == 0:
                #check number of components
                F_np = np.array(connectivity.F)
                cc = traced_edge_separated_components(F_np, 
                                                        patch_regular_space_data.traced_edge_set,
                                                        V)
                print_if_verbose(f"Number of components found when tracing non-initial loop fallback: {len(cc)}")
                print_if_verbose("Error: Cannot find path to connect traced loop to boundary during non-initial loop fallback tracing.")
                raise ValueError("Error: Cannot find path to connect traced loop to boundary during non-initial loop fallback tracing.")
            # add the traced vids to the prohibited vids set and the frozen edge set
            noninitial_loop_fallback_update_regular_space_data(traced_vids,
                                                            patch_regular_space_data)
        patch_non_initial_loop_treatment_data.already_treated_non_initial_loops.update(patch_non_initial_loop_treatment_data.loops_need_treatment)
        patch_non_initial_loop_treatment_data.loops_need_treatment = []
        
        # Trace the new loop again.
        before_tracing_V_len = len(V)
        before_tracing_F_copy = [face[:] for face in connectivity.F]

        after_trace_non_initial_loop_success = True
        previous_loop_to_treat = []
        visited_corner_ids = set()
        for i, halfedge in enumerate(loop_topology_data.halfedges):
            if halfedge.feature_id in loop_tracing_data.traced_id_set:
                print_if_verbose(f"    Non initial loop fallback tracing: contain seam edges {halfedge.feature_id}")
                raise Exception(f"Non initial loop contains seam edge {halfedge.feature_id}")
            print_if_verbose(f"    Non initial loop fallback tracing: tracing halfedge {halfedge.feature_id} from corner {halfedge.start_corner_id} to {halfedge.end_corner_id}")
            visited_corner_ids.add(halfedge.start_corner_id)
            # now with the updated connectivity we can trace the feature edge
            first_corner_vid = lookup_corner_vid(halfedge.start_corner_id, loop_tracing_data, shared_corner_to_vid)
            second_corner_vid = lookup_corner_vid(halfedge.end_corner_id, loop_tracing_data, shared_corner_to_vid)
            edge_tracing_return_data = EdgeTracingData()
            # since the polyline is always oriented in True, we want the corners to match the insertion order
            # flip the first and the second if the halfedge orientation is not True
            first_corner_pair = [halfedge.start_corner_id, first_corner_vid]
            second_corner_pair = [halfedge.end_corner_id, second_corner_vid]
            print_if_verbose(f"    Tracing halfedge {halfedge.feature_id} from corner {first_corner_pair[0]} to {second_corner_pair[0]}")
            # get the polyline vertices for this halfedge
            curve_poly_V = loop_geometry_data.feature_id_poly_V_True_orientation[halfedge.feature_id]
            if not halfedge.orientation:
                curve_poly_V = curve_poly_V[::-1]
            connectivity.F_len_before_tracing = len(connectivity.F)
            trace_one_feature_edge(V,
                                connectivity,
                                patch_regular_space_data,
                                curve_poly_V,
                                edge_tracing_return_data,
                                first_corner_pair,
                                second_corner_pair,
                                closest_fids_matrix = None,
                                ignore_trace_geometry=ignore_trace_geometry)
            # update loop tracing data from the edge tracing result
            update_loop_tracing_data(edge_tracing_return_data,
                                    first_corner_pair,
                                    second_corner_pair,
                                    loop_tracing_data,
                                    halfedge)
            # Same sub-loop closure pruning as the primary path: keep the
            # disk-side component each time a sub-loop closes.
            if halfedge.end_corner_id in visited_corner_ids:
                prune_success, trapped_loops = prune_after_subloop_closure(V,
                                                                           connectivity,
                                                                           patch_regular_space_data)
                if not prune_success:
                    after_trace_non_initial_loop_success = False
                    previous_loop_to_treat = trapped_loops
                    break
            visited_corner_ids.add(halfedge.end_corner_id)

        if len(previous_loop_to_treat) > 0:
            patch_non_initial_loop_treatment_data.loops_need_treatment = previous_loop_to_treat
            restore_connectivity(V,
                             connectivity,
                             before_tracing_V_len,
                             before_tracing_F_copy)
            # prepare the regular space data for fallback tracing 
            # 1. clear out the previous tracing data. 
            revert_regular_space_data_after_non_initial_loop(patch_regular_space_data, loop_tracing_data)
            loop_tracing_data.feature_id_to_edge_tracing_data.clear()
            loop_tracing_data.feature_id_to_segment_upsample_size.clear()
            loop_tracing_data.feature_id_to_edge_orientation.clear()
            loop_tracing_data.corner_id_to_vid.clear()
            loop_tracing_data.traced_id_set.clear()

    if not after_trace_non_initial_loop_success:
        print_if_verbose(f"Error: Non initial loop fallback tracing failed. With loops needing treatment: {previous_loop_to_treat}")
        raise ValueError("Error: Non initial loop fallback tracing failed.")
    return

def is_non_initial_no_path_error(error):
    # True when a non-initial loop failed because no path could be found
    # between two corners (the two corners are walled into separate
    # components by previously traced loops / boundary). Excludes unrelated
    # failures like nonmanifold meshes or excessive traced distance.
    msg = str(error).lower()
    return ("no valid path found" in msg
            or "there is no valid path found" in msg
            or "failed to find path between two vertices" in msg
            or "no path found" in msg)

def is_sub_loop_closure_error(error):
    # True when a non-initial loop failed because a sub-loop closed and trapped
    # previously traced loops inside. Excludes unrelated failures like
    # nonmanifold meshes or excessive traced distance.
    msg = str(error).lower()
    return ("sub-loop closure produced unexpected component structure with no trapped previous loops." in msg)


def snapshot_regular_space_data(regular_space_data):
    # deepcopy → the snapshot shares no mutable container with the live data,
    # so later .add/.discard/.update/.append or attribute reassignment cannot
    # touch it. All leaves are immutable (int / edge-tuple); only the
    # containers need copying, and deepcopy handles every nesting level.
    return deepcopy((regular_space_data.prohibited_vids_set,
                     regular_space_data.frozen_edges_set,
                     regular_space_data.traced_edge_set,
                     regular_space_data.previous_loops,
                     regular_space_data.current_loop))


def restore_regular_space_data(regular_space_data, snapshot):
    # deepcopy again so the live attrs don't alias the snapshot, keeping the
    # snapshot reusable for a possible second restore.
    (prohibited_vids_set,
     frozen_edges_set,
     traced_edge_set,
     previous_loops,
     current_loop) = deepcopy(snapshot)
    regular_space_data.prohibited_vids_set = prohibited_vids_set
    regular_space_data.frozen_edges_set = frozen_edges_set
    regular_space_data.traced_edge_set = traced_edge_set
    regular_space_data.previous_loops = previous_loops
    regular_space_data.current_loop = current_loop


def trace_remaining_non_initial_loops(connectivity,
                                      patch_input_data,
                                      regular_space_data,
                                      patch_tracing_data,
                                      summary,
                                      ignore_trace_geometry,
                                      disable_corner_reuse):
    # Trace every non-initial loop in the patch once. Returns the number of
    # loops that needed their own (per-loop) fallback. Raises on failure; the
    # caller decides whether to revert all loops and retry without reuse.
    non_initial_loop_fallback_cnt = 0
    non_initial_loop_treatment_data = NonInitialLoopTreatmentData()
    for loop_id in patch_input_data.patch_loop_topology_data:
        if loop_id == patch_input_data.initial_loop_id:
            continue
        print_if_verbose(f"    Tracing non-initial loop {loop_id}...")
        loop_tracing_data = LoopTracingData()
        # Build cross-loop corner map: corners shared with the initial loop
        # or any earlier-traced non-initial loop should reuse the existing
        # vid rather than insert a duplicate (which edit_loops_with_shared_corners
        # later collapses, often producing zero-area triangles). When
        # disable_corner_reuse is set (revert-all fallback), skip the reuse so
        # each loop embeds fresh corners off the existing traced walls.
        shared_corner_to_vid = None
        if not disable_corner_reuse:
            shared_corner_to_vid = dict(patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid)
            for prev_loop_data in patch_tracing_data.remaining_loops_tracing_data.values():
                shared_corner_to_vid.update(prev_loop_data.corner_id_to_vid)
        loop_tracing_data, non_initial_loop_fallback = trace_non_initial_loop(patch_input_data.V,
                               connectivity,
                               regular_space_data,
                               patch_input_data.patch_loop_topology_data[loop_id],
                               patch_input_data.patch_loop_geometry_data[loop_id],
                               loop_tracing_data,
                               non_initial_loop_treatment_data,
                               ignore_trace_geometry=ignore_trace_geometry,
                               shared_corner_to_vid=shared_corner_to_vid)
        patch_tracing_data.remaining_loops_tracing_data[loop_id] = loop_tracing_data
        if non_initial_loop_fallback:
            non_initial_loop_fallback_cnt += 1
        mainfold = igl.is_edge_manifold(np.array(connectivity.F))
        if not mainfold:
            print_if_verbose(f"Error: The mesh is not edge manifold after tracing the non-initial loop {loop_id}.")
            summary["ERROR"] = "Nonmanifold"
            raise ValueError(f"Error: The mesh is not edge manifold after tracing the non-initial loop {loop_id}.")
    return non_initial_loop_fallback_cnt


def trace_all_loops_one_patch(connectivity,
                              patch_input_data,
                              regular_space_data,
                              patch_tracing_data,
                              summary= None,
                              ignore_trace_geometry=False,
                              is_last_attempt=False):
    print_if_verbose(" Tracing initial loop...")
    # trace initial loop.
    mainfold = igl.is_edge_manifold(np.array(connectivity.F))
    if not mainfold:
        print_if_verbose("Error: The input mesh is not edge manifold before tracing the initial loop.")
        summary["ERROR"] = "Nonmanifold"
        raise ValueError("Error: The input mesh is not edge manifold before tracing the initial loop.")
    initial_fallback_used =  trace_initial_loop_new(patch_input_data.V,
                                                connectivity,
                                                regular_space_data,
                                                patch_input_data.patch_loop_topology_data[patch_input_data.initial_loop_id],
                                                patch_input_data.patch_loop_geometry_data[patch_input_data.initial_loop_id],
                                                patch_input_data.surface_merged_edges,
                                                patch_tracing_data.initial_loop_tracing_data,
                                                ignore_trace_geometry=ignore_trace_geometry,
                                                is_last_attempt=is_last_attempt)
    summary["initial_loop_fallback"] = initial_fallback_used
    mainfold = igl.is_edge_manifold(np.array(connectivity.F))
    if not mainfold:
        print_if_verbose("Error: The mesh is not edge manifold after tracing the initial loop.")
        summary["ERROR"] = "Nonmanifold"
        raise ValueError("Error: The mesh is not edge manifold after tracing the initial loop.")
    # trace the remaining non-initial loops.
    # trace_non_initial_loop already retries a failed halfedge a few times
    # (component recovery). If a no-path error still escapes, the two corners
    # are walled into separate components -- often because a reused corner sits
    # on a previous loop and is locally strangled by prohibited neighbors.
    # In that case revert *all* non-initial loops back to the post-initial-loop
    # state and retrace them once with corner reuse disabled, so each loop
    # embeds fresh corners. Snapshot that state now: V/F via restore_connectivity
    # below, regular_space_data via deepcopy.
    before_non_initial_V_len = len(patch_input_data.V)
    before_non_initial_F_copy = [face[:] for face in connectivity.F]
    before_non_initial_rsd = snapshot_regular_space_data(regular_space_data)

    disable_corner_reuse = False
    while True:
        try:
            non_initial_loop_fallback_cnt = trace_remaining_non_initial_loops(
                connectivity,
                patch_input_data,
                regular_space_data,
                patch_tracing_data,
                summary,
                ignore_trace_geometry,
                disable_corner_reuse)
            break
        except ValueError as e:
            if disable_corner_reuse or (not is_non_initial_no_path_error(e) and not is_sub_loop_closure_error(e)):
                # Already retried without reuse, or this is not a path-finding
                # failure (e.g. nonmanifold / excessive distance): no recovery.
                raise
            print_if_verbose(f"    Non-initial loop tracing failed with a no-path error: {e}. "
                             f"Reverting all non-initial loops and retracing without corner reuse.")
            restore_connectivity(patch_input_data.V,
                                 connectivity,
                                 before_non_initial_V_len,
                                 before_non_initial_F_copy)
            restore_regular_space_data(regular_space_data, before_non_initial_rsd)
            patch_tracing_data.remaining_loops_tracing_data.clear()
            disable_corner_reuse = True
    summary["non_initial_loop_fallback_cnt"] = non_initial_loop_fallback_cnt
    summary["non_initial_corner_reuse_disabled"] = disable_corner_reuse
    # Return the traced (unconsolidated) mesh; V is patch_input_data.V, modified in place.
    # The caller consolidates it later, before the linear solve.
    V = patch_input_data.V
    F = connectivity.F
    return V, F

def after_trace_initial_loop(connectivity,
                             loop_topology_data,
                             loop_tracing_data,
                             patch_regular_space_data,
                             V = None):
    ''' The function returns True and connectivity data updated if exactly one component meets
    conditions 2-4 (see operations/extract.py), or if exactly one annulus that meets 2 and 4 can be
    repaired by stitching its untraced holes. Otherwise, returns False.
    '''
    disk_component = None
    # 1. Find all the connected components
    cc = traced_edge_separated_components(connectivity.F, 
                                          patch_regular_space_data.traced_edge_set)
    F_np = np.array(connectivity.F)
    disk_component = components_fullfill_2_3_4(F_np, 
                                                loop_topology_data,
                                                loop_tracing_data,
                                                patch_regular_space_data.traced_edge_set,
                                                cc)

    if len(disk_component) == 1:
        # update the connectivity if we find the disk after tracing initial loop
        # convert the numpy array to a list
        disk_F = disk_component[0]
        connectivity.F = F_np[disk_F].tolist()
        return True

    # Fallback: no clean disk. A component may still be a disk-with-holes (annulus)
    # that meets conditions 2 & 4 and only fails 3 because of untraced boundary
    # loop(s). Repair it by stitching those holes closed with a triangle fan instead
    # of rejecting it. Only act when exactly one such component exists; otherwise we
    # cannot tell which side of the loop is the disk, so fail as before.
    stitchable = components_pass_2_4_fail_3(F_np,
                                            loop_topology_data,
                                            loop_tracing_data,
                                            patch_regular_space_data.traced_edge_set,
                                            cc)
    if len(stitchable) != 1:
        print_if_verbose(f"After tracing initial loop, the mesh is separated into {len(cc)} components, "
                         f"but {len(disk_component)} clean disks and {len(stitchable)} stitchable annuli were found.")
        return False

    disk_faces = F_np[stitchable[0]].tolist()
    stitched_F, ok = stitch_untraced_boundary_loops(V,
                                                     disk_faces,
                                                     patch_regular_space_data.traced_edge_set)
    if not ok:
        print_if_verbose("After tracing initial loop, the annulus has a partially-traced boundary loop; not stitching.")
        return False
    connectivity.F = stitched_F
    rebuild_connectivity_after_stitch(connectivity, V)
    return True

def rebuild_connectivity_after_stitch(connectivity, V):
    '''Rebuild VF, adjacency list and edge-length mapping from scratch after the
    stitch added new apex vertices and fan faces - update_connectivity_after_F_change
    only handles face removal, not added geometry. Backup state is cleared because
    the initial loop is fully traced by this point (mirrors the clear_backup path).'''
    connectivity.VF = compute_VF(connectivity.F)
    connectivity.adj_list = adjacency_list(connectivity.F, len(V))
    connectivity.edge_set_to_length_mapping = edge_length_mapping(V, connectivity.F)
    connectivity.adj_list_copy_dict = {}
    connectivity.modified_edge_set = set()

def after_trace_initial_loop_fallback(connectivity,
                                      patch_regular_space_data):
    '''
    This function extract the component that is entirely bounded by only the traced loop
    '''
    disk_component = None
    # 1. find all the connected components separated by the traced edges
    cc = traced_edge_separated_components(connectivity.F, patch_regular_space_data.traced_edge_set)
    F_np = np.array(connectivity.F)

    disk_component = components_fullfill_2_3(F_np,
                                              patch_regular_space_data.traced_edge_set,
                                              cc)
    # If there's more than one component that fulfills the conditions, we want the one the does not contain the untraceable edges/vertices
    if len(disk_component) > 1:
        frozen_vids_set = set[int]()
        untraceable_edge_set = patch_regular_space_data.frozen_edges_set - patch_regular_space_data.traced_edge_set
        # congregate all the frozen edges in a vertex set
        for edge in untraceable_edge_set:
            frozen_vids_set.add(edge[0])
            frozen_vids_set.add(edge[1])
        for component in disk_component:
            contains_prohibited = False
            face_set = set(F_np[component].flatten().tolist())
            if len(face_set.intersection(frozen_vids_set)) > 0:
                contains_prohibited = True
                continue
            if not contains_prohibited:
                disk_component = [component]
                break
    # update the connectivity if we find the disk after tracing initial loop 
    # convert the numpy array to a list
    disk_F = disk_component[0]
    connectivity.F = F_np[disk_F].tolist()
    return True

def prune_after_subloop_closure(V,
                                connectivity,
                                patch_regular_space_data):
    # Component check applied mid-loop.
    # When a halfedge ends at a corner that has already been visited in this
    # loop's traversal, a sub-loop has just closed and split the disk into 2
    # components. Keep the one with all traced edges; the other is the
    # sub-loop's interior and is safe to discard unless it contains a
    # previous loop.
    F_np = np.array(connectivity.F)
    cc = traced_edge_separated_components(F_np,
                                          patch_regular_space_data.traced_edge_set,
                                          V)
    if len(cc) != 2:
        print_if_verbose(f"   Sub-loop closure: expected 2 components, got {len(cc)}.")
        return False, find_previous_loops_in_component(patch_regular_space_data, cc, F_np)
    disk_component = None
    other_components = []
    for c in cc:
        if all_traced_edges_in_component(patch_regular_space_data.traced_edge_set,
                                         F_np[c]):
            disk_component = c
        else:
            other_components.append(c)
    if disk_component is None:
        print_if_verbose("   Sub-loop closure: no component contains all traced edges.")
        return False, find_previous_loops_in_component(patch_regular_space_data, cc, F_np)
    trapped_loops = find_previous_loops_in_component(patch_regular_space_data,
                                                     other_components,
                                                     F_np)
    if trapped_loops:
        print_if_verbose(f"   Sub-loop closure would trap previous loops {trapped_loops}.")
        return False, trapped_loops
    connectivity.F = F_np[disk_component].tolist()
    update_connectivity_after_F_change(connectivity,
                                       patch_regular_space_data.current_loop)
    return True, []

def find_previous_loops_in_component(patch_regular_space_data,
                                     cc,
                                     F_np):
    previous_loops_ids = []
    for c in cc:
        F_subset = F_np[c]
        # Use vids is enough. no need to use edges
        vids_in_component = set(F_subset.flatten().tolist())
        # use initial loop to check first. If the initial loop is in the component, then it's the component that should be kept.
        # If the initial loop is not in the component, then we check other previous loops one by one.
        initial_loop_vids = set(patch_regular_space_data.previous_loops[0])
        if initial_loop_vids.issubset(vids_in_component):
            continue
        for i in range(1, len(patch_regular_space_data.previous_loops)):
            loop_vids = set(patch_regular_space_data.previous_loops[i])
            if loop_vids.issubset(vids_in_component):
                previous_loops_ids.append(i)
    return previous_loops_ids


def restore_connectivity(V,
                         connectivity,
                         before_tracing_V_len,
                         before_tracing_F_copy):
    # Restore the connectivity to the state before tracing
    V[:] = V[:before_tracing_V_len]
    # Make a deep copy to avoid modifying the backup on subsequent operations
    connectivity.F = [face[:] for face in before_tracing_F_copy]
    new_adj_list = adjacency_list(connectivity.F, max_coeff=before_tracing_V_len)
    connectivity.adj_list = new_adj_list
    # restore edge length mapping
    connectivity.edge_set_to_length_mapping = edge_length_mapping(V, connectivity.F)
    connectivity.VF = compute_VF(connectivity.F)
def update_edge_tracing_data(traced_path,
                             edge_tracing_data): 
    for vid in traced_path:
        if len(edge_tracing_data.traced_vids) > 0 and edge_tracing_data.traced_vids[-1] == vid:
            continue
        edge_tracing_data.traced_vids.append(vid)
    # upsample size removes the first and the last vertices of the traced.
    if len(traced_path) < 2:
        raise ValueError(f"Traced path {traced_path} has fewer than 2 vertices.")
    edge_tracing_data.segment_upsample_size.append(len(traced_path) - 2)
    return

def update_loop_tracing_data(edge_tracing_data,
                             first_corner_vid_tuple,
                             second_corner_vid_tuple,
                             loop_tracing_data,
                             halfedge):
    # Ensure the key exists in corner_id_to_vid
    if first_corner_vid_tuple[0] not in loop_tracing_data.corner_id_to_vid:
        loop_tracing_data.corner_id_to_vid[first_corner_vid_tuple[0]] = first_corner_vid_tuple[1]
    if second_corner_vid_tuple[0] not in loop_tracing_data.corner_id_to_vid:
        loop_tracing_data.corner_id_to_vid[second_corner_vid_tuple[0]] = second_corner_vid_tuple[1]
    feature_edge_id = halfedge.feature_id
    # Add the feature_edge id to the traced edge id set
    loop_tracing_data.traced_id_set.add(feature_edge_id)
    # Update the loop tracing data
    loop_tracing_data.feature_id_to_edge_tracing_data[feature_edge_id] = edge_tracing_data.traced_vids
    loop_tracing_data.feature_id_to_segment_upsample_size[feature_edge_id] = edge_tracing_data.segment_upsample_size
    loop_tracing_data.feature_id_to_edge_orientation[feature_edge_id] = halfedge.orientation
