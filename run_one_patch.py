from operations.extract import traced_edge_separated_components
from operations.trace import trace_all_loops_one_patch
from operations.preprocess import find_degenerate_corners
from utils.data_structures import AggregatedPatchTracingData, ConnectivityMaintainance, PatchTracingData, PatchRegularSpaceData
from utils.mesh_utils import adjacency_list, compute_VF, create_edge, edge_length_mapping
from utils.regular_space import regular_space_dumb, construct_patch_regular_space_from_tracing_data
from utils.print import print_if_verbose
from operations.snapping import solve_one_patch_linear_system
from operations.consolidate import consolidate_one_patch
from operations.insert_trace_one_loop import trace_two_vertices
from operations.internal_operations import safe_collapse_with_frozen_edges
import igl
import numpy as np
import utils.config as config
ATTEMPT_MAX = 3

def construct_connectivity_data(V, F):
    vertex_to_faces = compute_VF(F)
    adj_list = adjacency_list(F,  len(V))
    edge_set_to_length_mapping = edge_length_mapping(V, F)
    adj_list_copy_dict = {}
    modified_edge_set = set()
    connectivity =  ConnectivityMaintainance(F, 
                                            vertex_to_faces, 
                                            adj_list, 
                                            edge_set_to_length_mapping, 
                                            adj_list_copy_dict, 
                                            modified_edge_set)
    return connectivity

def aggregate_patch_tracing_data(patch_tracing_data):
    aggregated_data = AggregatedPatchTracingData()
    for feature_id, traced_vids in patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data.items():
        aggregated_data.feature_id_to_traced_vids[feature_id] = traced_vids
        aggregated_data.feature_id_to_segment_upsample_size[feature_id] =  patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.feature_id_to_segment_upsample_size[feature_id]
        aggregated_data.feature_id_to_orientation[feature_id] = patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_orientation[feature_id]
    aggregated_data.corner_id_to_vid = patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid
    for loop_id, loop_tracing_data in patch_tracing_data.remaining_loops_tracing_data.items():
        for feature_id, traced_vids in loop_tracing_data.feature_id_to_edge_tracing_data.items():
            aggregated_data.feature_id_to_traced_vids[feature_id] = traced_vids
            aggregated_data.feature_id_to_segment_upsample_size[feature_id] = loop_tracing_data.feature_id_to_segment_upsample_size[feature_id]
            aggregated_data.feature_id_to_orientation[feature_id] = loop_tracing_data.feature_id_to_edge_orientation[feature_id]
        for corner_id, vid in loop_tracing_data.corner_id_to_vid.items():
            aggregated_data.corner_id_to_vid[corner_id] = vid
    # Sanity check: alignment slices traced_vids by these sizes, so they must be consistent.
    for feature_id, sizes in aggregated_data.feature_id_to_segment_upsample_size.items():
        if sizes is None:
            continue  # degenerate (single-corner) feature edge
        traced_vids = aggregated_data.feature_id_to_traced_vids[feature_id]
        if any(s < 0 for s in sizes) or sum(sizes) + len(sizes) + 1 != len(traced_vids):
            raise ValueError(f"Inconsistent tracing data for feature edge {feature_id}: "
                             f"segment_upsample_size={sizes}, len(traced_vids)={len(traced_vids)}")
    return aggregated_data

def check_apexes_on_traced_features(patch_tracing_data):
    """A singular apex is held in the snap only through the traced feature that
    ends at it (its own degenerate feature has no polyline). If no traced feature
    reaches it, tracing went wrong (e.g. a substitute corner vertex next to the
    apex) and the snap would move the apex off the tip: fail instead."""
    loops = [patch_tracing_data.initial_loop_tracing_data.loop_tracing_data]
    loops += list(patch_tracing_data.remaining_loops_tracing_data.values())
    traced = set()
    for loop in loops:
        for feature_id, traced_vids in loop.feature_id_to_edge_tracing_data.items():
            if loop.feature_id_to_segment_upsample_size.get(feature_id) is not None:
                traced.update(traced_vids)
    for corner_id, apex_vid in patch_tracing_data.initial_loop_tracing_data.singularity_corner_mapping.items():
        if apex_vid not in traced:
            raise ValueError(f"Singular apex vid {apex_vid} (corner {corner_id}) is not on any traced feature edge")

def solve_patch_linear_system(V,
                              F,
                              patch_data,
                              patch_tracing_data,
                              summary=None,
                              ignore_errors=False):
    check_apexes_on_traced_features(patch_tracing_data)
    consolidated_V, consolidated_F = consolidate_one_patch(V, F, patch_tracing_data)
    linear_solved_V = solve_one_patch_linear_system(
            consolidated_V,
            consolidated_F,
            patch_tracing_data,
            patch_data.patch_loop_geometry_data,
            summary=summary,
            ignore_errors=ignore_errors
        )
    connectivity = construct_connectivity_data(linear_solved_V, consolidated_F)
    return linear_solved_V, connectivity
    
def trace_and_collapse_between_two_vertices(V,
                                              connectivity,
                                              corner_id,
                                              vids,
                                              corners_to_process,
                                              regular_space_data,
                                              patch_tracing_data):
    """
    Trace and collapse paths between corner vids to merge them.

    Args:
        corners_to_process: dict of corner_id -> vids for corners still pending processing

    Returns:
        set of corner_ids that would be separated by this operation,
        or None if successful
    """
    apex_vid = vids[0]

    # Do one regular space to ensure the path between corners exists
    regular_space_dumb(V,
                        connectivity,
                        regular_space_data)

    # Find shortest path between apex_vid and the other vids
    for i in range(len(vids)-1):
        v_end = vids[i+1]
        path_vids, distance = trace_two_vertices(V, connectivity.F,
                                                    start_vid=apex_vid,
                                                    end_vid=v_end,
                                                    tracing_prohibited_vids_set=regular_space_data.prohibited_vids_set,
                                                    adj_list=connectivity.adj_list,
                                                    edge_lengths=connectivity.edge_set_to_length_mapping)

        # Freeze the path edges
        path_edges = set()
        for j in range(len(path_vids)-1):
            edge_start = path_vids[j]
            edge_end = path_vids[j+1]
            path_edges.add(create_edge(edge_start, edge_end))

        # Check if this path would separate other pending corners
        test_frozen_edges = regular_space_data.frozen_edges_set.union(path_edges)
        separated_components = traced_edge_separated_components(connectivity.F, test_frozen_edges)

        if len(separated_components) > 1:
            # Check which other pending corners would be separated
            F_np = np.array(connectivity.F)
            separated_corner_ids = set()

            for other_corner_id, other_vids in corners_to_process.items():
                if other_corner_id == corner_id:
                    continue
                # Check if the vids of this corner are in different components
                vid_components = []
                for vid in other_vids:
                    for comp_idx, comp in enumerate(separated_components):
                        if vid in set(F_np[comp].flatten()):
                            vid_components.append(comp_idx)
                            break
                # If vids are in different components, this corner would be separated
                if len(set(vid_components)) > 1:
                    separated_corner_ids.add(other_corner_id)

            if separated_corner_ids:
                print_if_verbose(f" Path would separate corners {separated_corner_ids}, deferring corner {corner_id}")
                return separated_corner_ids

        # Safe to proceed - add path edges to frozen set
        regular_space_data.frozen_edges_set.update(path_edges)

        # Collapse edges along the path
        edge_start = apex_vid
        for j in range(len(path_vids)-1):
            edge_end = path_vids[j+1]
            print_if_verbose(f" Collapsing edge from {edge_start} to {edge_end} to apex vid {apex_vid} ")
            safe_collapse_with_frozen_edges(V,
                                        connectivity,
                                        [edge_start, edge_end],
                                        collapse_to_position=V[apex_vid],
                                        collapse_to_vid=edge_start,
                                        frozen_edge_set=regular_space_data.frozen_edges_set)

    # Update the tracing data
    for loop_id, loop_tracing_data in patch_tracing_data.remaining_loops_tracing_data.items():
        if corner_id in loop_tracing_data.corner_id_to_vid:
            loop_tracing_data.corner_id_to_vid[corner_id] = apex_vid
        for feature_id, traced_vids in loop_tracing_data.feature_id_to_edge_tracing_data.items():
            for idx in {0, len(traced_vids)-1}:
                if traced_vids[idx] in vids:
                    traced_vids[idx] = apex_vid
    if corner_id in patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid:
        patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[corner_id] = apex_vid

    for feature_id, traced_vids in patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data.items():
        for idx in {0, len(traced_vids)-1}:
            if traced_vids[idx] in vids:
                traced_vids[idx] = apex_vid

    return None  # Success


def edit_loops_with_shared_corners(V, F,
                                     regular_space_data,
                                     patch_tracing_data):
    connectivity = construct_connectivity_data(V, F)

    # Build corners_to_process: dict of corner_id -> vids for corners that need stitching.
    # Trace-time corner reuse (shared_corner_to_vid) can leave the same vid
    # registered under one corner_id in multiple loops, so deduplicate per
    # corner before deciding whether stitching is actually needed.
    patch_corners = dict()
    for corner_id, vid in patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid.items():
        patch_corners[corner_id] = [vid]
    for loop_id, loop_tracing_data in patch_tracing_data.remaining_loops_tracing_data.items():
        for corner_id, vid in loop_tracing_data.corner_id_to_vid.items():
            if corner_id in patch_corners:
                if vid not in patch_corners[corner_id]:
                    patch_corners[corner_id].append(vid)
            else:
                patch_corners[corner_id] = [vid]

    # Only keep corners that have more than 1 distinct vid (need stitching)
    corners_to_process = {cid: vids for cid, vids in patch_corners.items() if len(vids) > 1}

    # Keep processing until all corners are done
    max_iterations = len(corners_to_process) * len(corners_to_process) + 1  # Safety bound
    iteration = 0

    while corners_to_process and iteration < max_iterations:
        iteration += 1
        made_progress = False

        for corner_id in list(corners_to_process.keys()):
            vids = corners_to_process[corner_id]

            # Rebuild prohibited_vids and traced_edge_set from current tracing data
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
            regular_space_data.frozen_edges_set = traced_edge_set
            regular_space_data.traced_edge_set = traced_edge_set

            separated_corners = trace_and_collapse_between_two_vertices(
                V,
                connectivity,
                corner_id,
                vids,
                corners_to_process,
                regular_space_data,
                patch_tracing_data)

            if separated_corners is None:
                # Success - remove from pending
                del corners_to_process[corner_id]
                made_progress = True
                print_if_verbose(f" Successfully processed corner {corner_id}")
            else:
                print_if_verbose(f" Deferring corner {corner_id}, would separate {separated_corners}")

        if not made_progress and corners_to_process:
            raise Exception(f"No progress made in iteration {iteration}. Remaining corners: {list(corners_to_process.keys())}")

    if corners_to_process:
        raise Exception(f"Could not process all corners after {max_iterations} iterations. Remaining: {list(corners_to_process.keys())}")

    return V, connectivity.F

def local_step_one_patch(patch_id,
                         patch_data,
                         patch_singularity,
                         all_corners_for_part,
                         summary=None,
                         ignore_errors=True,
                         attempt = ATTEMPT_MAX,
                         ignore_trace_geometry=False
                         ):
    print_if_verbose(f"///// Start processing patch {patch_id} /////")
    # Label any oversized-patch dumps (runaway regular-space refinement) with this id.
    config.current_patch_id = patch_id
    connectivity = construct_connectivity_data(patch_data.V, patch_data.F)
    patch_tracing_data = PatchTracingData()
    patch_tracing_data.initial_loop_tracing_data.initial_loop_id = patch_data.initial_loop_id
    # --- 1. Map degenerate corners to singularity apex vids ---
    patch_degenerate_corner_id_set = find_degenerate_corners(patch_data)
    if patch_singularity is not None:
        for corner in patch_degenerate_corner_id_set:
            corner_position = all_corners_for_part.corner_id_to_position[corner]
            if corner_position is not None:
                for singularity in patch_singularity:
                    if np.allclose(singularity.position, corner_position):
                        apex_vid = singularity.singularity_vids[0]
                        patch_tracing_data.initial_loop_tracing_data.singularity_corner_mapping[corner] = apex_vid
                        patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid[corner] = apex_vid
                        break

    # Check mesh manifoldness after preprocessing
    if not igl.is_edge_manifold(np.array(connectivity.F)):
        print_if_verbose(f"Error: The mesh is not edge manifold after preprocessing the patch {patch_id}.")
        summary["value_error_found"]["non_manifold_error"] = True
        raise ValueError(f"Error: The mesh is not edge manifold after preprocessing the patch {patch_id}.")
        # mark as preprocessing failure
    print_if_verbose(f"///// Preprocessed patch {patch_id} Success /////")

    # --- 2. Trace loops ---
    print_if_verbose(f"--- Tracing loops for patch {patch_id} ---")
    connectivity = construct_connectivity_data(patch_data.V, connectivity.F)
    regular_space_data = PatchRegularSpaceData()
    V, F = trace_all_loops_one_patch(
        connectivity,
        patch_data,
        regular_space_data,
        patch_tracing_data,
        summary,
        ignore_trace_geometry=ignore_trace_geometry,
        is_last_attempt=(attempt == ATTEMPT_MAX - 1)
    )

    # --- 2.5 post trace to enforce topology ----
    print_if_verbose(f"--- Post-trace processing for patch {patch_id} ---")
    # When tracing, loops with shared corners need to be pinched together.
    V, F = edit_loops_with_shared_corners(V, F,
                                          regular_space_data,
                                          patch_tracing_data)

    # //// do one last simplicial embedding to ensure no triangel have all vertices as traced
    regular_space_data = PatchRegularSpaceData()
    construct_patch_regular_space_from_tracing_data(regular_space_data,
                                                  patch_tracing_data)
    print_if_verbose("----One last simplicial embedding /////")
    connectivity = construct_connectivity_data(V, F)
    regular_space_dumb(V,
                        connectivity,
                        regular_space_data)
    print_if_verbose(f"///// Traced patch {patch_id} Success /////")
    # --- 3. Solve linear system ---
    print_if_verbose(f"--- Solving linear system for patch {patch_id} ---")
    linear_solved_V, consolidated_conn = solve_patch_linear_system(V,
                                                                connectivity.F, 
                                                                patch_data=patch_data,
                                                                patch_tracing_data=patch_tracing_data,
                                                                summary=summary,
                                                                ignore_errors=ignore_errors
                                                               )
    print_if_verbose(f"///// Solved linear system for patch {patch_id} Success /////")

    # --- 4. Aggregate tracing data ---
    aggregated_tracing_data = aggregate_patch_tracing_data(patch_tracing_data)
    return linear_solved_V, consolidated_conn, aggregated_tracing_data