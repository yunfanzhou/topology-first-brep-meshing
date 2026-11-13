from utils.data_structures import PatchTracingData
from utils.print import print_if_verbose
def consolidate(V, F):
    # convert F to a set of unique vids
    old2new = {}
    new2old = {}
    newF = []
    newV = []
    for face in F:
        new_face = []
        for vid in face:
            if vid not in old2new:
                old2new[vid] = len(old2new)
                new2old[len(new2old)] = vid
                newV.append(V[vid])
            new_face.append(old2new[vid])
        newF.append(new_face)
    return newV, newF, old2new, new2old

def consolidate_one_patch(V, F,
                          patch_tracing_data: PatchTracingData):
    # consolidate
    consolidate_V, consolidate_F, old2new, new2old = consolidate(V, F)
    # update initial loop  
    consolidate_inserted_and_traced = update_inserted_and_traced_vids_after_consolidate(patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data, old2new)
    consolidate_corners = update_corners(patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid, old2new)
    patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data = consolidate_inserted_and_traced
    patch_tracing_data.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid = consolidate_corners

    # loop through all subsequent traced loops and update them as well
    for loop_id, loop_tracing_data in patch_tracing_data.remaining_loops_tracing_data.items():
        consolidate_inserted_and_traced = update_inserted_and_traced_vids_after_consolidate(loop_tracing_data.feature_id_to_edge_tracing_data, old2new)
        consolidate_corners = update_corners(loop_tracing_data.corner_id_to_vid, old2new)
        loop_tracing_data.feature_id_to_edge_tracing_data = consolidate_inserted_and_traced
        loop_tracing_data.corner_id_to_vid = consolidate_corners
    return consolidate_V, consolidate_F
def update_inserted_and_traced_vids_after_consolidate(inserted_and_traced_current_patch, old2new):
    consolidate_inserted_and_traced = {}
    for feature_edge_id in inserted_and_traced_current_patch.keys():
        consolidate_inserted_and_traced[feature_edge_id] = [old2new[vid] for vid in inserted_and_traced_current_patch[feature_edge_id] if vid in old2new]
        if len(consolidate_inserted_and_traced[feature_edge_id]) < len(inserted_and_traced_current_patch[feature_edge_id]):
            print_if_verbose(f"consolidated  {len(consolidate_inserted_and_traced[feature_edge_id])}")
            print_if_verbose(f"original {len(inserted_and_traced_current_patch[feature_edge_id])}")
            print_if_verbose(f"missing vids for feature edge id {feature_edge_id}: {[vid for vid in inserted_and_traced_current_patch[feature_edge_id] if vid not in old2new]}")
            print_if_verbose(f"Feature edge id {feature_edge_id} partially not found in the consolidated mesh") 
            raise Exception(f"Feature edge id {feature_edge_id} partially not found in the consolidated mesh")
    return consolidate_inserted_and_traced

def update_corners(corner_to_vid_current_patch, old2new):
    updated_corners = {}
    for corner, vid in corner_to_vid_current_patch.items():
        print_if_verbose(f"corner {corner} vid {vid} ")
        if vid not in old2new:
            raise ValueError(f"Corner {corner} maps to vertex {vid}, which is not in the consolidated mesh")
        updated_corners[corner] = old2new[vid]
    return updated_corners