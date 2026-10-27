from cad_io.write import output_fused_mesh
from cad_io.read import load_intermediate_pkl
import utils.config as config
from operations.consolidate import consolidate
import argparse
import numpy as np
import igl
from utils.mesh_utils import compute_VF, adjacency_list, create_edge
from utils.regular_space import regular_space_internal_on_entire_loop
from utils.print import print_if_verbose

def regular_space_on_boundary_edges(V, F, prohibited_vids=None, frozen_edge_set=None):
    ''' If there is edge between any non-adjacent boundary vertices pair, we split this edge.
     This is to avoid non-manifold edges in fused mesh. '''
    vertex_to_faces = compute_VF(F)
    all_boundary_loops = igl.all_boundary_loop(np.array(F))
    adj_list = adjacency_list(F)
    for loop in all_boundary_loops:
        regular_space_internal_on_entire_loop(prohibited_vids,
                                                V, F, 
                                                loop,
                                                vertex_to_faces,
                                                adj_list,
                                                frozen_edge_set=frozen_edge_set)
    return


def fuse_meshing(patch_to_V, patch_to_connectivity, patch_to_aggregated_tracing_data, result_log=None):
    '''Fuse the meshing results into a single mesh. 
    Input: patch_to_V, patch_to_connectivity, patch_to_aggregated_tracing_data, result_log (optional, receives vertex-mismatch warnings)
    Output: V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids
    '''
    
    V_final, F_final = np.empty((0, 3)), np.empty((0, 3), dtype=int)
    patch_to_fids = {}
    feature_edge_id_to_global_vids = {}
    corner_id_to_vids = {}

    for patch_id in patch_to_V.keys():
        print_if_verbose(f"Fusing patch {patch_id}")
        V = patch_to_V[patch_id]
        F = patch_to_connectivity[patch_id].F
        # add all the traced vids of the patch to prohibited vids
        prohibited_vids = set()
        frozen_edge_set = set()
        tracing_data = patch_to_aggregated_tracing_data[patch_id]
        for feature_edge_id, traced_vids in tracing_data.feature_id_to_traced_vids.items():
            for i in range(len(traced_vids)-1):
                v_start = traced_vids[i]
                v_end = traced_vids[i+1]
                frozen_edge_set.add(create_edge(v_start, v_end))
            for vid in traced_vids:
                prohibited_vids.add(vid)
        regular_space_on_boundary_edges(V, F, prohibited_vids, frozen_edge_set)

        feature_edge_bnd_index = patch_to_aggregated_tracing_data[patch_id].feature_id_to_traced_vids
        corners_dict = patch_to_aggregated_tracing_data[patch_id].corner_id_to_vid
        vid_to_corner_dict = {}
        for corner_id, vid in corners_dict.items():
            if vid == None:
                print_if_verbose(f"Corner {corner_id} in patch {patch_id} has no mapped vid.")
                raise ValueError(f"Fusing: Corner {corner_id} in patch {patch_id} has no mapped vid.")
            vid_to_corner_dict[vid] = corner_id
        patch_to_fids[patch_id] = list(range(len(F_final), len(F_final) + len(F)))
        # remove duplicate vertices 
        local_vid_to_global_vid = {}

        for feature_edge_id, vids in feature_edge_bnd_index.items():
            if len(vids) < 1:
                print_if_verbose(f"Feature edge {feature_edge_id} in patch {patch_id} has less than 1 vertices.")
                raise ValueError(f"Fusing: Feature edge {feature_edge_id} in patch {patch_id} has less than 1 vertices.")
            if feature_edge_id not in feature_edge_id_to_global_vids.keys():
                feature_edge_id_to_global_vids[feature_edge_id] = [vid + len(V_final) for vid in vids ]
                
                for i, vid in enumerate(vids):
                    if vid in vid_to_corner_dict:
                        corner_id = vid_to_corner_dict[vid]
                        if corner_id not in corner_id_to_vids:
                            corner_id_to_vids[corner_id] = vid + len(V_final)
                        else:
                            feature_edge_id_to_global_vids[feature_edge_id][i] = corner_id_to_vids[corner_id]
                    else:
                        if i == 0 or i == len(vids)-1:
                            print_if_verbose(f"Feature edge {feature_edge_id} in patch {patch_id} has endpoint vertex {vid} that is not a corner.")
                            raise ValueError(f"Fusing: Feature edge {feature_edge_id} in patch {patch_id} has endpoint vertex {vid} that is not a corner.")
            
            for i, vid in enumerate(vids):
                if vid in vid_to_corner_dict.keys():
                    # this is a corner vid. Find the global vid the corner_id is mapped to
                    corner_id = vid_to_corner_dict[vid]
                    mapped_to_gobal_vid = corner_id_to_vids[corner_id]
                    local_vid_to_global_vid[vid] = mapped_to_gobal_vid
                else:
                    local_vid_to_global_vid[vid] = feature_edge_id_to_global_vids[feature_edge_id][i]
        # map the F
        for f, face in enumerate(F):
            for i in range(3):
                if face[i] in local_vid_to_global_vid.keys():
                    face[i] = local_vid_to_global_vid[face[i]]
                else:
                    face[i] += len(V_final)
            F[f] = face
        V_final = np.vstack([V_final, V])
        F_final = np.vstack([F_final, F])
        for local_vid, global_vid in local_vid_to_global_vid.items():
            if np.linalg.norm( (V[local_vid] - V_final[global_vid]) )> 1e-10:
                if result_log:
                    result_log.add_warning("fusing", f"Vertex {local_vid} in patch {patch_id} does not match.", patch_id=patch_id)
                print_if_verbose(f"    Error: local_vid {local_vid} V[local_vid] {V[local_vid]} global vid is  {global_vid} V_final[global_vid] {V_final[global_vid]}")
    # consolidate
    V_consolidate, F_consolidate, old2new, new2old = consolidate(V_final, F_final)

    for feature_edge_id, vids in feature_edge_id_to_global_vids.items():
        feature_edge_id_to_global_vids[feature_edge_id] = [old2new[vid] for vid in vids]
    for corner_id, vid in corner_id_to_vids.items():
        corner_id_to_vids[corner_id] = old2new[vid]

    return V_consolidate, F_consolidate, patch_to_fids, feature_edge_id_to_global_vids, corner_id_to_vids
def main():
    parser = argparse.ArgumentParser(description="Load and process a pkl file.")
    parser.add_argument("input_file", type=str, help="Path to the pkl file")
    parser.add_argument("output_folder", type=str, help="Folder to write fused.pkl into")
    parser.add_argument("--verbose", type=int, default=0, help="Print verbose logs when non-zero")

    args = parser.parse_args()
    config.verbose = args.verbose
    data = load_intermediate_pkl(args.input_file)
    patch_to_V = data["V"]
    patch_to_connectivity = data["connectivity"]
    patch_aggregated_tracing_data = data["tracing"]
    # Command to execute
    V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids = fuse_meshing(
        patch_to_V,
        patch_to_connectivity,
        patch_aggregated_tracing_data,
    )
    output_fused_mesh(args.output_folder, V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids)
if __name__ == "__main__":
    main()