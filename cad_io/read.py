from utils.print import print_if_verbose
import pickle


def load_fused_output(fused_file_path):
    """Loads the fused output data from the pickle file written by output_fused_mesh."""
    print_if_verbose(f"Loading fused output data from {fused_file_path}")
    with open(fused_file_path, 'rb') as file:
        data = pickle.load(file)


        V = data[0]
        F = data[1]

        # key: is the face id, value: list of triangle ids
        patch2fid = data[2]

        # key: edge id, value: list of vertex ids forming a polygon
        edge2vidschain = data[3]

        corner2vids = data[4]
        new_corner_id_mapping = data[5] if len(data) > 5 else None
        print_if_verbose(f"Loaded new_corner_id_mapping: {new_corner_id_mapping}")
        return V, F, patch2fid, edge2vidschain, corner2vids, new_corner_id_mapping

def load_intermediate_pkl(file_path):
    """Loads intermediate data from a pickle file."""
    print_if_verbose(f"Loading intermediate data from {file_path}")
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def atomic_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
