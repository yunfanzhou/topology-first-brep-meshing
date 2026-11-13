import os
import pickle
import tempfile
def output_fused_mesh(output_folder, V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids):
    file_path_pkl = os.path.join(output_folder, "fused.pkl")

    # atomic save pkl
    atomic_save((V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids), file_path_pkl)

    print(f"Data saved at: {file_path_pkl}")


def atomic_save(obj, path):
    dir_name = os.path.dirname(path) or os.curdir
    os.makedirs(dir_name, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_name) as tmp:
        tmp_path = tmp.name
        try:
            pickle.dump(obj, tmp)
        except BaseException:
            # don't leave the partial temp file behind
            tmp.close()
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
    os.replace(tmp_path, path)
