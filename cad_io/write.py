import os
import pickle
import uuid
def output_fused_mesh(output_folder, V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids):
    file_path_pkl = os.path.join(output_folder, "fused.pkl")

    # atomic save pkl
    atomic_save((V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids), file_path_pkl)

    print(f"Data saved at: {file_path_pkl}")


def atomic_save(obj, path):
    dir_name = os.path.dirname(path) or os.curdir
    os.makedirs(dir_name, exist_ok=True)
    # Write a temp file next to the target, then rename it over the target. The temp file comes from open(),
    # not tempfile, so it gets the usual permissions (tempfile creates files only their owner can read).
    tmp_path = os.path.join(dir_name, f".{os.path.basename(path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp_path, "xb") as tmp:
            pickle.dump(obj, tmp)
        os.replace(tmp_path, path)
    except BaseException:
        # don't leave the partial temp file behind
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
