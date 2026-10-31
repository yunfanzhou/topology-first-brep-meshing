import importlib.util
import os
import shutil
from os.path import exists
from run_one_patch import local_step_one_patch
from initial_meshing import (
    CollapsedInteriorIsoLineError,
    create_feature_edge_geometry_data,
    run_initial_meshing_for_face,
    build_singularity_data_from_face,
    build_corner_data_from_abs_part,
    build_feature_edge_to_patches_from_abs_part,
    build_patch_input_data_from_abs_face,
)
from cad_io.write import atomic_save, output_fused_mesh
from cad_io.read import atomic_load
from operations.align import align_adjacent_patches
import numpy as np
import argparse
from utils.print import print_if_verbose, ResultLog, classify_fail_reason
import utils.config as config
from fuse import fuse_meshing
from pathlib import Path
from datetime import datetime

ATTEMPT_MAX = 3
# Fewer faces than this to mesh: serial (worker start-up costs ~1 s).
PARALLEL_MIN_FACES = 16


def default_mmg2d_path():
    """mmg2d_O3 of the mmgpy package (requirements.txt) if installed, else mmg2d_O3 on PATH.

    The mmg2d_O3 command mmgpy installs is a Python launcher, which would start an interpreter for
    every mmg call, so the executable inside the package is called directly.
    """
    spec = importlib.util.find_spec("mmgpy")
    if spec is not None and spec.submodule_search_locations:
        exe = os.path.join(spec.submodule_search_locations[0], "bin",
                           "mmg2d_O3.exe" if os.name == "nt" else "mmg2d_O3")
        if exists(exe):
            return exe
    return "mmg2d_O3"


def default_workers():
    """CPUs this process may run on (its CPU affinity) where the OS reports it, else the CPU count."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def form_per_patch_summary():
    return {
        "success": False,
        "rerun_attempts": 0,
        "initial_loop_fallback": False,
        "non_initial_loop_fallback_cnt": -1,
        "value_error_found": {
            "non_manifold_error": False,
            "other_value_error": False,
            "other_value_error_details": []
        },
        "max_error": None,
        "avg_error": None,
        "number_of_boundary_vertices": 0,
    }


def mesh_one_face(face, edge_geometry_data, all_corners, target_error, mmg2d_path, diag_bbox):
    """Initial meshing + local step of one face with up to ATTEMPT_MAX attempts.

    Returns a dict: ok, patch_id, duration, attempts, and V/connectivity/tracing/summary on
    success or error on failure. target_error is the model's; only the first attempt remeshes
    with it. A retry (attempt k > 0) runs no mmg: it refines the UV mesh saved by the first
    attempt (or the initial grid if there is none) with k levels of igl.upsample, then runs the
    local step again on it. An attempt that raises an Exception is retried, and the face fails
    after the last one. MemoryError is re-raised, and CollapsedInteriorIsoLineError (a
    BaseException) passes through: both end the run.
    """
    patch_id = face.id
    config.current_patch_id = patch_id  # labels oversized-patch dumps
    patch_start_time = datetime.now()
    patch_summary = form_per_patch_summary()
    prev_V_uv = prev_F = prev_sub_domains = None
    for attempt in range(ATTEMPT_MAX):
        try:
            V_3d, F, u_periodic_edges, v_periodic_edges, sing_info, surface_merged_edges, fallback_V_uv, fallback_F, fallback_sub_domains = run_initial_meshing_for_face(
                face,
                edge_geometry_data,
                target_error=target_error,
                mmg2d_path=mmg2d_path,
                prev_V_uv=prev_V_uv,
                prev_F=prev_F,
                prev_sub_domains=prev_sub_domains,
                attempt=attempt,
                model_diag=diag_bbox,
            )
            if fallback_V_uv is not None:
                prev_V_uv = fallback_V_uv
                prev_F = fallback_F
                prev_sub_domains = fallback_sub_domains
            patch_data = build_patch_input_data_from_abs_face(
                face, edge_geometry_data,
                V_3d, F, u_periodic_edges, v_periodic_edges,
                surface_merged_edges=surface_merged_edges,
            )
            patch_singularity = build_singularity_data_from_face(face, sing_info)
            linear_solved_V, connectivity, aggregated_tracing_data = local_step_one_patch(
                patch_id,
                patch_data,
                patch_singularity,
                all_corners,
                summary=patch_summary,
                attempt=attempt,
                ignore_trace_geometry=True,
            )
            patch_summary["success"] = True
            return dict(ok=True, patch_id=patch_id, V=linear_solved_V, connectivity=connectivity,
                        tracing=aggregated_tracing_data, summary=patch_summary, attempts=attempt + 1,
                        duration=(datetime.now() - patch_start_time).total_seconds())
        except MemoryError:
            raise  # a retry runs on an upsampled mesh: it would need more memory
        except Exception as e:
            error = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            print_if_verbose(
                f"!!!!! Local steps failed for patch {patch_id} on attempt {attempt} due to: {error} !!!!!"
            )
            if attempt == ATTEMPT_MAX - 1:
                return dict(ok=False, patch_id=patch_id, error=error, attempts=attempt + 1,
                            duration=(datetime.now() - patch_start_time).total_seconds())


_WORKER = {}


def _worker_init(abs_file, part_index, edge_geometry_path, verbose, oversized_dump_dir):
    # Only paths cross the pipe: a large payload here makes the parent block on each worker's start-up.
    from abs import read_parts
    part = read_parts(abs_file)[part_index]
    _WORKER["faces"] = {face.id: face for face in part.Solid.faces}
    _WORKER["edge_geometry_data"] = atomic_load(edge_geometry_path)["edge_geometry_data"]
    _WORKER["all_corners"] = build_corner_data_from_abs_part(part)
    config.verbose = verbose
    config.oversized_dump_dir = oversized_dump_dir


def _worker_mesh_face(face_id, target_error, mmg2d_path, diag_bbox):
    return mesh_one_face(_WORKER["faces"][face_id], _WORKER["edge_geometry_data"], _WORKER["all_corners"],
                         target_error, mmg2d_path, diag_bbox)


def main():
    parser = argparse.ArgumentParser(
        description="Python-only CAD meshing pipeline (initial meshing → local step → global alignment → fusion)."
    )
    parser.add_argument("abs_file", type=str,
                        help="Path to the ABS-readable CAD input file")
    parser.add_argument("output_folder", type=str,
                        help="Directory where results and checkpoints are written")
    parser.add_argument("--error_rel", type=float, default=1e-3,
                        help="Relative target error with respect to the model bounding box diagonal (default: 1e-3)")
    parser.add_argument("--mmg2d_path", type=str, default=None,
                        help="Path to the mmg2d executable (default: the one of the mmgpy package, else mmg2d_O3 on PATH)")
    parser.add_argument("--part", type=int, default=0,
                        help="Part index within the ABS file (default: 0)")
    parser.add_argument("--patch", type=int, default=None,
                        help="If specified, only run the local step of the face with this id (no alignment or fusion)")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse the checkpoints of a previous run in output_folder")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="Processes for the per-face local step (default: the number of CPUs this "
                             "process may run on, i.e. its CPU affinity, or the CPU count where the OS does "
                             "not report it; 1 = serial). The output does not depend on it.")
    args = parser.parse_args()
    if args.mmg2d_path is None:
        args.mmg2d_path = default_mmg2d_path()
    if shutil.which(args.mmg2d_path) is None:
        parser.error(f"mmg2d executable not found: {args.mmg2d_path} (pip install -r requirements.txt, "
                     "or pass --mmg2d_path)")

    os.makedirs(args.output_folder, exist_ok=True)
    ckpt_folder = Path(args.output_folder) / "checkpoints"
    ckpt_folder.mkdir(exist_ok=True)
    result_json = os.path.join(args.output_folder, "result.json")

    if args.verbose:
        config.verbose = args.verbose
    config.oversized_dump_dir = os.path.join(args.output_folder, "oversized_patch_dumps")

    if exists(result_json) and not args.resume:
        os.remove(result_json)

    result_log = ResultLog(result_json, resume=args.resume)
    stage = "init"

    try:
        # ------------------------------------------------------------------
        # Load ABS data
        # ------------------------------------------------------------------
        stage = "loading"
        from abs import read_parts
        parts = read_parts(args.abs_file)
        part = parts[args.part]
        bbox = part._geometry_data.bbox

        diag_bbox = np.linalg.norm(np.array(bbox[0][1]) - np.array(bbox[0][0]))
        print_if_verbose(f"Part {args.part} bounding box diagonal length: {diag_bbox:.3e}")
        target_error = args.error_rel * diag_bbox
        print_if_verbose(f"Target error: {target_error}")

        # All corner positions (needed by local_step_one_patch → solve)
        all_corners = build_corner_data_from_abs_part(part)
        # Edge geometry: edge.id -> (nv,3) polyline in true orientation
        edge_geometry_ckpt = ckpt_folder / "edge_geometry.pkl"
        if args.resume and edge_geometry_ckpt.exists():
            print_if_verbose("Loading edge geometry from checkpoint")
            edge_geom_data = atomic_load(edge_geometry_ckpt)
            edge_geometry_data = edge_geom_data["edge_geometry_data"]
            edge_t_values = edge_geom_data["edge_t_values"]
        else:
            edge_geometry_data, edge_t_values = create_feature_edge_geometry_data(part,
                                                                                  all_corners,
                                                                                  target_error=target_error)
            atomic_save({"edge_geometry_data": edge_geometry_data,
                         "edge_t_values": edge_t_values}, edge_geometry_ckpt)

        # Feature edge → adjacent face ids (needed by global alignment)
        feature_edge_to_patches = build_feature_edge_to_patches_from_abs_part(part)

        result_log.set_total_patches(1 if args.patch is not None else len(part.Solid.faces))

        # ------------------------------------------------------------------
        # Local step: initial meshing + tracing + linear solve
        # ------------------------------------------------------------------
        stage = "local"
        local_ckpt_folder = ckpt_folder / "local"
        local_ckpt_folder.mkdir(exist_ok=True)

        patch_to_tracing_data = {}
        patch_to_V = {}
        patch_to_connectivity = {}

        global_align_ckpt = ckpt_folder / "global_alignment.pkl"
        fuse_dir = Path(args.output_folder) / "fused.pkl"

        if len(part.Solid.faces) == 0:
            raise Exception(" EMPTY_MODEL")
        if (global_align_ckpt.exists() or fuse_dir.exists()) and args.resume:
            print_if_verbose("Global alignment or fused mesh checkpoint found. Skipping local steps.")
        else:
            print_if_verbose("///// Start local steps for all patches /////")
            todo = []
            for face in part.Solid.faces:
                patch_id = face.id
                if args.patch is not None and patch_id != args.patch:
                    continue
                ckpt_path = local_ckpt_folder / f"patch_{patch_id}.pkl"
                if args.resume and ckpt_path.exists():
                    print_if_verbose(f"Loading checkpoint for patch {patch_id}")
                    data = atomic_load(ckpt_path)
                    patch_to_V[patch_id] = data["V"]
                    patch_to_connectivity[patch_id] = data["connectivity"]
                    patch_to_tracing_data[patch_id] = data["tracing"]
                    continue
                todo.append(face)
            if args.patch is not None and not todo and args.patch not in patch_to_V:
                raise ValueError(f"No face with id {args.patch} in part {args.part}")

            def save_checkpoint(r):
                atomic_save({"V": r["V"],
                             "connectivity": r["connectivity"],
                             "tracing": r["tracing"]}, local_ckpt_folder / f"patch_{r['patch_id']}.pkl")

            def finish_patch(r):
                """Record one face's result; raise on failure."""
                patch_id = r["patch_id"]
                if not r["ok"]:
                    result_log.patch_failed(
                        patch_id, r["duration"],
                        reason=classify_fail_reason(r["error"]),
                        detail=r["error"],
                        attempts=r["attempts"])
                    raise ValueError(f"Failed to process patch {patch_id} after {ATTEMPT_MAX} attempts: {r['error']}")
                result_log.patch_done(patch_id, r["duration"], attempts=r["attempts"])
                patch_to_tracing_data[patch_id] = r["tracing"]
                patch_to_V[patch_id] = r["V"]
                patch_to_connectivity[patch_id] = r["connectivity"]
                print_if_verbose(f"///// Finished local steps for patch {patch_id} /////")

            if args.workers <= 1 or len(todo) < PARALLEL_MIN_FACES:
                for face in todo:
                    r = mesh_one_face(face, edge_geometry_data, all_corners, target_error,
                                      args.mmg2d_path, diag_bbox)
                    if r["ok"]:
                        save_checkpoint(r)
                    finish_patch(r)
            else:
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor, as_completed

                def face_size(face):  # polyline points on the face's edges: predicts its meshing time
                    return sum(len(edge_geometry_data.get(he.edge.id, ())) for loop in face.loops for he in loop.halfedges)

                done, failed = {}, None  # results of the finished faces by id; the first failed face
                with ProcessPoolExecutor(max_workers=min(args.workers, len(todo)),
                                         mp_context=mp.get_context("spawn"),
                                         initializer=_worker_init,
                                         initargs=(args.abs_file, args.part, str(edge_geometry_ckpt),
                                                   config.verbose, config.oversized_dump_dir)) as pool:
                    futures = [pool.submit(_worker_mesh_face, face.id, target_error, args.mmg2d_path, diag_bbox)
                               for face in sorted(todo, key=face_size, reverse=True)]  # largest first
                    try:
                        for future in as_completed(futures):
                            r = future.result()
                            if not r["ok"]:
                                failed = r
                                break
                            save_checkpoint(r)  # right away, so that --resume keeps it if the run is killed
                            done[r["patch_id"]] = r
                    finally:
                        if len(done) < len(todo):  # a face failed: drop the queued faces
                            pool.shutdown(wait=False, cancel_futures=True)
                        for face in todo:  # record in face order, as in the serial loop
                            if face.id in done:
                                finish_patch(done[face.id])
                    if failed is not None:
                        finish_patch(failed)  # records the failure and raises

        if args.patch is not None:  # alignment and fusion need every face
            print(f"Local step of face {args.patch} done; checkpoint in {local_ckpt_folder}")
            result_log.finish()
            return

        # ------------------------------------------------------------------
        # Global alignment and fusion
        # ------------------------------------------------------------------
        print_if_verbose("///// Start global alignment and fusion /////")

        if fuse_dir.exists() and args.resume:
            print_if_verbose("Fused mesh checkpoint found. Skipping global alignment steps.")
        else:
            if args.resume and global_align_ckpt.exists():
                print_if_verbose("Loading global alignment checkpoint")
                aligned = atomic_load(global_align_ckpt)
                patch_to_V = aligned["V"]
                patch_to_connectivity = aligned["connectivity"]
                patch_to_tracing_data = aligned["tracing"]
            else:
                stage = "alignment"
                align_adjacent_patches(patch_to_V,
                                       patch_to_connectivity,
                                       patch_to_tracing_data,
                                       feature_edge_to_patches)
                atomic_save({"V": patch_to_V,
                             "connectivity": patch_to_connectivity,
                             "tracing": patch_to_tracing_data}, global_align_ckpt)

        stage = "fusing"
        if args.resume and fuse_dir.exists():
            print_if_verbose("Loading fused mesh from checkpoint")
            fused_data = atomic_load(fuse_dir)
            V_consolidate, F_consolidate, patch_to_fids, feature_edge_id_to_global_vids, corner_id_to_vids = fused_data
        else:
            print_if_verbose("Fusing meshes from aligned patches")
            (V_consolidate,
             F_consolidate,
             patch_to_fids,
             feature_edge_id_to_global_vids,
             corner_id_to_vids) = fuse_meshing(patch_to_V,
                                               patch_to_connectivity,
                                               patch_to_tracing_data,
                                               result_log=result_log)
            output_fused_mesh(args.output_folder,
                              V_consolidate,
                              F_consolidate,
                              patch_to_fids,
                              feature_edge_id_to_global_vids,
                              corner_id_to_vids)

        V_np = np.array(V_consolidate)
        bbox_diag = float(np.linalg.norm(V_np.max(axis=0) - V_np.min(axis=0)))
        print_if_verbose(f"Fused mesh bounding box diagonal: {bbox_diag}")

        result_log.finish()

    # CollapsedInteriorIsoLineError is a BaseException so that no retry swallows it; record it too
    except (Exception, CollapsedInteriorIsoLineError) as e:
        result_log.mark_failed(stage, e)
        raise


if __name__ == "__main__":
    main()
