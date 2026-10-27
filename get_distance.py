"""
Per-mesh hausdorff distance from a CAD-sampled point cloud to the remeshed
triangle mesh. Independent from get_metrics.py so it can be run on every model
without being tied to mesh-quality sample sizes.

You hand it the directory layout directly:
    -i <results>     each <results>/<mesh_id>/remeshed.vtu
    --cad-root <cad> each <cad>/<mesh_id>.hdf5 (as in data/simple), or
                     <cad>/<mesh_id>/assembly.hdf5 (or <mesh_id>_*.hdf5)

Only the first CAD part (part_id = 0) is meshed in this pipeline, so the point
cloud is always sampled from part 0.

Writes a CSV: mesh_id, hausdorff. Use plot_metrics.py for plotting.
hausdorff = max distance from CAD point-cloud samples (random in each face's
parameter domain, inside its trimming loops) to (V, F) of remeshed.vtu,
normalized by the CAD bbox diagonal. Distance is computed with
igl.point_mesh_squared_distance.

Example (batch):
    python get_distance.py -i out --cad-root data/simple -o distance.csv -j 8

Single-folder mode (auto-detected when -i contains remeshed.vtu directly).
-i points at the per-mesh folder, --cad-root at the model file (or its folder):
    python get_distance.py -i out/Box --cad-root data/simple/Box.hdf5 --plot
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import igl
import meshio
import numpy as np
import pandas as pd

from abs import read_parts, sampler


def load_triangles(mesh_path):
    mesh = meshio.read(mesh_path)
    if "triangle" not in mesh.cells_dict:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return mesh.points, mesh.cells_dict["triangle"]


def _resolve_assembly(cad_root, mesh_id):
    if os.path.isfile(cad_root):
        return cad_root
    flat = os.path.join(cad_root, f"{mesh_id}.hdf5")
    if os.path.exists(flat):
        return flat
    direct = os.path.join(cad_root, mesh_id, "assembly.hdf5")
    if os.path.exists(direct):
        return direct
    from glob import glob

    matches = sorted(glob(os.path.join(cad_root, mesh_id, f"{mesh_id}_*.hdf5")))
    return matches[0] if matches else None


def _sample_cad_points(assembly_path, num_samples, seed, points_ratio=5):
    """Sample a point cloud from the faces of the first CAD part (part_id=0).

    Each face gets points_ratio * num_samples * (its share of the area) points,
    uniformly random in its parameter domain, and keeps those inside its
    trimming loops. This is what abs-hdf5 1.0's sample_parts does with
    use_poisson=False, force_num_points=False, apply_transform=False; the
    pinned abs-hdf5 0.2.2 has no such options.
    """
    parts = read_parts(assembly_path)
    faces = parts[0].Solid.faces
    total_area = sum(face.get_area() for face in faces)
    np.random.seed(seed)  # sampler draws from np.random
    points = [np.zeros((0, 3))]
    for face in faces:
        n = int(np.ceil(face.get_area() / total_area * points_ratio * num_samples))
        uv, pt = sampler.random_sample(face, n, min_pts=2)
        points.append(pt[face.filter_outside_points(uv)])
    bbox = np.asarray(parts[0]._geometry_data.bbox)
    return np.concatenate(points).astype(np.float64), bbox


def _read_done_ids(out_path):
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return set()
    try:
        df = pd.read_csv(out_path)
    except Exception:
        return set()
    if "mesh_id" not in df.columns:
        return set()
    return set(df["mesh_id"].astype(str).tolist())


def _append_row(out_path, mesh_id, value):
    with open(out_path, "a", buffering=1) as f:
        f.write(f"{mesh_id},{value}\n")


def hausdorff_one(results_path, mesh_id, cad_root, num_samples, seed=0,
                  plot=False, plot_save_dir=None):
    folder = os.path.join(results_path, mesh_id)
    V, F = load_triangles(os.path.join(folder, "remeshed.vtu"))
    if F.shape[0] == 0:
        return float("nan")

    assembly = _resolve_assembly(cad_root, mesh_id)
    if assembly is None:
        raise FileNotFoundError(f"no CAD model for {mesh_id} in {cad_root}")

    P, bbox = _sample_cad_points(assembly, num_samples, seed)
    if P.shape[0] == 0:
        return float("nan")

    sqr_d, _, _ = igl.point_mesh_squared_distance(P, V, F)
    d = np.sqrt(np.maximum(sqr_d, 0.0))
    max_d = float(d.max())
    diag = float(np.linalg.norm(bbox[0, 1] - bbox[0, 0]))
    if diag == 0:
        diag = 1.0

    if plot or plot_save_dir is not None:
        _plot_pointcloud_and_mesh(
            mesh_id, P, d / diag, V, F,
            interactive=plot, save_dir=plot_save_dir,
        )

    return max_d / diag


def _plot_pointcloud_and_mesh(mesh_id, points, dist_normalized, V, F,
                              interactive=True, save_dir=None):
    """Overlay CAD-sampled point cloud on the remeshed surface using PyVista."""
    import pyvista as pv

    faces = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
    mesh = pv.PolyData(V, faces)
    cloud = pv.PolyData(points)
    cloud["dist"] = dist_normalized

    plotter = pv.Plotter(off_screen=(save_dir is not None and not interactive))
    plotter.add_mesh(mesh, color="lightgray", opacity=0.55, show_edges=True,
                     edge_color="black", line_width=0.4, label="remeshed")
    plotter.add_mesh(cloud, scalars="dist", point_size=6.0,
                     render_points_as_spheres=True, cmap="viridis",
                     scalar_bar_args={"title": "dist / diag"}, label="CAD samples")
    plotter.add_text(mesh_id, font_size=10)
    plotter.add_legend()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_png = os.path.join(save_dir, f"{mesh_id}.png")
        plotter.show(screenshot=out_png, auto_close=not interactive)
        print(f"  saved plot: {out_png}")
    elif interactive:
        plotter.show()
    plotter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", type=str, default=".", help="Results directory.")
    parser.add_argument("--cad-root", type=str, required=True,
                        help="CAD root with <mesh_id>.hdf5 or <mesh_id>/assembly.hdf5 (single-folder mode: the model file).")
    parser.add_argument("-o", type=str, default="distance.csv", help="Output CSV.")
    parser.add_argument("--num-samples", type=int, default=8000, help="CAD point-cloud samples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed of the CAD point-cloud sampling.")
    parser.add_argument("-j", type=int, default=1, help="Parallel jobs.")
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="Per-mesh retries when a worker pool dies. Mesh_ids that exceed this are skipped.")
    parser.add_argument("--plot", action="store_true",
                        help="Open an interactive PyVista window per mesh overlaying the CAD point cloud and remeshed surface. Forces -j 1.")
    parser.add_argument("--plot-save-dir", type=str, default=None,
                        help="Directory to save per-mesh PyVista screenshots (off-screen unless --plot is also set).")
    args = parser.parse_args()

    if args.plot and args.j > 1:
        print("--plot requires sequential execution; setting -j 1.")
        args.j = 1

    start = time.time()

    # Single-folder mode: -i points directly at a folder containing remeshed.vtu,
    # and --cad-root points directly at the corresponding CAD folder
    # (containing assembly.hdf5 or <id>_*.hdf5). Rewrite to the parent/<id>
    # layout the rest of the script assumes.
    single_mode = os.path.isfile(os.path.join(args.i, "remeshed.vtu"))
    if single_mode:
        i_abs = os.path.abspath(args.i.rstrip("/"))
        cad_abs = os.path.abspath(args.cad_root.rstrip("/"))
        mesh_id = os.path.basename(i_abs)
        if not os.path.isfile(cad_abs):  # a file is used as the model as is
            if os.path.basename(cad_abs) != mesh_id:
                print(f"Warning: -i basename '{mesh_id}' != --cad-root basename "
                      f"'{os.path.basename(cad_abs)}'. Using -i basename as mesh_id.")
            cad_abs = os.path.dirname(cad_abs)
        args.i = os.path.dirname(i_abs)
        args.cad_root = cad_abs
        mesh_ids = [mesh_id]
        print(f"Single-folder mode: mesh_id={mesh_id}")
    else:
        mesh_ids = sorted(
            d for d in os.listdir(args.i) if os.path.isdir(os.path.join(args.i, d))
        )
        print(f"Found {len(mesh_ids)} folders in {args.i}.")

    done_ids = _read_done_ids(args.o)
    if not os.path.exists(args.o) or os.path.getsize(args.o) == 0:
        with open(args.o, "w") as f:
            f.write("mesh_id,hausdorff\n")
    if done_ids:
        print(f"Resuming: {len(done_ids)} mesh_ids already in {args.o}, skipping those.")

    todo = [m for m in mesh_ids if m not in done_ids]
    print(f"To process: {len(todo)}/{len(mesh_ids)}.")

    attempts = {m: 0 for m in todo}
    remaining = list(todo)
    n_ok = 0
    n_skip_broken = 0

    while remaining:
        if args.j > 1:
            print(f"Using {args.j} parallel jobs ({len(remaining)} mesh_ids in this round).")
            processed_this_round = set()
            pool_broke = False
            try:
                with ProcessPoolExecutor(max_workers=args.j) as ex:
                    futs = {
                        ex.submit(
                            hausdorff_one, args.i, mid, args.cad_root, args.num_samples,
                            args.seed, False, args.plot_save_dir,
                        ): mid
                        for mid in remaining
                    }
                    for fut in as_completed(futs):
                        mid = futs[fut]
                        try:
                            val = fut.result()
                        except BrokenProcessPool:
                            pool_broke = True
                            break
                        except Exception as exc:
                            print(f"Failed on {mid}: {exc}")
                            processed_this_round.add(mid)
                            continue
                        _append_row(args.o, mid, val)
                        processed_this_round.add(mid)
                        n_ok += 1
                        if n_ok % 25 == 0:
                            print(f"  processed {n_ok}/{len(todo)}")
                            sys.stdout.flush()
            except BrokenProcessPool:
                pool_broke = True

            if not pool_broke:
                break

            new_remaining = []
            for m in remaining:
                if m in processed_this_round:
                    continue
                attempts[m] += 1
                if attempts[m] >= args.max_attempts:
                    print(f"Skipping {m}: hit max-attempts ({args.max_attempts}).")
                    n_skip_broken += 1
                    continue
                new_remaining.append(m)
            if not new_remaining:
                break
            print(f"Pool died; restarting with {len(new_remaining)} mesh_ids left.")
            sys.stdout.flush()
            remaining = new_remaining
        else:
            for mid in remaining:
                try:
                    val = hausdorff_one(
                        args.i, mid, args.cad_root, args.num_samples,
                        seed=args.seed, plot=args.plot,
                        plot_save_dir=args.plot_save_dir,
                    )
                except Exception as exc:
                    print(f"Failed on {mid}: {exc}")
                    continue
                _append_row(args.o, mid, val)
                n_ok += 1
                if n_ok % 25 == 0:
                    print(f"  processed {n_ok}/{len(todo)}")
                    sys.stdout.flush()
            remaining = []

    print(f"Successfully evaluated {n_ok} new meshes (skipped {n_skip_broken} after repeated pool kills).")
    print(f"Output CSV: {args.o}")
    print(f"Total runtime: {time.time() - start:.2f}s")
