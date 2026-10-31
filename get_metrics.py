"""
Per-triangle mesh quality metrics for remeshed outputs (or input meshes).

Layout expected:
    <results_path>/<mesh_id>/remeshed.vtu        (default)
    <results_path>/<mesh_id>/mesh.msh            (used with --before)

Writes a CSV with one row per triangle: mesh_id, min_angle, radius_ratio.
Also writes a sibling <stem>_counts.csv with one row per mesh:
    mesh_id, num_vertices, num_faces
Use plot_metrics.py to plot histograms from either CSV.

Single-folder mode is auto-detected when -i contains the target mesh
(remeshed.vtu, or mesh.msh with --before) directly.

Examples:
    python get_metrics.py -i /path/to/results -o metrics.csv --sample 500 -j 8
    python get_metrics.py -i /path/to/results --before -o metrics_before.csv
    python get_metrics.py -i /path/to/<mesh_id> -o /path/to/<mesh_id>/metrics.csv
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import igl
import meshio
import numpy as np
import pandas as pd


def load_triangles(mesh_path):
    mesh = meshio.read(mesh_path)
    if "triangle" not in mesh.cells_dict:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return mesh.points, mesh.cells_dict["triangle"]


def per_triangle_quality(V, F):
    """Per-triangle min_angle (deg, via igl.internal_angles) and radius_ratio.

    Degenerate triangles (zero-length edge or zero area) get min_angle=0 and
    radius_ratio=0 instead of being filtered out, so the caller sees the bad
    triangles in the output rather than having them silently dropped.
    """
    if F.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty

    V64 = np.asarray(V, dtype=np.float64)
    F64 = np.asarray(F, dtype=np.int64)

    ia = igl.internal_angles(V64, F64)
    min_angle = np.degrees(ia.min(axis=1))
    min_angle = np.where(np.isfinite(min_angle), min_angle, 0.0)

    v0, v1, v2 = V64[F64[:, 0]], V64[F64[:, 1]], V64[F64[:, 2]]
    a = np.linalg.norm(v1 - v2, axis=1)
    b = np.linalg.norm(v0 - v2, axis=1)
    c = np.linalg.norm(v0 - v1, axis=1)
    edge_ok = (a > 0) & (b > 0) & (c > 0)

    a_s = np.where(edge_ok, a, 1.0)
    b_s = np.where(edge_ok, b, 1.0)
    c_s = np.where(edge_ok, c, 1.0)
    s = (a_s + b_s + c_s) / 2.0
    area = np.sqrt(np.clip(s * (s - a_s) * (s - b_s) * (s - c_s), a_min=0, a_max=None))
    nonzero = edge_ok & (area > 0)
    inradius = np.where(nonzero, area / s, 0.0)
    circumradius = np.where(nonzero, (a_s * b_s * c_s) / (4.0 * area + (~nonzero)), 1.0)
    radius_ratio = np.where(nonzero, 2.0 * inradius / circumradius, 0.0)

    return min_angle.astype(np.float32), radius_ratio.astype(np.float32)


def process_one(results_path, mesh_id, before):
    folder = os.path.join(results_path, mesh_id)
    name = "mesh.msh" if before else "remeshed.vtu"
    V, F = load_triangles(os.path.join(folder, name))
    min_angle, radius_ratio = per_triangle_quality(V, F)
    return min_angle, radius_ratio, int(V.shape[0]), int(F.shape[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", type=str, default=".", help="Results directory.")
    parser.add_argument("-o", type=str, default="metrics.csv", help="Output CSV.")
    parser.add_argument("--sample", type=int, default=500, help="Random sample size (<=0 keeps all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--before", action="store_true", help="Read mesh.msh instead of remeshed.vtu.")
    parser.add_argument("-j", type=int, default=1, help="Parallel jobs.")
    args = parser.parse_args()

    print(f"Mode: {'before (mesh.msh)' if args.before else 'after (remeshed.vtu)'}")
    start = time.time()

    mesh_name = "mesh.msh" if args.before else "remeshed.vtu"
    # Single-folder mode: -i points directly at a folder containing the target
    # mesh file. Rewrite to the parent/<id> layout the rest of the loop uses.
    if os.path.isfile(os.path.join(args.i, mesh_name)):
        i_abs = os.path.abspath(args.i.rstrip("/"))
        mesh_id = os.path.basename(i_abs)
        args.i = os.path.dirname(i_abs)
        mesh_ids = [mesh_id]
        print(f"Single-folder mode: mesh_id={mesh_id}")
    else:
        all_dirs = sorted(
            d for d in os.listdir(args.i) if os.path.isdir(os.path.join(args.i, d))
        )
        print(f"Found {len(all_dirs)} folders in {args.i}.")
        mesh_ids = [
            d for d in all_dirs if os.path.isfile(os.path.join(args.i, d, mesh_name))
        ]
        print(f"{len(mesh_ids)} contain {mesh_name}.")
    if args.sample > 0 and len(mesh_ids) > args.sample:
        random.seed(args.seed)
        mesh_ids = random.sample(mesh_ids, args.sample)
        print(f"Sampled {len(mesh_ids)} meshes (seed={args.seed}).")

    rows_id, rows_ang, rows_rr = [], [], []
    counts_rows = []  # (mesh_id, num_vertices, num_faces) per mesh
    n_ok = 0

    if args.j > 1:
        print(f"Using {args.j} parallel jobs.")
        with ProcessPoolExecutor(max_workers=args.j) as ex:
            futs = {
                ex.submit(process_one, args.i, mid, args.before): mid
                for mid in mesh_ids
            }
            for fut in as_completed(futs):
                mid = futs[fut]
                try:
                    ang, rr, nV, nF = fut.result()
                except Exception as exc:
                    print(f"Failed on {mid}: {exc}")
                    continue
                counts_rows.append((mid, nV, nF))
                if ang.size == 0:
                    continue
                rows_id.append(np.full(ang.shape, mid, dtype=object))
                rows_ang.append(ang)
                rows_rr.append(rr)
                n_ok += 1
                if n_ok % 25 == 0:
                    print(f"  processed {n_ok}/{len(mesh_ids)}")
                    sys.stdout.flush()
    else:
        for mid in mesh_ids:
            try:
                ang, rr, nV, nF = process_one(args.i, mid, args.before)
            except Exception as exc:
                print(f"Failed on {mid}: {exc}")
                continue
            counts_rows.append((mid, nV, nF))
            if ang.size == 0:
                continue
            rows_id.append(np.full(ang.shape, mid, dtype=object))
            rows_ang.append(ang)
            rows_rr.append(rr)
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  processed {n_ok}/{len(mesh_ids)}")
                sys.stdout.flush()

    print(f"Successfully evaluated {n_ok} meshes.")

    if counts_rows:
        counts_df = pd.DataFrame(counts_rows, columns=["mesh_id", "num_vertices", "num_faces"])
        stem, ext = os.path.splitext(args.o)
        counts_path = f"{stem}_counts{ext or '.csv'}"
        counts_df.to_csv(counts_path, index=False)
        print(f"Saved per-mesh #V/#F to {counts_path} ({len(counts_df)} meshes)")

    if not rows_ang:
        print("No per-triangle data to write.")
        sys.exit(0)

    df = pd.DataFrame(
        {
            "mesh_id": np.concatenate(rows_id),
            "min_angle": np.concatenate(rows_ang),
            "radius_ratio": np.concatenate(rows_rr),
        }
    )
    print(f"Total triangles: {len(df):,}")
    print(df.head())
    df.to_csv(args.o, index=False)
    print(f"Saved to {args.o}")
    print(f"Total runtime: {time.time() - start:.2f}s")
