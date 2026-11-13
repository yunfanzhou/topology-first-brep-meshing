"""Remeshing stage: remeshes the forward stage's fused.pkl with wildmeshing-toolkit's isotropic remeshing.

usage: python remeshing/remesh.py <output_folder> [--len_rel 0.05] [--cleanup_angle 0] [--threads 0]

<output_folder> is the folder main.py wrote fused.pkl into. Runs the isotropic remeshing of the
wildmeshing package (requirements.txt) in memory on the fused mesh with its patch ids, its feature
chains and its corners, which it keeps, and writes remeshed.vtu, remeshed_edges.vtu and
remeshed_report.json (see README).
--len_rel L is the target edge length, L times the bounding box diagonal (default 0.05, the value of the
paper's experiments). It is independent of the forward stage's --error_rel, which is a geometric error,
not a length. With --len_rel 0 no target is given and the remeshing uses the average edge length of the
fused mesh. That average is far below what the large faces need (the forward mesh is much finer near the
features), so it refines: on assemblies the meshes grew 2 to 4 times and the remeshing took up to an hour
or more.
--cleanup_angle A repairs, after every iteration, the triangles whose smallest angle is under A degrees
(collapse, swap or vertex move). It is off by default (0): it runs serially over most of the mesh while
a fine forward mesh is coarsened, and took 97% of the remeshing time (20 min of 20.5) on a 517k-triangle
input.
"""
import argparse
import json
import os
import pickle

import meshio
import numpy as np


def load_fused(path):
    """V, F, patch id per triangle, B-rep edge -> chain of vertices, B-rep vertex -> vertex."""
    with open(path, "rb") as f:
        V, F, patch2fids, edge2chain, corner2vid = pickle.load(f)[:5]
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    patch = np.full(len(F), -1, dtype=np.int64)
    for p, fids in patch2fids.items():
        patch[np.asarray(list(fids), dtype=np.int64)] = int(p)
    if (patch < 0).any():
        raise ValueError(f"{path}: {(patch < 0).sum()} triangles have no patch")
    return V, F, patch, edge2chain, corner2vid


def features(V, edge2chain, corner2vid):
    """The remeshing's features, a mesh on the rows of V: every chain edge, with its B-rep edge id,
    and every corner as a fixed vertex, with its B-rep vertex id. Zero-length B-rep edges (one-vertex
    chains) add nothing."""
    edges, edge_ids = [], []
    for eid, chain in sorted(edge2chain.items()):
        for a, b in zip(chain[:-1], chain[1:]):
            edges.append([int(a), int(b)])
            edge_ids.append(int(eid))
    corners = sorted(corner2vid.items())
    cells = {}
    if edges:
        cells["edge"] = {"connectivity": np.array(edges, dtype=np.int64),
                         "attributes": {"edge_id": np.array(edge_ids, dtype=np.int64)}}
    if corners:
        cells["point"] = {"connectivity": np.array([[int(v)] for _, v in corners], dtype=np.int64),
                          "attributes": {"vertex_id": np.array([int(c) for c, _ in corners], dtype=np.int64)}}
    return {"V": V, "cells": cells}


def remesh(V, F, patch, edge2chain, corner2vid, len_rel=None, threads=0, cleanup_angle=0.0):
    """Remeshes (V, F) in memory, keeping the patches, the chains and the corners. Returns the
    remeshed vertices and triangles, the patch id of each triangle, the B-rep vertex id of each
    vertex (-1 for none), the chain edges with their B-rep edge ids, and the remeshing's report."""
    import wildmeshing as wm

    # The envelope that the earlier remeshing app used: half the input's average edge length.
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    eps_rel = 0.5 * float(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean()) / diag

    surface = {"V": V, "cells": {"triangle": {"connectivity": F, "attributes": {"patch_id": patch}}}}
    outputs, report = wm.isotropic_remeshing({
        "input": "surface",
        "features": "features",
        "feature_fields": ["patch_id"],  # carried to every remeshed triangle
        # no target (None, 0 or negative): negative, and the remeshing takes the input's average edge length
        "length_rel": len_rel if len_rel is not None and len_rel > 0 else -1.0,
        "eps_rel": eps_rel,
        "use_sample_envelope": True,
        "max_iterations": 3,
        "check_inversions": True,
        "swap_quality": True,
        "limit_collapse_length": True,  # no collapse leaves an edge over 4/3 of the target
        "cleanup_angle": float(cleanup_angle),  # > 0: repair the triangles under it after every iteration
        "freeze_boundary": False,  # an open boundary is B-rep edges, which stay as curves
        "remove_duplicate_eps": -1.0,  # fused.pkl is welded already: merge nothing
        "num_threads": threads,
    }, {"surface": surface, "features": features(V, edge2chain, corner2vid)})
    mesh, curves = outputs["mesh"], outputs["edges"]
    triangles = mesh["cells"]["triangle"]
    return (mesh["V"], triangles["connectivity"], triangles["attributes"]["patch_id"]["values"][:, 0],
            mesh["vertex_attributes"]["vertex_id"]["values"][:, 0],
            curves["cells"]["edge"]["connectivity"],
            curves["cells"]["edge"]["attributes"]["edge_id"]["values"][:, 0], report)


def main():
    parser = argparse.ArgumentParser(description="Remesh the forward stage output with wildmeshing-toolkit.")
    parser.add_argument("output_folder", type=str, help="Folder containing fused.pkl from main.py")
    parser.add_argument("--len_rel", type=float, default=0.05,
                        help="Target edge length relative to the bounding box diagonal (default: 0.05). "
                             "0: no target, the remeshing uses the average edge length of the fused mesh")
    parser.add_argument("--cleanup_angle", type=float, default=0.0,
                        help="Repair, after every iteration, the triangles whose smallest angle is under this "
                             "many degrees (default: 0, no repair; the repair is slow on fine forward meshes)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Threads of the remeshing (default: 0, serial and deterministic)")
    args = parser.parse_args()

    out = os.path.abspath(args.output_folder)
    fused = os.path.join(out, "fused.pkl")
    if not os.path.exists(fused):
        raise SystemExit(f"No fused.pkl in {out}; run main.py first.")
    V, F, patch_id, vertex_id, E, edge_id, report = remesh(*load_fused(fused), args.len_rel, args.threads,
                                                           args.cleanup_angle)

    meshio.write(os.path.join(out, "remeshed.vtu"), meshio.Mesh(
        V, [("triangle", F)], point_data={"vertex_id": vertex_id}, cell_data={"patch_id": [patch_id]}))
    meshio.write(os.path.join(out, "remeshed_edges.vtu"), meshio.Mesh(
        V, [("line", E)], point_data={"vertex_id": vertex_id}, cell_data={"edge_id": [edge_id]}))
    with open(os.path.join(out, "remeshed_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print(f"remeshed mesh written to {os.path.join(out, 'remeshed.vtu')}: {len(F)} triangles")


if __name__ == "__main__":
    main()
