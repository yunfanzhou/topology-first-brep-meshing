"""Topology check: does the output mesh match the B-rep's topology?

usage: python check_topology.py <model.hdf5> <output_folder>

Checks the forward stage's fused.pkl and, when present, the remeshing stage's remeshed.vtu /
remeshed_edges.vtu in <output_folder>:
- corners: every B-rep vertex is one mesh vertex, distinct per vertex, and every incident B-rep
  edge's chain ends there;
- edges: every B-rep edge is a chain of mesh edges between the mesh vertices of its two corners;
- faces: every B-rep face is a patch, and every chain segment borders exactly the patches of the
  B-rep faces that share that edge.
Exits with status 1 if any check fails.
"""
import argparse
import os
import sys
from collections import defaultdict

import meshio
import numpy as np
from abs import read_parts

from cad_io.read import load_fused_output


def brep_topology(part):
    """Edge -> (corner, corner), edge -> faces, corner -> edges, and the face ids of the B-rep."""
    edge2corners, edge2faces, corner2edges, face_ids = {}, {}, {}, set()
    for face in part.Solid.faces:
        face_ids.add(face.id)
        for loop in face.loops:
            for he in loop.halfedges:
                eid, cA, cB = he.edge.id, he.edge.start_vertex, he.edge.end_vertex
                if eid in edge2corners and edge2corners[eid] not in [(cA, cB), (cB, cA)]:
                    raise ValueError(f"Input: edge {eid} has inconsistent corners.")
                edge2corners.setdefault(eid, (cA, cB))
                faces = edge2faces.setdefault(eid, [])
                if face.id not in faces:
                    faces.append(face.id)
                corner2edges.setdefault(cA, set()).add(eid)
                corner2edges.setdefault(cB, set()).add(eid)
    return edge2corners, edge2faces, corner2edges, face_ids


def verify_corners(corner2vid, corner2edges, edge2chain):
    for cid, vid in corner2vid.items():
        if cid not in corner2edges:
            raise ValueError(f"Corner {cid} has no edges in the input.")
        for eid in corner2edges[cid]:
            if eid not in edge2chain:
                raise ValueError(f"Corner {cid}: incident edge {eid} missing in the mesh.")
            chain = edge2chain[eid]
            if vid not in (chain[0], chain[-1]):
                raise ValueError(f"Corner {cid}: edge {eid} chain ends {chain[0], chain[-1]} "
                                 f"do not include the corner vertex {vid}.")
    seen = {}
    for cid, vid in corner2vid.items():
        if vid in seen:
            raise ValueError(f"Corners {seen[vid]} and {cid} share mesh vertex {vid}.")
        seen[vid] = cid


def verify_edges(corner2vid, edge2corners, edge2chain):
    for eid, (cA, cB) in edge2corners.items():
        if eid not in edge2chain:
            raise ValueError(f"Edge {eid} missing in the mesh.")
        chain = edge2chain[eid]
        ends = (corner2vid[cA], corner2vid[cB])
        for v in (chain[0], chain[-1]):
            if v not in ends:
                raise ValueError(f"Edge {eid}: chain end {v} is not one of its corner vertices {ends}.")


def verify_faces(F, edge2chain, edge2faces, patch2fid, face_ids):
    fid2patch = {}
    for patch_id, fids in patch2fid.items():
        for fid in fids:
            fid2patch[fid] = patch_id
    for face_id in face_ids:
        if face_id not in patch2fid:
            raise ValueError(f"Face {face_id} missing in the mesh.")

    mesh_edge2tris = defaultdict(set)
    for t, tri in enumerate(F):
        for i in range(3):
            mesh_edge2tris[tuple(sorted((tri[i], tri[(i + 1) % 3])))].add(t)

    # every chain segment borders exactly the patches of the B-rep faces sharing the edge
    mesh_edge2brep_edge = {}
    for eid, chain in edge2chain.items():
        brep_faces = set(edge2faces[eid])
        for i in range(len(chain) - 1):
            e = tuple(sorted((chain[i], chain[i + 1])))
            mesh_edge2brep_edge[e] = eid
            patches = {fid2patch.get(t) for t in mesh_edge2tris.get(e, set())}
            if patches != brep_faces:
                raise ValueError(f"Edge {eid}: segment {e} borders patches {patches}, "
                                 f"the B-rep edge borders faces {brep_faces}.")

    # a triangle may only touch chains of edges of its own B-rep face
    for t, tri in enumerate(F):
        for i in range(3):
            e = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            eid = mesh_edge2brep_edge.get(e)
            if eid is not None and fid2patch.get(t) not in edge2faces[eid]:
                raise ValueError(f"Triangle {t} (patch {fid2patch.get(t)}) has segment {e} of "
                                 f"edge {eid}, which is not an edge of that face.")


def verify_topology(part, F, patch2fid, edge2chain, corner2vid):
    edge2corners, edge2faces, corner2edges, face_ids = brep_topology(part)
    for cid in corner2edges:
        if corner2vid.get(cid) is None:
            raise ValueError(f"Corner {cid} has no mesh vertex.")
    verify_corners(corner2vid, corner2edges, edge2chain)
    verify_edges(corner2vid, edge2corners, edge2chain)
    verify_faces(F, edge2chain, edge2faces, patch2fid, face_ids)


def ordered_chain(pairs, corner_vids):
    """Order unordered segments into one chain; a closed chain starts and ends at a corner."""
    adj = defaultdict(list)
    for a, b in pairs:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    for v, nbrs in adj.items():
        if len(nbrs) > 2:
            raise ValueError(f"vertex {v} has {len(nbrs)} segments, the chain branches")
    ends = [v for v, nbrs in adj.items() if len(nbrs) == 1]
    if ends:
        start = next((v for v in ends if v in corner_vids), ends[0])
    else:
        start = next((v for v in corner_vids if v in adj), None)
        if start is None:
            raise ValueError("closed chain without a corner vertex")
    chain, used, cur = [start], set(), start
    while True:
        nxt = next((n for n in adj[cur] if (min(cur, n), max(cur, n)) not in used), None)
        if nxt is None:
            break
        used.add((min(cur, nxt), max(cur, nxt)))
        chain.append(nxt)
        cur = nxt
    if len(used) != len(pairs):
        raise ValueError("segments are not connected")
    return chain


def load_remeshed(folder):
    """Triangles, patch -> triangles, edge -> chain and corner -> vertex of the remeshed output."""
    tri = meshio.read(os.path.join(folder, "remeshed.vtu"))
    F = tri.cells_dict["triangle"]
    patch2fid = defaultdict(list)
    for t, p in enumerate(np.asarray(tri.cell_data_dict["patch_id"]["triangle"]).ravel()):
        patch2fid[int(p)].append(t)

    edges = meshio.read(os.path.join(folder, "remeshed_edges.vtu"))
    corner2vid = {}
    for vid, c in enumerate(np.asarray(edges.point_data["vertex_id"]).ravel()):
        if c >= 0:
            corner2vid[int(c)] = vid
    lines = np.vstack([cb.data for cb in edges.cells if cb.type == "line"])
    curve_ids = np.concatenate([np.asarray(a).ravel() for a in edges.cell_data_dict["edge_id"].values()])
    edge2pairs = defaultdict(list)
    for (a, b), cid in zip(lines, curve_ids):
        if cid >= 0:
            edge2pairs[int(cid)].append((a, b))
    corner_vids = set(corner2vid.values())
    edge2chain = {}
    for eid, pairs in edge2pairs.items():
        try:
            edge2chain[eid] = ordered_chain(pairs, corner_vids)
        except ValueError as e:
            raise ValueError(f"Edge {eid}: {e}")
    # zero-length B-rep edges have no segments: they are just their corner's vertex. In fused.pkl,
    # next to the remeshed mesh, their chain is that corner's vertex alone.
    _, _, _, fused_chains, fused_corners = load_fused_output(os.path.join(folder, "fused.pkl"))[:5]
    corner_of_vid = {vid: c for c, vid in fused_corners.items()}
    for eid, chain in fused_chains.items():
        if len(chain) < 2:
            edge2chain[int(eid)] = [corner2vid[corner_of_vid[chain[0]]]]
    return F, patch2fid, edge2chain, corner2vid


def main():
    ap = argparse.ArgumentParser(description="Check that the output mesh matches the B-rep's topology.")
    ap.add_argument("model", help="Input model (.hdf5)")
    ap.add_argument("output_folder", help="Folder with fused.pkl (and remeshed.vtu after remeshing)")
    args = ap.parse_args()
    part = read_parts(args.model)[0]

    checks = []
    fused = os.path.join(args.output_folder, "fused.pkl")
    if os.path.exists(fused):
        checks.append(("forward", lambda: load_fused_output(fused)[1:5]))
    if os.path.exists(os.path.join(args.output_folder, "remeshed.vtu")):
        checks.append(("remeshed", lambda: load_remeshed(args.output_folder)))
    if not checks:
        sys.exit(f"No fused.pkl or remeshed.vtu in {args.output_folder}.")

    ok = True
    for name, load in checks:
        try:
            F, patch2fid, edge2chain, corner2vid = load()
            verify_topology(part, F, patch2fid, edge2chain, corner2vid)
            print(f"{name}: topology OK")
        except Exception as e:
            ok = False
            print(f"{name}: topology FAILED: {e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
