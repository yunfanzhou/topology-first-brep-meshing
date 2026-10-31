"""The remeshing stage on a cube of six patches, in memory."""
import itertools
import os

import numpy as np

from remeshing.remesh import remesh


def cube(n):
    """The unit cube, each face an n x n grid of its own patch, welded: V, F, patch per triangle,
    B-rep edge -> chain of vertices, B-rep vertex -> vertex."""
    index, V, F, patch = {}, [], [], []

    def vertex(p):
        key = tuple(int(round(c * n)) for c in p)
        if key not in index:
            index[key] = len(V)
            V.append(p)
        return index[key]

    for pid, (axis, side) in enumerate(itertools.product(range(3), (0, 1))):
        u, v = [a for a in range(3) if a != axis]
        for i, j in itertools.product(range(n), repeat=2):
            def at(di, dj):
                p = np.zeros(3)
                p[axis], p[u], p[v] = side, (i + di) / n, (j + dj) / n
                return vertex(p)
            a, b, c, d = at(0, 0), at(1, 0), at(1, 1), at(0, 1)
            # outward normals: flip the faces with u x v pointing in
            inward = (side == 0) == (np.cross(np.eye(3)[u], np.eye(3)[v])[axis] > 0)
            for t in ([a, b, c], [a, c, d]):
                F.append(t[::-1] if inward else t)
                patch.append(pid)
    corners = {cid: index[tuple(int(c) * n for c in p)]
               for cid, p in enumerate(itertools.product((0, 1), repeat=3))}
    edges = {}
    for eid, (a, b) in enumerate((a, b) for a, b in itertools.combinations(range(8), 2)
                                 if bin(a ^ b).count("1") == 1):
        pa, pb = np.array(V[corners[a]]), np.array(V[corners[b]])
        edges[eid] = [vertex(pa + (pb - pa) * k / n) for k in range(n + 1)]
    return np.array(V), np.array(F, dtype=np.int64), np.array(patch, dtype=np.int64), edges, corners


def test_remesh_keeps_patches_chains_and_corners(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    V, F, patch, edges, corners = cube(8)
    Vo, Fo, patch_id, vertex_id, E, edge_id, report = remesh(V, F, patch, edges, corners, 0.1)

    assert report["#f"] == len(Fo) and report["nonmanifold_edges"] == 0
    assert set(patch_id.tolist()) == set(range(6))
    # Each corner is one fixed vertex, with its id; each B-rep edge a chain of the result.
    assert sorted(vertex_id[vertex_id >= 0].tolist()) == sorted(corners)
    assert set(edge_id.tolist()) == set(edges)
    # The patches meet along the chains alone: across any other edge the patch stays the same.
    side = {}
    for t, tri in enumerate(Fo):
        for k in range(3):
            side.setdefault(tuple(sorted((tri[k], tri[(k + 1) % 3]))), []).append(patch_id[t])
    chain = {tuple(sorted(e)) for e in E.tolist()}
    assert all(len(p) == 2 for p in side.values())
    for e, (p, q) in side.items():
        assert (p != q) == (e in chain)
    # In memory: nothing on the disk.
    assert os.listdir(tmp_path) == []
