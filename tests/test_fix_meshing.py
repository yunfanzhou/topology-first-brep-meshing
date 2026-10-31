"""Regression tests for the meshing-group fixes (main.py, initial_meshing.py,
run_one_patch.py, utils/regular_space.py)."""
import pytest

from utils.data_structures import LoopTracingData, PatchRegularSpaceData, PatchTracingData


def test_corner_stitch_without_path_raises():
    # corner 5 has vid 0 in the initial loop and vid 3 in loop 1, on two separate
    # triangles: no path joins them, so the corner cannot be stitched
    from run_one_patch import edit_loops_with_shared_corners
    V = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]]
    F = [[0, 1, 2], [3, 4, 5]]
    ptd = PatchTracingData()
    ptd.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid = {5: 0}
    ptd.remaining_loops_tracing_data = {1: LoopTracingData(corner_id_to_vid={5: 3})}
    with pytest.raises(ValueError, match="No path from apex vid 0 to vid 3"):
        edit_loops_with_shared_corners(V, F, PatchRegularSpaceData(), ptd)
    assert ptd.remaining_loops_tracing_data[1].corner_id_to_vid == {5: 3}


def test_fallback_mesh_keeps_its_sub_domains(monkeypatch):
    # The per-sub-domain mmg call fails after decide_trim_domain returned new
    # sub-domains: the fallback is still the initial grid, so it must come with
    # the grid's sub-domains, both for the fallback processing and the next attempt.
    import numpy as np
    import initial_meshing as im
    grid_V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    grid_F = np.array([[0, 1, 2], [0, 2, 3]])
    grid_sd = [(0., 1., 0., 1.)]
    new_sd = [(0., 0.4, 0., 1.), (0.6, 1., 0., 1.)]

    def create_initial_mesh(face, u_lower=None, **kw):
        return grid_V.copy(), grid_F.copy(), {}, False, grid_sd

    def remesh(surface, uvs, F, **kw):
        if remesh.calls:
            raise RuntimeError("mmg failed")
        remesh.calls += 1
        return uvs * 0.5, F, 1.0
    remesh.calls = 0

    seen = {}

    def simple_processing(V, F, face, sub_domains, *rest):
        seen["sub_domains"] = sub_domains
        return (None,) * 6

    class Surface:
        def sample(self, uv):
            return np.c_[uv, np.zeros(len(uv))]

    class Face:
        id, singularities, surface = 0, {}, Surface()

    monkeypatch.setattr(im, "curvature_capped_error", lambda surface, e: e)
    monkeypatch.setattr(im, "create_initial_mesh", create_initial_mesh)
    monkeypatch.setattr(im, "detect_periodicity", lambda *a: (False, False))
    monkeypatch.setattr(im, "remesh_with_error_and_edge_metric", remesh)
    monkeypatch.setattr(im, "detect_collapsed_iso_lines", lambda *a, **kw: {})
    monkeypatch.setattr(im, "decide_trim_domain", lambda face, **kw: (new_sd, 0.1, False, 0))
    monkeypatch.setattr(im, "insert_feature_edges_uv", lambda V, F, *a, **kw: (V, F))
    monkeypatch.setattr(im, "initial_mesh_simple_processing", simple_processing)
    with pytest.warns(UserWarning, match="mmg failed"):
        *_, V_fb, F_fb, sd_fb = im._run_initial_meshing_for_face(Face(), {}, target_error=0.1)
    assert np.array_equal(V_fb, grid_V) and np.array_equal(F_fb, grid_F)
    assert sd_fb == grid_sd
    assert seen["sub_domains"] == grid_sd


def test_init_grid_mesh_rejects_two_interior_singularities():
    from initial_meshing import init_grid_mesh

    def sing(u):
        return {'first2d': (u, 0.), 'uiso': True, 'point2d': (u, 0.5), 'point3d': (0., 0., 0.)}
    # one interior singularity is supported: two grids, no unfilled triangle
    uvs, F, _, sub_domains = init_grid_mesh({'000': sing(0.5)}, 4, 4, 0., 1., 0., 1.)
    assert len(sub_domains) == 2 and (F != 0).any(axis=1).all()
    with pytest.raises(ValueError, match="at most one interior singularity"):
        init_grid_mesh({'000': sing(0.3), '001': sing(0.7)}, 4, 4, 0., 1., 0., 1.)


def traced_ptd():
    """Corner 5 at vid 0 (initial loop) and vid 3 (loop 1); feature 1 traced as 0-1-2."""
    ptd = PatchTracingData()
    lt = ptd.initial_loop_tracing_data.loop_tracing_data
    lt.corner_id_to_vid = {5: 0}
    lt.feature_id_to_edge_tracing_data = {1: [0, 1, 2]}
    ptd.remaining_loops_tracing_data = {1: LoopTracingData(corner_id_to_vid={5: 3})}
    return ptd


def test_frozen_and_traced_edge_sets_are_separate_after_construction():
    from utils.regular_space import construct_patch_regular_space_from_tracing_data
    rsd = PatchRegularSpaceData()
    construct_patch_regular_space_from_tracing_data(rsd, traced_ptd())
    assert rsd.frozen_edges_set == rsd.traced_edge_set == {(0, 1), (1, 2)}
    rsd.frozen_edges_set.add((2, 3))
    assert (2, 3) not in rsd.traced_edge_set


def test_frozen_and_traced_edge_sets_are_separate_when_stitching(monkeypatch):
    import run_one_patch
    seen = []

    def stitch(V, connectivity, corner_id, vids, corners_to_process, rsd, ptd):
        seen.append(rsd.frozen_edges_set is not rsd.traced_edge_set)
        rsd.frozen_edges_set.add((2, 3))  # a stitch path is frozen, not traced
        assert (2, 3) not in rsd.traced_edge_set
        return None
    monkeypatch.setattr(run_one_patch, "trace_and_collapse_between_two_vertices", stitch)
    V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    run_one_patch.edit_loops_with_shared_corners(V, [[0, 1, 2], [0, 2, 3]], PatchRegularSpaceData(), traced_ptd())
    assert seen == [True]


class FakeFace:
    id, surface = 7, None


def failing_initial_meshing(monkeypatch, exc):
    import main
    calls = []

    def run_initial_meshing_for_face(face, *args, attempt, **kwargs):
        calls.append(attempt)
        raise exc
    monkeypatch.setattr(main, "run_initial_meshing_for_face", run_initial_meshing_for_face)
    return main, calls


@pytest.mark.parametrize("exc", [RuntimeError("mmg failed"), AssertionError(), IndexError("out of range"),
                                 Exception("No progress made"), ValueError("not edge manifold")])
def test_mesh_one_face_retries_any_exception(monkeypatch, exc):
    main, calls = failing_initial_meshing(monkeypatch, exc)
    r = main.mesh_one_face(FakeFace(), {}, None, 0.1, "mmg2d_O3", 1.0)
    assert calls == list(range(main.ATTEMPT_MAX))
    assert not r["ok"] and r["patch_id"] == 7 and r["attempts"] == main.ATTEMPT_MAX
    assert r["error"] == (str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}")


@pytest.mark.parametrize("exc_type", ["MemoryError", "CollapsedInteriorIsoLineError"])
def test_mesh_one_face_does_not_retry_fatal_errors(monkeypatch, exc_type):
    import initial_meshing
    exc_class = MemoryError if exc_type == "MemoryError" else initial_meshing.CollapsedInteriorIsoLineError
    main, calls = failing_initial_meshing(monkeypatch, exc_class("fatal"))
    with pytest.raises(exc_class):
        main.mesh_one_face(FakeFace(), {}, None, 0.1, "mmg2d_O3", 1.0)
    assert calls == [0]


def test_mesh_one_face_passes_the_model_target_to_every_attempt(monkeypatch):
    import main
    targets = []

    def run_initial_meshing_for_face(face, *args, target_error, **kwargs):
        targets.append(target_error)
        raise ValueError("retry")
    monkeypatch.setattr(main, "run_initial_meshing_for_face", run_initial_meshing_for_face)
    main.mesh_one_face(FakeFace(), {}, None, 0.1, "mmg2d_O3", 1.0)
    assert targets == [0.1] * main.ATTEMPT_MAX


def test_retry_skips_the_curvature_cap(monkeypatch):
    import numpy as np
    import initial_meshing as im
    V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    F = np.array([[0, 1, 2], [0, 2, 3]])

    def curvature_capped_error(surface, e):
        raise AssertionError("curvature cap computed on a retry")
    monkeypatch.setattr(im, "curvature_capped_error", curvature_capped_error)
    monkeypatch.setattr(im, "create_initial_mesh", lambda face, **kw: (V, F, {}, False, [(0., 1., 0., 1.)]))
    monkeypatch.setattr(im, "detect_periodicity", lambda *a: (False, False))
    monkeypatch.setattr(im, "initial_mesh_simple_processing", lambda V, F, *rest: (len(V),) + (None,) * 5)
    out = im._run_initial_meshing_for_face(FakeFace(), {}, target_error=0.1, attempt=1,
                                           prev_V_uv=V, prev_F=F, prev_sub_domains=[(0., 1., 0., 1.)])
    assert out[0] == 9 and out[6:] == (None, None, None)  # one upsample level of the 4-vertex mesh


class FakePool:
    """Stands in for ProcessPoolExecutor. plan maps face id -> (delay in s, ok): a face
    with delay 0 is done at submit, the others are finished by a timer unless cancelled."""

    def __init__(self, plan):
        self.plan, self.futures, self.timers, self.shutdowns = plan, {}, [], []

    def submit(self, fn, face_id, *args):
        from concurrent.futures import Future
        import threading
        delay, ok = self.plan[face_id]
        r = dict(ok=ok, patch_id=face_id, attempts=1, duration=0.01)
        r.update(dict(V=[[0., 0., 0.]], connectivity=None, tracing=None) if ok else dict(error="forced failure"))
        fut = self.futures[face_id] = Future()

        def finish():
            if fut.set_running_or_notify_cancel():
                fut.set_result(r)
        if delay == 0:
            finish()
        else:
            self.timers.append(threading.Timer(delay, finish))
            self.timers[-1].start()
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdowns.append(cancel_futures)
        if cancel_futures:  # finished futures stay as they are
            for fut in self.futures.values():
                fut.cancel()
            for timer in self.timers:
                timer.cancel()
        if wait:
            for timer in self.timers:
                timer.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown(wait=True)


def run_main_with_pool(monkeypatch, tmp_path, plan):
    """main() on data/simple/Box.hdf5 (faces 0-5) through the parallel path with a FakePool."""
    import concurrent.futures
    import json
    import os
    import sys
    import main
    import utils.config as config
    pool = FakePool(plan)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", lambda **kwargs: pool)
    monkeypatch.setattr(main, "PARALLEL_MIN_FACES", 2)
    monkeypatch.setattr(config, "oversized_dump_dir", config.oversized_dump_dir)  # restored after the test
    box = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "simple", "Box.hdf5")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["main.py", box, str(out), "--workers", "4"])
    with pytest.raises(Exception) as exc_info:
        main.main()
    checkpoints = sorted(int(p.stem.split("_")[1]) for p in (out / "checkpoints" / "local").glob("patch_*.pkl"))
    with open(out / "result.json") as f:
        result = json.load(f)
    return pool, exc_info.value, checkpoints, result


def test_parallel_failure_cancels_queued_faces(monkeypatch, tmp_path):
    # face 0 is slow, faces 1-4 finish at once, face 5 fails after them
    plan = {0: (0.5, True), 1: (0, True), 2: (0, True), 3: (0, True), 4: (0, True), 5: (0.05, False)}
    pool, exc, checkpoints, result = run_main_with_pool(monkeypatch, tmp_path, plan)
    assert isinstance(exc, ValueError) and "Failed to process patch 5" in str(exc)
    assert pool.shutdowns[0] is True and pool.futures[0].cancelled()
    assert checkpoints == [1, 2, 3, 4]  # written as each face finished
    assert list(result["patches"]) == ["1", "2", "3", "4", "5"]
    assert result["patches"]["5"]["status"] == "failed" and result["status"] == "failed"


def test_parallel_success_records_in_face_order(monkeypatch, tmp_path):
    import main

    def stop(*args):
        raise RuntimeError("stop after the local step")
    monkeypatch.setattr(main, "align_adjacent_patches", stop)
    plan = {0: (0.1, True), 1: (0, True), 2: (0.05, True), 3: (0, True), 4: (0, True), 5: (0, True)}
    pool, exc, checkpoints, result = run_main_with_pool(monkeypatch, tmp_path, plan)
    assert str(exc) == "stop after the local step"
    assert pool.shutdowns == [False]
    assert checkpoints == [0, 1, 2, 3, 4, 5]
    assert list(result["patches"]) == ["0", "1", "2", "3", "4", "5"]
    assert all(p["status"] == "success" for p in result["patches"].values())


def test_default_workers_follows_cpu_affinity(monkeypatch):
    import os
    import main
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 3}, raising=False)
    assert main.default_workers() == 2
    monkeypatch.delattr(os, "sched_getaffinity")  # e.g. macOS, Windows
    assert main.default_workers() == 64
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert main.default_workers() == 1
