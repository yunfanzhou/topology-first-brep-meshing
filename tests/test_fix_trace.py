"""Regression tests for the fixes in operations/trace.py (initial and non-initial loop tracing)."""
import numpy as np
import pytest

import operations.trace as trace
from run_one_patch import construct_connectivity_data
from utils.data_structures import (HalfEdge, InitialLoopTracingData, LoopGeometryData, LoopTopologyData,
                                   LoopTracingData, NonInitialLoopTreatmentData, PatchRegularSpaceData)


def square():
    V = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    return V, construct_connectivity_data(V, [[0, 1, 2], [0, 2, 3]])


def initial_loop_new(is_last_attempt):
    V, conn = square()
    return trace.trace_initial_loop_new(V, conn, PatchRegularSpaceData(), LoopTopologyData(), LoopGeometryData(), [],
                                        InitialLoopTracingData(), ignore_trace_geometry=True,
                                        is_last_attempt=is_last_attempt)


def stub_initial_loop(monkeypatch, heuristic_results, fallback=None):
    """Stub the heuristic (one result per call: True = failed, False = ok, an exception = raised)
    and the hard fallback; returns the list of calls."""
    results = iter(heuristic_results)
    calls = []

    def heuristic(*args, **kwargs):
        calls.append("heuristic")
        r = next(results)
        if isinstance(r, Exception):
            raise r
        return r

    def run_fallback(*args, **kwargs):
        calls.append("fallback")
        if fallback is not None:
            raise fallback

    monkeypatch.setattr(trace, "trace_initial_loop_with_heuristics", heuristic)
    monkeypatch.setattr(trace, "trace_initial_loop_fallback", run_fallback)
    return calls


NO_PATH = ValueError("No valid path found among closest candidates")


@pytest.mark.parametrize("first", [True, NO_PATH])
def test_first_heuristic_failure_retries_the_face_before_the_last_attempt(monkeypatch, first):
    # a failed first heuristic means the mesh is not fine enough: ValueError (mesh_one_face
    # retries on a refined mesh), no second heuristic, no fallback
    calls = stub_initial_loop(monkeypatch, [first])
    with pytest.raises(ValueError):
        initial_loop_new(is_last_attempt=False)
    assert calls == ["heuristic"]


@pytest.mark.parametrize("results, expected_calls, fallback_used", [
    ([False], ["heuristic"], False),
    ([True, False], ["heuristic", "heuristic"], False),
    ([NO_PATH, False], ["heuristic", "heuristic"], False),
    ([True, True], ["heuristic", "heuristic", "fallback"], True),
    ([NO_PATH, NO_PATH], ["heuristic", "heuristic", "fallback"], True),
])
def test_last_attempt_runs_second_heuristic_then_fallback(monkeypatch, results, expected_calls, fallback_used):
    # the return value is summary["initial_loop_fallback"]
    calls = stub_initial_loop(monkeypatch, results)
    assert initial_loop_new(is_last_attempt=True) is fallback_used
    assert calls == expected_calls


def test_first_heuristic_success_needs_no_last_attempt(monkeypatch):
    calls = stub_initial_loop(monkeypatch, [False])
    assert initial_loop_new(is_last_attempt=False) is False
    assert calls == ["heuristic"]


def test_fallback_value_error_is_a_bug(monkeypatch):
    # the hard fallback must always succeed: its ValueError ends the run instead of failing the face
    cause = ValueError("Error: The initial loop fallback tracing failed.")
    stub_initial_loop(monkeypatch, [True, True], fallback=cause)
    with pytest.raises(RuntimeError, match="must always succeed") as info:
        initial_loop_new(is_last_attempt=True)
    assert info.value.__cause__ is cause


@pytest.mark.parametrize("exc", [IndexError("list index out of range"), RuntimeError("bug")])
def test_non_value_errors_are_not_caught(monkeypatch, exc):
    stub_initial_loop(monkeypatch, [exc])
    with pytest.raises(type(exc)):
        initial_loop_new(is_last_attempt=True)


def test_no_progress_error_is_a_sub_loop_closure_error():
    # so trace_all_loops_one_patch first retraces the non-initial loops without corner reuse
    e = ValueError("Error: Non initial loop fallback tracing made no progress: trapped loops [1] are already treated.")
    assert trace.is_sub_loop_closure_error(e)
    assert not trace.is_sub_loop_closure_error(ValueError("Error: Non initial loop fallback tracing failed."))


def test_non_initial_fallback_stops_when_trapped_loops_repeat(monkeypatch):
    # previous loop 1 is connected to the boundary, the retrace traps it again: with
    # nothing new to connect, the fallback raises instead of retracing forever
    prunes = []

    def prune(*args):
        prunes.append(1)
        if len(prunes) > 3:
            raise RuntimeError("the fallback retraces the same trapped loops forever")
        return False, [1]

    def trace_edge(V, conn, rsd, poly, edge_tracing_data, first_pair, second_pair, **kwargs):
        edge_tracing_data.traced_vids = [1, 2, 1]

    monkeypatch.setattr(trace, "regular_space_using_PRSD", lambda *args: None)
    monkeypatch.setattr(trace, "dijkstra_shortest_path_between_loops", lambda *args: [3, 0])
    monkeypatch.setattr(trace, "trace_one_feature_edge", trace_edge)
    monkeypatch.setattr(trace, "prune_after_subloop_closure", prune)
    V, conn = square()
    rsd = PatchRegularSpaceData(previous_loops=[[0, 1, 2], [3]])
    loop = LoopTopologyData(halfedges=[HalfEdge(5, True, False, 2, 2)])
    treatment = NonInitialLoopTreatmentData(loops_need_treatment=[1])
    with pytest.raises(ValueError, match=r"made no progress: trapped loops \[1\]"):
        trace.trace_non_initial_loop_fallback(V, conn, rsd, loop, LoopGeometryData({5: np.zeros((3, 3))}),
                                              LoopTracingData(), treatment, ignore_trace_geometry=True)
    assert len(prunes) == 1


def test_initial_fallback_without_disk_component_raises_value_error():
    # no component is bounded by traced edges only: after_trace_initial_loop_fallback
    # returns False instead of an IndexError, and the fallback raises ValueError
    V, conn = square()
    F = [face[:] for face in conn.F]
    assert trace.after_trace_initial_loop_fallback(conn, PatchRegularSpaceData(traced_edge_set={(0, 1)})) is False
    assert conn.F == F
    with pytest.raises(ValueError, match="initial loop fallback tracing failed"):
        trace.trace_initial_loop_fallback(V, conn, PatchRegularSpaceData(), LoopTopologyData(), LoopGeometryData(), [],
                                          InitialLoopTracingData(), ignore_trace_geometry=True)


def cone_initial_loop(corner_id_to_vid):
    """Initial loop of a cone: degenerate apex feature 0 at corner 0 (apex vid 7), seam 1
    from corner 0 to 1 and back, rim 2 around corner 1."""
    topo = LoopTopologyData(halfedges=[HalfEdge(0, True, True, 0, 0), HalfEdge(1, True, False, 0, 1),
                                       HalfEdge(2, True, False, 1, 1), HalfEdge(1, False, False, 1, 0)],
                            seam_ids_set={1})
    geometry = LoopGeometryData({0: np.empty((0, 3)), 1: np.zeros((3, 3)), 2: np.zeros((3, 3))})
    init = InitialLoopTracingData(initial_loop_id=0, singularity_corner_mapping={0: 7})
    init.loop_tracing_data.corner_id_to_vid = dict(corner_id_to_vid)
    return topo, geometry, init


def fake_trace(new_vids, record=None):
    """Stand-in for trace_one_feature_edge: corners without a vid get the next of new_vids,
    one interior vertex per edge, and the regular space data is updated as by a real trace."""
    def trace_edge(V, conn, rsd, poly, edge_tracing_data, first_pair, second_pair, *args, **kwargs):
        if record is not None:
            record(rsd)
        if len(poly) == 0:
            return  # an empty (degenerate) polyline traces nothing
        for pair in (first_pair, second_pair):
            if pair[1] is None:
                pair[1] = new_vids.pop(0)
        edge_tracing_data.traced_vids = [first_pair[1], new_vids.pop(0), second_pair[1]]
        trace.update_regular_space_data(edge_tracing_data.traced_vids, rsd)
    return trace_edge


def test_apex_check_runs_when_boundary_is_prohibited(monkeypatch):
    # second heuristic: the boundary is prohibited, so the boundary-hug check is
    # skipped; the seam traced from a substitute vertex 11 next to the apex must
    # still be rejected
    monkeypatch.setattr(trace, "trace_one_feature_edge", fake_trace([5, 20, 21]))
    monkeypatch.setattr(trace, "after_trace_initial_loop", lambda *args: True)
    V, conn = square()
    rsd = PatchRegularSpaceData(prohibited_vids_set={0, 1, 2, 3})
    topo, geometry, init = cone_initial_loop({0: 11})
    with pytest.raises(ValueError, match=r"apex corner 0 \(vid 7\) is not the end"):
        trace.trace_initial_loop_with_heuristics(V, conn, rsd, topo, geometry, init, off_boundary=True)


def test_apex_stays_prohibited_after_boundary_hug_revert(monkeypatch):
    # the first halfedge ends at the apex 7 and hugs the boundary: reverting it must not
    # release the apex, or the retried trace could pass through it
    apex_prohibited = []
    hug = iter([True])
    monkeypatch.setattr(trace, "trace_one_feature_edge",
                        fake_trace([5, 20, 6, 21, 22], record=lambda rsd: apex_prohibited.append(7 in rsd.prohibited_vids_set)))
    monkeypatch.setattr(trace, "detect_boundary_hug_after_halfedge", lambda *args: next(hug, False))
    monkeypatch.setattr(trace, "after_trace_initial_loop", lambda *args: True)
    V, conn = square()
    rsd = PatchRegularSpaceData()
    topo = LoopTopologyData(halfedges=[HalfEdge(1, True, False, 1, 0), HalfEdge(0, True, True, 0, 0),
                                       HalfEdge(2, True, False, 0, 1)])
    geometry = LoopGeometryData({1: np.zeros((3, 3)), 0: np.empty((0, 3)), 2: np.zeros((3, 3))})
    init = InitialLoopTracingData(initial_loop_id=0, singularity_corner_mapping={0: 7})
    init.loop_tracing_data.corner_id_to_vid = {0: 7}
    assert trace.trace_initial_loop_with_heuristics(V, conn, rsd, topo, geometry, init, off_boundary=True) is False
    # first trace, retry after the revert, last halfedge
    assert apex_prohibited == [True, True, True]
    assert init.loop_tracing_data.feature_id_to_edge_tracing_data[1] == [6, 21, 7]


def test_non_initial_fallback_skips_degenerate_halfedges(monkeypatch):
    # a non-initial loop through an apex: degenerate feature 0 at corner 0, then features
    # 1 and 2 back to corner 0. Traced, the degenerate halfedge mapped corner 0 to None
    # (the loop then never closed there) and looked like a sub-loop closure.
    prunes = []

    def prune(*args):
        prunes.append(1)
        return True, []

    monkeypatch.setattr(trace, "regular_space_using_PRSD", lambda *args: None)
    monkeypatch.setattr(trace, "dijkstra_shortest_path_between_loops", lambda *args: [3, 0])
    monkeypatch.setattr(trace, "trace_one_feature_edge", fake_trace([5, 6, 20, 7, 21]))
    monkeypatch.setattr(trace, "prune_after_subloop_closure", prune)
    V, conn = square()
    rsd = PatchRegularSpaceData(previous_loops=[[0, 1, 2], [3]])
    loop = LoopTopologyData(halfedges=[HalfEdge(0, True, True, 0, 0), HalfEdge(1, True, False, 0, 1),
                                       HalfEdge(2, True, False, 1, 0)])
    geometry = LoopGeometryData({0: np.empty((0, 3)), 1: np.zeros((3, 3)), 2: np.zeros((3, 3))})
    loop_tracing_data = LoopTracingData()
    trace.trace_non_initial_loop_fallback(V, conn, rsd, loop, geometry, loop_tracing_data,
                                          NonInitialLoopTreatmentData(loops_need_treatment=[1]),
                                          ignore_trace_geometry=True)
    assert loop_tracing_data.corner_id_to_vid[0] == 5
    assert loop_tracing_data.feature_id_to_edge_tracing_data[2][-1] == 5
    assert 0 not in loop_tracing_data.traced_id_set
    assert len(prunes) == 1
