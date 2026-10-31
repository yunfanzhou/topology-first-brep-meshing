"""Regression tests for fixes in the geometry code (alignment, snapping, mesh utilities,
consolidation), on small hand-built inputs."""
import numpy as np
import pytest

from operations.align import align_one_segment_of_patches
from operations.consolidate import consolidate, consolidate_one_patch, update_corners
from operations.internal_operations import split_edge_in_mesh_overloaded
from operations.snapping import (get_patch_boundary_condition, solve_one_patch_linear_system,
                                 upsample_feature_edge_polyline_V_wrt_arclen)
from utils.data_structures import ConnectivityMaintainance, LoopGeometryData, LoopTracingData, PatchTracingData
from utils.mesh_utils import adjacency_list, compute_VF, edge_length_mapping


def fan_patch(xs, apex):
    """Triangle fan over the polyline (x, 0, 0), x in xs, to one apex vertex."""
    V = [np.array([x, 0.0, 0.0]) for x in xs] + [np.array(apex, dtype=float)]
    a = len(xs)
    F = [[k, k + 1, a] for k in range(len(xs) - 1)]
    conn = ConnectivityMaintainance(F, compute_VF(F), adjacency_list(F, len(V)),
                                    edge_length_mapping(V, F), {}, set())
    return V, conn


@pytest.mark.parametrize("scale", [0.1, 100.0])
def test_align_snaps_by_normalized_parameter_at_any_scale(scale):
    # Interior vertices at normalized parameters 0.2, 0.6 (patch A) and 0.4, 0.605 (patch B):
    # only 0.6 and 0.605 are within 1% of the segment, at any model scale.
    VA, cA = fan_patch([0.0, 0.2 * scale, 0.6 * scale, scale], [0.5 * scale, scale, 0.0])
    VB, cB = fan_patch([0.0, 0.4 * scale, 0.605 * scale, scale], [0.5 * scale, -scale, 0.0])
    Vs = [VA, VB]
    segments = [[0, 1, 2, 3], [0, 1, 2, 3]]
    aligned = align_one_segment_of_patches(Vs, [cA, cB], segments)
    expected = [0.0, 0.2, 0.4, 0.6025, 1.0]
    for V, conn, vids in zip(Vs, [cA, cB], aligned):
        chain = vids + [3]  # the returned chain leaves out the segment's last vertex
        assert all(b in conn.adj_list[a] for a, b in zip(chain, chain[1:]))
        assert np.allclose([V[v][0] / scale for v in chain], expected)


def triangle_loop_patch(p0_on_feature_12=(0.0, 0.0, 0.0)):
    """Triangle loop of three traced features 10, 11, 12 (corners 0, 1, 2, one inner vertex
    3, 4, 5 each) around interior vertex 6. Corner 0 sits 0.3 above its target."""
    P0, P1, P2 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    V0 = P0 + [0.0, 0.0, 0.3]
    V = [V0, P1, P2, (V0 + P1) / 2, (P1 + P2) / 2, (P2 + V0) / 2, np.array([0.3, 0.3, 0.0])]
    F = [[0, 3, 6], [3, 1, 6], [1, 4, 6], [4, 2, 6], [2, 5, 6], [5, 0, 6]]
    geometry = {0: LoopGeometryData({10: np.array([P0, P1]), 11: np.array([P1, P2]),
                                     12: np.array([P2, p0_on_feature_12])})}
    tracing = PatchTracingData()
    tracing.initial_loop_tracing_data.initial_loop_id = 0
    loop = tracing.initial_loop_tracing_data.loop_tracing_data
    loop.feature_id_to_edge_tracing_data = {10: [0, 3, 1], 11: [1, 4, 2], 12: [2, 5, 0]}
    loop.feature_id_to_segment_upsample_size = {10: [1], 11: [1], 12: [1]}
    loop.feature_id_to_edge_orientation = {10: True, 11: True, 12: True}
    return V, F, geometry, tracing


def test_shared_corner_is_constrained_once_in_harmonic_solve():
    V, F, geometry, tracing = triangle_loop_patch()
    bi, _ = get_patch_boundary_condition(V, geometry, tracing)
    assert sorted(int(v) for v in bi) == [0, 1, 2, 3, 4, 5]
    newV = solve_one_patch_linear_system(V, F, tracing, geometry, summary={})
    # Vertex 6 moves by the mean displacement of its six boundary neighbours:
    # -0.3 (corner 0) and -0.15 (vertices 3 and 5) in z, i.e. -0.1.
    assert np.allclose(newV[6], [0.3, 0.3, -0.1])


def test_shared_corner_with_conflicting_targets_raises():
    V, F, geometry, tracing = triangle_loop_patch(p0_on_feature_12=(0.0, 0.0, 1e-6))
    with pytest.raises(ValueError, match="Boundary vertex 0"):
        get_patch_boundary_condition(V, geometry, tracing)


@pytest.mark.parametrize("edge", [(0, 1), (1, 0)])
def test_split_edge_stores_real_lengths_of_both_halves(edge):
    V, conn = fan_patch([0.0, 10.0], [5.0, 5.0, 0.0])
    split_edge_in_mesh_overloaded(V, conn, edge, new_point=np.array([1.0, 0.0, 0.0]))
    lengths = conn.edge_set_to_length_mapping
    assert lengths[(0, 3)] == pytest.approx(1.0)
    assert lengths[(1, 3)] == pytest.approx(9.0)
    assert lengths == pytest.approx(edge_length_mapping(V, conn.F))


def test_upsample_spaces_points_evenly_on_zero_length_segment():
    polyline = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    patch_V = [np.zeros(3)] * 5  # the traced segment 0-1-2-3-4 has zero length
    out = upsample_feature_edge_polyline_V_wrt_arclen(polyline, [3], [0, 1, 2, 3, 4], patch_V)
    assert np.allclose(out, [[x, 0.0, 0.0] for x in (0.0, 1.0, 2.0, 3.0, 4.0)])


@pytest.mark.parametrize("vid", [3, None])
def test_update_corners_rejects_corner_without_vertex(vid):
    _, _, old2new, _ = consolidate([np.zeros(3)] * 5, [[0, 1, 2], [0, 2, 4]])  # vertex 3 unused
    with pytest.raises(ValueError, match="Corner 7"):
        update_corners({6: 0, 7: vid}, old2new)
    assert 3 not in old2new


def test_consolidate_one_patch_rejects_stale_corner():
    V = [np.array([float(i), float(i * i), 0.0]) for i in range(5)]
    F = [[0, 1, 2], [0, 2, 4]]  # vertex 3 unused
    tracing = PatchTracingData()
    tracing.initial_loop_tracing_data.loop_tracing_data.feature_id_to_edge_tracing_data = {10: [0, 1]}
    tracing.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid = {7: 3}
    tracing.remaining_loops_tracing_data = {1: LoopTracingData({11: [1, 3, 2]})}
    with pytest.raises(ValueError, match="Corner 7"):
        consolidate_one_patch(V, F, tracing)
