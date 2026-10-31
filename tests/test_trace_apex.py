"""Degenerate (singular apex) feature edges in the initial-loop tracing:
once every feature edge with a polyline is traced, an apex is accepted only when its
corner maps to a vertex that ends a traced feature edge; otherwise tracing raises."""
import numpy as np
import pytest

from utils.data_structures import HalfEdge, LoopTopologyData, InitialLoopTracingData
from operations.trace import check_degenerate_apex_features, detect_boundary_hug_after_halfedge


def cone_loop(apex_corner_vid=None, singular_vid=7, seam=(7, 3, 4, 5)):
    """Cone loop: degenerate apex feature 0 at corner 0, seam 1 from corner 0 to 1,
    rim 2 around corner 1. The seam is traced as `seam` (first vid = apex end)."""
    topo = LoopTopologyData(halfedges=[
        HalfEdge(0, True, True, 0, 0),
        HalfEdge(1, True, False, 0, 1),
        HalfEdge(2, True, False, 1, 1),
    ])
    init = InitialLoopTracingData(initial_loop_id=0, singularity_corner_mapping={0: singular_vid})
    lt = init.loop_tracing_data
    lt.feature_id_to_edge_tracing_data = {1: list(seam), 2: [5, 8, 9, 5]}
    lt.traced_id_set = {1, 2}
    lt.corner_id_to_vid = {1: 5}
    if apex_corner_vid is not None:
        lt.corner_id_to_vid[0] = apex_corner_vid
    polylines = {0: np.empty((0, 3)), 1: np.zeros((4, 3)), 2: np.zeros((4, 3))}
    return topo, init, polylines


def test_apex_at_end_of_traced_seam_is_skipped():
    topo, init, polylines = cone_loop(apex_corner_vid=7)
    check_degenerate_apex_features(topo, init)
    # all features with a polyline traced: no hug, nothing else is computed
    assert detect_boundary_hug_after_halfedge(None, None, None, polylines, {1, 2}) is False


def test_apex_found_through_corner_mapping_is_skipped():
    topo, init, _ = cone_loop(apex_corner_vid=7)
    init.singularity_corner_mapping = {}
    check_degenerate_apex_features(topo, init)


def test_apex_found_through_singularity_mapping_is_skipped():
    topo, init, _ = cone_loop(apex_corner_vid=None, singular_vid=7)
    check_degenerate_apex_features(topo, init)


def test_unmapped_apex_corner_raises():
    topo, init, _ = cone_loop(apex_corner_vid=None)
    init.singularity_corner_mapping = {}
    with pytest.raises(ValueError, match="apex corner 0 is not mapped"):
        check_degenerate_apex_features(topo, init)


def test_apex_not_at_end_of_traced_feature_raises():
    # the retry bug: corner 0 got a substitute vertex 11 next to the apex and the
    # seam was traced from it, so the real apex 7 ends no traced feature
    topo, init, _ = cone_loop(apex_corner_vid=11, seam=(11, 3, 4, 5))
    with pytest.raises(ValueError, match=r"apex corner 0 \(vid 7\) is not the end"):
        check_degenerate_apex_features(topo, init)


def test_degenerate_feature_ignored_while_real_features_remain():
    # mid-loop: the rim is not traced yet, so the apex is not checked (the seam
    # reaching it may come later in the loop)
    topo, init, polylines = cone_loop(apex_corner_vid=None)
    init.singularity_corner_mapping = {}
    init.loop_tracing_data.traced_id_set = {1}
    # with real features remaining the function goes on to inspect the mesh; give it one
    # mesh piece so it returns False without touching the apex
    from run_one_patch import construct_connectivity_data
    from utils.data_structures import PatchRegularSpaceData
    V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    conn = construct_connectivity_data(V, [[0, 1, 2], [0, 2, 3]])
    rsd = PatchRegularSpaceData(traced_edge_set={(0, 1)})
    assert detect_boundary_hug_after_halfedge(V, conn, rsd, polylines, {1}) is False


def test_hard_fallback_maps_apex_corner_to_stitched_vertex():
    from operations.trace import adopt_stitched_apex
    init = InitialLoopTracingData(initial_loop_id=0, singularity_corner_mapping={0: 7})
    init.loop_tracing_data.corner_id_to_vid = {0: 11, 1: 5}   # corner 0 got the stitched vertex 11
    adopt_stitched_apex(init, {0: 7})
    assert init.singularity_corner_mapping == {0: 11}
    # a trace that kept the apex changes nothing
    init2 = InitialLoopTracingData(initial_loop_id=0, singularity_corner_mapping={0: 7})
    init2.loop_tracing_data.corner_id_to_vid = {0: 7}
    adopt_stitched_apex(init2, {0: 7})
    assert init2.singularity_corner_mapping == {0: 7}
