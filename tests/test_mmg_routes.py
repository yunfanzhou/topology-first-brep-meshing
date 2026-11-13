"""mmg_remesh's two routes: Mmg in memory (mmgpy, in a helper process) and the mmg2d executable."""
import os
import time

import numpy as np
import pytest

import operations.remesh_background as rb
import utils.config as config
from main import default_mmg2d_path

# Unit square, two triangles.
SQ_V = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
SQ_F = np.array([[0, 1, 2], [0, 2, 3]])
# The unit square fanned from (0.5, 0), with (0.25, 0) on its bottom edge: (0, 5, 1) has zero
# area, which mmg's file reader takes and its API refuses; (1, 3, 2) is clockwise.
FAN_V = np.array([[0, 0], [0.5, 0], [1, 0], [1, 1], [0, 1], [0.25, 0]], dtype=float)
FAN_F = np.array([[1, 3, 2], [1, 3, 4], [1, 4, 0], [0, 5, 1]])


def _metric(n, m11, m12, m22):
    return dict(EE=np.full(n, m11), FF_m=np.full(n, m12), GG=np.full(n, m22))


CASES = {
    # anisotropic metric, two feature-edge tags, a required edge
    "features": dict(V=SQ_V, F=SQ_F, **_metric(4, 400.0, 50.0, 100.0), metric_factor=0.5,
                     feature_edges={7: [(1, 2)], 9: [(2, 3), (4, 1)]}, required_edges=[(3, 4)]),
    # the options of the feature passes
    "no gradation": dict(V=SQ_V, F=SQ_F, **_metric(4, 900.0, 0.0, 900.0), hgrad=-1,
                         nosizreq=True, feature_edges={3: [(1, 2), (2, 3), (3, 4), (4, 1)]}),
    "zero area": dict(V=FAN_V, F=FAN_F, **_metric(6, 100.0, 0.0, 100.0),
                      feature_edges={7: [(1, 6), (6, 2), (2, 3)]}, required_edges=[(4, 5)]),
}


@pytest.mark.parametrize("case", CASES)
def test_routes_give_the_same_mesh(case):
    """In memory and as the executable, mmg returns the same mesh, bit for bit, at full
    precision (not the 15 digits of mmg's .mesh text output)."""
    mem = rb.mmg_remesh(**CASES[case])
    exe = rb.mmg_remesh(**CASES[case], mmg2d_path=default_mmg2d_path())
    for a, b in zip(mem[:2], exe[:2]):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    assert mem[2] == exe[2] and list(mem[2]) == list(exe[2])
    assert len(mem[1]) > 2 * len(CASES[case]["F"])
    assert any(float(f"{x:.15g}") != x for x in mem[0].ravel())


def test_memory_route_times_out(monkeypatch):
    """A run longer than config.mmg_timeout_seconds kills the helper process and raises the
    RuntimeError the callers handle; the next call starts a new helper."""
    rb.mmg_remesh(SQ_V, SQ_F, **_metric(4, 1.0, 0.0, 1.0))  # the helper is up
    monkeypatch.setattr(config, "mmg_timeout_seconds", 0.2)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="mmg2d timed out after 0.2 s"):
        rb.mmg_remesh(SQ_V, SQ_F, **_metric(4, 1e6, 0.0, 1e6))  # about 10^6 triangles
    assert time.monotonic() - t0 < 5.0
    assert rb._mmg_helper is None
    monkeypatch.undo()
    V, F, _ = rb.mmg_remesh(SQ_V, SQ_F, **_metric(4, 100.0, 0.0, 100.0))
    assert len(F) > 2


class _ExitWhenUnpickled:
    """Ends the process that unpickles it, as a crash in mmg would end the helper."""

    def __reduce__(self):
        return os._exit, (3,)


def test_memory_route_survives_a_crash():
    """The helper ending mid-request is a RuntimeError, and the next call starts a new one."""
    rb.mmg_remesh(SQ_V, SQ_F, **_metric(4, 1.0, 0.0, 1.0))
    with pytest.raises(RuntimeError, match=r"helper process ended \(exit code 3\)"):
        rb._mmg_in_helper({"crash": _ExitWhenUnpickled()})
    V, F, _ = rb.mmg_remesh(SQ_V, SQ_F, **_metric(4, 100.0, 0.0, 100.0))
    assert len(F) > 2
