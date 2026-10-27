"""Arguments whose defaults could not work are required, or rejected up front."""
import inspect

import pytest

from operations.embed_geodesic import embed_loops_one_patch
from operations.snapping import snapping_for_patch, solve_one_patch_linear_system
from run_one_patch import local_step_one_patch, solve_patch_linear_system


@pytest.mark.parametrize("func", [snapping_for_patch, solve_one_patch_linear_system,
                                  local_step_one_patch, solve_patch_linear_system])
def test_summary_is_required(func):
    # these write summary[...] unconditionally, so a None default could only crash
    assert inspect.signature(func).parameters["summary"].default is inspect.Parameter.empty


def test_embed_loops_rejects_zero_iterations():
    with pytest.raises(ValueError, match="max_iterations >= 1"):
        embed_loops_one_patch(None, None, None, None, None, {}, None, 1.0, max_iterations=0)
