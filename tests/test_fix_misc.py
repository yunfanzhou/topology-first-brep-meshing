"""Regression tests for the result log and the atomic save."""
import json
import sys

import pytest

from utils.print import ResultLog


@pytest.mark.skipif(sys.platform == "win32", reason="no resource module on Windows")
def test_peak_memory_counts_finished_workers(tmp_path, monkeypatch):
    import resource

    class Usage:
        def __init__(self, maxrss):
            self.ru_maxrss = maxrss

    # ru_maxrss is in bytes on macOS, in KiB on Linux. The parent peaks at 100 MiB, a finished worker at 5000.
    mib = 1024 * 1024 if sys.platform == "darwin" else 1024
    sizes = {resource.RUSAGE_SELF: 100 * mib, resource.RUSAGE_CHILDREN: 5000 * mib}
    monkeypatch.setattr(resource, "getrusage", lambda who: Usage(sizes[who]))
    log = ResultLog(str(tmp_path / "result.json"))
    log.finish()
    with open(tmp_path / "result.json") as f:
        assert json.load(f)["peak_memory_mb"] == 5000.0


def test_atomic_save_removes_its_temp_file_on_failure(tmp_path):
    import os
    from cad_io.write import atomic_save

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("cannot pickle")

    with pytest.raises(RuntimeError):
        atomic_save(Unpicklable(), str(tmp_path / "fused.pkl"))
    assert os.listdir(tmp_path) == []
