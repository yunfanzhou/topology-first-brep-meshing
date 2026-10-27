from datetime import datetime
import os
import numpy as np
import utils.config as config
import json
def print_with_time(*args, **kwargs):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(now, *args, **kwargs)

def print_if_verbose(message):
    if config.verbose:
        print_with_time(message)

# Tracks (patch_id, tag) pairs already dumped, so an oversized patch is
# captured once per call-site rather than on every repeated call as it keeps
# growing (each dump is large; re-dumping would defeat "monitoring, not
# interfering").
_oversized_dumped_keys = set()


def dump_oversized_patch_state(V, F, tag="", threshold=None):
    """Dump mesh state when a patch grows too large.

    Pure monitoring: when ``len(V)`` reaches ``threshold`` (default
    ``config.oversized_vertex_threshold``), the raw ``V``/``F`` are written to a
    compressed ``.npz`` so a runaway patch can be investigated later (re-plot it
    offline). It never raises and never alters control flow — callers dump and
    continue; if the dump cannot be written, a message is printed instead. Each
    ``(patch_id, tag)`` is dumped (or attempted) at most once per process.

    Parameters
    ----------
    V : (nv, 3 or 2) vertex positions (list or ndarray).
    F : (nf, 3) triangle connectivity (list or ndarray).
    tag : short label identifying the call site (e.g. which regular-space pass).
    threshold : override for ``config.oversized_vertex_threshold``.

    Returns
    -------
    bool
        ``True`` if a dump was written on this call, ``False`` otherwise.
    """
    thr = config.oversized_vertex_threshold if threshold is None else threshold
    nv = len(V)
    if nv < thr:
        return False

    patch_id = getattr(config, "current_patch_id", None)
    key = (patch_id, tag)
    if key in _oversized_dumped_keys:
        return False
    _oversized_dumped_keys.add(key)

    out_dir = config.oversized_dump_dir
    try:
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"patch_{patch_id}_{tag}_{nv}v_{stamp}"
        npz_path = os.path.join(out_dir, base + ".npz")

        V_arr = np.asarray(V, dtype=float)
        F_arr = np.asarray(F, dtype=np.int64)
        np.savez_compressed(npz_path, V=V_arr, F=F_arr,
                            patch_id=str(patch_id), nv=nv, tag=str(tag))
    except Exception as e:
        print_with_time(
            f"[OVERSIZED PATCH] patch {patch_id} ({tag}) reached {nv} vertices "
            f">= threshold {thr}; could not dump state to {out_dir}: {e}")
        return False
    print_with_time(
        f"[OVERSIZED PATCH] patch {patch_id} ({tag}) reached {nv} vertices "
        f">= threshold {thr}; dumped state -> {npz_path}")
    return True


def classify_fail_reason(error_msg):
    """Classify an error message into a short failure reason tag."""
    msg = str(error_msg).lower()
    if "zero-area" in msg or "zero area" in msg:
        return "zero_area"
    if "non_manifold" in msg or "non-manifold" in msg or "not edge manifold" in msg:
        return "non_manifold"
    return "other"


class ResultLog:
    """Simple structured result log for the meshing pipeline.

    Output JSON::

        {
          "status": "running|success|failed",
          "runs": [{"started_at": ..., "finished_at": ..., "duration_s": ...}],
          "total_patches": N,
          "patches": {"id": {"duration_s": ..., "status": "success|failed", ...}},
          "failures": [{"patch_id": ..., "reason": ..., "detail": ..., "stage": ...}],
          "warnings": [...],
          "peak_memory_mb": ...
        }
    """

    def __init__(self, path, resume=False):
        self.path = path
        self.data = None
        if resume and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        if self.data is None:
            self.data = {
                "status": "running",
                "runs": [],
                "total_patches": 0,
                "patches": {},
                "failures": [],
                "warnings": [],
                "peak_memory_mb": 0.0,
            }

        for key, default in [("runs", []), ("patches", {}), ("failures", []),
                              ("warnings", []), ("peak_memory_mb", 0.0)]:
            self.data.setdefault(key, default)

        self._run_start = datetime.now()
        # Don't overwrite a completed model on resume
        if self.data.get("status") == "success" and resume:
            self._already_done = True
        else:
            self._already_done = False
            self.data["status"] = "running"
            self.data["runs"].append({"started_at": str(self._run_start)})
            self._save()

    def set_total_patches(self, n):
        self.data["total_patches"] = n
        self._save()

    def patch_done(self, patch_id, duration_s, attempts=1):
        self.data["patches"][str(patch_id)] = {
            "status": "success",
            "duration_s": round(duration_s, 3),
            "attempts": attempts,
        }
        self._save()

    def patch_failed(self, patch_id, duration_s, reason, detail="", attempts=1):
        self.data["patches"][str(patch_id)] = {
            "status": "failed",
            "duration_s": round(duration_s, 3),
            "fail_reason": reason,
            "fail_detail": detail,
            "attempts": attempts,
        }
        self.data["failures"].append({
            "patch_id": patch_id,
            "reason": reason,
            "detail": detail,
            "stage": "local",
            "time_stamp": str(datetime.now()),
        })
        self._save()

    def mark_failed(self, stage, exception):
        """Mark the run as failed and close it."""
        now = datetime.now()
        run = self.data["runs"][-1]
        if "finished_at" not in run:
            run["finished_at"] = str(now)
            run["duration_s"] = round((now - self._run_start).total_seconds(), 3)
        self.data["status"] = "failed"
        reason = "oom" if isinstance(exception, MemoryError) else classify_fail_reason(str(exception))
        self.data["fail_stage"] = stage
        self.data["fail_reason"] = reason
        self.data["fail_detail"] = str(exception)
        self._save()

    def add_warning(self, stage, detail, patch_id=None):
        entry = {"stage": stage, "detail": detail}
        if patch_id is not None:
            entry["patch_id"] = patch_id
        self.data["warnings"].append(entry)
        self._save()

    def finish(self):
        if self._already_done:
            return
        now = datetime.now()
        run = self.data["runs"][-1]
        run["finished_at"] = str(now)
        run["duration_s"] = round((now - self._run_start).total_seconds(), 3)
        self.data["status"] = "success"
        # Clear stale failure fields from previous runs
        self.data.pop("fail_stage", None)
        self.data.pop("fail_reason", None)
        self.data.pop("fail_detail", None)
        # Remove failure entries for patches that now succeeded
        succeeded = {pid for pid, info in self.data["patches"].items()
                     if info.get("status") == "success"}
        self.data["failures"] = [
            f for f in self.data["failures"]
            if str(f.get("patch_id", "")) not in succeeded
        ]
        self._save()

    def _update_memory(self):
        try:
            import resource
            import platform
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() == "Darwin":
                mem_mb = usage / (1024 * 1024)
            else:
                mem_mb = usage / 1024
            self.data["peak_memory_mb"] = round(
                max(self.data.get("peak_memory_mb", 0), mem_mb), 1)
        except Exception:
            pass

    def _save(self):
        # Write to a temp file in the same directory, then rename over the log, so an
        # interrupted write never leaves a truncated result.json.
        self._update_memory()
        tmp_path = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp_path, self.path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
