#!/usr/bin/env python3
"""Parse result.json files from a batch run folder.

Usage:
    # Print summary of all models (success/fail counts, failure details)
    python parse_results.py /path/to/results

    # Export per-model timing CSV for successful runs
    python parse_results.py /path/to/results --csv timing.csv
"""

import argparse
import csv
import json
import os
import sys


def load_result(folder):
    """Load result.json from a model folder. Returns (name, data) or (name, None)."""
    path = os.path.join(folder, "result.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
def check_topo(folder):
    """topo_ok of <folder>/eval.json, which the authors' batch evaluation writes (no script in this release
    does); None when the file is missing."""
    path = os.path.join(folder, "eval.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # topology failure
    if not data.get("topo_ok", False):
        return False
    return True

def total_duration(data):
    """Sum duration_s across all runs."""
    return sum(r.get("duration_s", 0) for r in data.get("runs", []))


def print_summary(results, list_path=None, zero_path = None):
    """Print a human-readable summary table."""
    success = []
    not_remeshed = []
    failed = []
    running = []
    for name, data in sorted(results.items()):
        status = data.get("status", "unknown")
        if status == "success":
            success.append(name)
        elif status == "not_remeshed":
            not_remeshed.append(name)
        elif status == "failed":
            failed.append(name)
        else:
            running.append(name)

    total = len(results)
    print(f"Total models: {total}")
    print(f"  Success: {len(success)}")
    if not_remeshed:
        print(f"  Forward stage only (not remeshed): {len(not_remeshed)}")
    print(f"  Failed:  {len(failed)}")
    print(f"  Running: {len(running)}")
    print()
    if list_path is not None:
        # Write failed and running model names, one per line.
        models_to_write = failed + running
        with open(list_path, "w") as f:
            if models_to_write:
                f.write("\n".join(models_to_write) + "\n")
        print(f"Wrote {len(models_to_write)} model names to {list_path}")

    zeros = []
    if failed:
        print("=" * 80)
        print("FAILURES")
        print("=" * 80)
        for name in failed:
            data = results[name]
            stage = data.get("fail_stage", "?")
            reason = data.get("fail_reason", "?")
            detail = data.get("fail_detail", "")
            duration = total_duration(data)
            patches = data.get("total_patches", "?")
            mem = data.get("peak_memory_mb", 0)

            if reason == "zero_area":
                zeros.append(name)

            print(f"\n  {name}")
            print(f"    stage:   {stage}")
            print(f"    reason:  {reason}")
            print(f"    patches: {patches}, time: {duration:.1f}s, memory: {mem:.0f} MB")
            if detail:
                # Truncate long details
                short = detail if len(detail) < 120 else detail[:117] + "..."
                print(f"    detail:  {short}")

            # Per-patch failures
            for f in data.get("failures", []):
                pid = f.get("patch_id", "?")
                fr = f.get("reason", "?")
                print(f"    - patch {pid}: {fr}")
        print()

    if running:
        print("=" * 80)
        print("STILL RUNNING / INTERRUPTED")
        print("=" * 80)
        for name in running:
            data = results[name]
            patches = data.get("patches", {})
            done = sum(1 for p in patches.values() if p.get("status") == "success")
            total_p = data.get("total_patches", "?")
            print(f"  {name}: {done}/{total_p} patches done")
        print()

    if zero_path is not None:
        # Write failed and running model names, one per line.
        zero_models_to_write = zeros
        with open(zero_path, "w") as f:
            if zero_models_to_write:
                f.write("\n".join(zero_models_to_write) + "\n")
        print(f"Wrote {len(zero_models_to_write)} model names to {zero_path}")



def write_timing_csv(results, csv_path):
    """Write per-model timing CSV for successful runs."""
    rows = []
    for name, data in sorted(results.items()):
        if data.get("status") not in ("success", "not_remeshed"):  # timings are of the forward stage
            continue

        patches = data.get("patches", {})
        n_patches = data.get("total_patches", len(patches))
        duration = total_duration(data)
        n_runs = len(data.get("runs", []))
        mem = data.get("peak_memory_mb", 0)

        patch_times = [p.get("duration_s", 0) for p in patches.values()]
        max_patch = max(patch_times) if patch_times else 0
        avg_patch = sum(patch_times) / len(patch_times) if patch_times else 0

        rows.append({
            "model": name,
            "total_patches": n_patches,
            "total_duration_s": round(duration, 3),
            "num_runs": n_runs,
            "avg_patch_s": round(avg_patch, 3),
            "max_patch_s": round(max_patch, 3),
            "peak_memory_mb": round(mem, 1),
        })

    if not rows:
        print("No successful models to write.")
        return

    fields = list(rows[0].keys())
    if os.path.exists(csv_path):
        print(f"Warning: {csv_path} already exists and will be overwritten.")
    else:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse meshing result.json files.")
    parser.add_argument("results_folder", help="Root folder containing per-model subfolders")
    parser.add_argument(
        "--list",
        default=None,
        help="Output .txt path for failed and running model names, one per line",
    )
    parser.add_argument("--zero_area", default= None, help="Output .txt path for models with zero areas")
    parser.add_argument("--csv", default=None, help="Output CSV path for timing of successful models")
    args = parser.parse_args()

    results = {}
    bad_topo = []
    for entry in os.listdir(args.results_folder):
        folder = os.path.join(args.results_folder, entry)
        if not os.path.isdir(folder):
            continue
        data = load_result(folder)
        if data is not None:
            if data.get("status") == "success" and not os.path.exists(
                os.path.join(folder, "remeshed.vtu")
            ):
                data["status"] = "not_remeshed"
            results[entry] = data
            ok = check_topo(folder)
            if ok is None:
                pass
            elif not ok:
                print(f"Topology failure detected in {entry}")
                bad_topo.append(entry)


    if not results:
        print(f"No result.json files found in {args.results_folder}")
        sys.exit(1)

    print(f"Parsed {len(results)} result.json files. ")
    print(f"Bad topology models: {', '.join(bad_topo)}")
    print_summary(results, args.list, args.zero_area)
    

    if args.csv:
        write_timing_csv(results, args.csv)


if __name__ == "__main__":
    main()
