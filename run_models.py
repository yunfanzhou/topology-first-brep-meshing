"""Run both stages and the topology check on every model of a folder.

usage: python run_models.py [<models folder>] [<output folder>] [--error_rel E] [--len_rel L]

For each <models>/<name>.hdf5: main.py (forward stage), remeshing/remesh.py and check_topology.py, with
the outputs in <output>/<name>. Defaults: data/models, out_models, and the defaults of main.py and
remesh.py (--error_rel 1e-3, --len_rel 0.05). The models run one after the other; main.py already
uses every CPU. Prints one line per model and exits with 1 if any model fails.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def stage(cmd, log):
    t0 = time.time()
    with open(log, "w") as fh:
        ok = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode == 0
    return ok, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="Run both stages and the topology check on a folder of models.")
    parser.add_argument("models", nargs="?", default=os.path.join(HERE, "data", "models"))
    parser.add_argument("output", nargs="?", default="out_models")
    parser.add_argument("--error_rel", type=float, default=None, help="Passed to main.py (default: its own)")
    parser.add_argument("--len_rel", type=float, default=None, help="Passed to remeshing/remesh.py (default: its own)")
    args = parser.parse_args()

    inputs = sorted(glob.glob(os.path.join(args.models, "*.hdf5")))
    if not inputs:
        raise SystemExit(f"No .hdf5 models in {args.models}")
    failed = []
    for inp in inputs:
        name = os.path.splitext(os.path.basename(inp))[0]
        out = os.path.abspath(os.path.join(args.output, name))
        os.makedirs(out, exist_ok=True)
        steps = [("forward", [sys.executable, "main.py", os.path.abspath(inp), out]
                  + (["--error_rel", str(args.error_rel)] if args.error_rel is not None else [])),
                 ("remesh", [sys.executable, "remeshing/remesh.py", out]
                  + (["--len_rel", str(args.len_rel)] if args.len_rel is not None else [])),
                 ("topology", [sys.executable, "check_topology.py", os.path.abspath(inp), out])]
        times, bad = [], None
        for step, cmd in steps:
            ok, seconds = stage(cmd, os.path.join(out, f"{step}.log"))
            times.append(f"{step} {seconds:.1f} s")
            if not ok:
                bad = step
                break
        print(f"{'ok  ' if bad is None else 'FAIL'} {name}: {', '.join(times)}"
              + ("" if bad is None else f"  (see {os.path.join(out, bad + '.log')})"), flush=True)
        if bad is not None:
            failed.append(name)
    print(f"{len(inputs) - len(failed)} of {len(inputs)} models pass" + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
