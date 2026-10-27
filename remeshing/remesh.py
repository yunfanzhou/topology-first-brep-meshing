"""Remeshing stage: converts the forward stage's fused.pkl and runs the wildmeshing-toolkit remeshing app.

usage: python remeshing/remesh.py <output_folder> [--len_rel 1e-3] [--app PATH]

<output_folder> is the folder main.py wrote fused.pkl into. Writes mesh.msh, features.json and remesh.json
there (convert_pkl_to_wmtk.py), then remeshed.vtu and remeshed_edges.vtu (remeshing_app).
--len_rel is the target edge length relative to the bounding box diagonal; use the --error_rel of the
forward run. remeshing_app is built with the CMakeLists.txt at the repository root (see README); by
default it is looked up in build/bin.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILT_APP = os.path.join(HERE, "..", "build", "bin", "remeshing_app.exe" if os.name == "nt" else "remeshing_app")


def main():
    parser = argparse.ArgumentParser(description="Remesh the forward stage output with wildmeshing-toolkit.")
    parser.add_argument("output_folder", type=str, help="Folder containing fused.pkl from main.py")
    parser.add_argument("--len_rel", type=float, default=1e-3,
                        help="Target edge length relative to the bounding box diagonal (default: 1e-3, "
                             "the default --error_rel of main.py)")
    parser.add_argument("--app", type=str, default=None,
                        help="Path to remeshing_app (default: build/bin/remeshing_app, built with CMake)")
    args = parser.parse_args()

    app = args.app if args.app is not None else os.path.normpath(BUILT_APP)
    if not os.path.exists(app):
        sys.exit(f"remeshing_app not found at {app}; build it with "
                 "`cmake -S . -B build && cmake --build build --config Release` (see README) or pass --app.")
    out = os.path.abspath(args.output_folder)
    if not os.path.exists(os.path.join(out, "fused.pkl")):
        sys.exit(f"No fused.pkl in {out}; run main.py first.")

    subprocess.run([sys.executable, os.path.join(HERE, "convert_pkl_to_wmtk.py"), out,
                    "--len_rel", str(args.len_rel)], check=True)
    subprocess.run([app, "-j", os.path.join(out, "remesh.json")], check=True)
    print("remeshed mesh written to", os.path.join(out, "remeshed.vtu"))


if __name__ == "__main__":
    main()
