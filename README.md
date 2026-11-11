# Topology-First B-Rep Meshing

[![CI](https://github.com/yunfanzhou/topology-first-brep-meshing/actions/workflows/ci.yml/badge.svg)](https://github.com/yunfanzhou/topology-first-brep-meshing/actions/workflows/ci.yml)

Triangle meshing of B-rep CAD models that preserves the model's topology: every B-rep face becomes a
patch of triangles, every B-rep edge a chain of mesh edges, and every B-rep vertex a mesh vertex.
The pipeline has two stages:

1. **Forward stage** (`main.py`, Python): meshes each face in its parameter domain, traces the face's
   boundary edges into the mesh, aligns the shared edges of adjacent faces, and fuses all patches into one
   conforming mesh (`fused.pkl`).
2. **Remeshing stage** (`remeshing/remesh.py`): remeshes the fused mesh to a target edge length with
   wildmeshing-toolkit's isotropic remeshing, in memory, keeping patches, feature chains and corners.

## Install

Python 3.11. The models are read with abs-hdf5, which has packages only for Python 3.9 to 3.11; the pipeline
uses its 0.2 API (`requirements.txt` pins 0.2.2), which abs-hdf5 1.0 changed (`part.Solid` became
`part.solids`).

```
pip install -r requirements.txt
```

This installs pygeodesic from a fork, https://github.com/yunfanzhou/pygeodesic (pinned commit), which adds
geodesic tracing between points on faces and edges. It is built from source, so a C++ compiler is needed.
The PyPI release of pygeodesic does not work with this pipeline; `main.py` stops with an ImportError if it
is installed.

The two C++ tools the pipeline calls also come from `requirements.txt`:

- Mmg 5.8.0 (forward stage) comes with the mmgpy package. By default `main.py` runs it in memory through
  mmgpy, in a helper process per worker, which kills a run longer than an hour (`mmg_timeout_seconds` in
  `utils/config.py`). `--mmg binary` runs its `mmg2d_O3` executable on files instead, the one in the
  mmgpy package or the build `--mmg2d_path` names; the meshes are the same, bit for bit.
- The isotropic remeshing (remeshing stage) is wildmeshing-toolkit's Python package `wildmeshing`, which
  pip builds from source (pinned commit). This needs git, a C++ compiler with C++20 library support (the
  Xcode command-line tools 16.3 or newer on macOS, g++ or clang on Linux, Visual Studio on Windows) and GMP
  on Linux (`apt install libgmp-dev`) and macOS (`brew install gmp`); on Windows the build downloads GMP.
  CMake comes from PyPI. The build downloads about 30 dependencies and takes several minutes (about 4 on a
  16-core laptop), with about 2.5 GB of memory per compile job; set `CMAKE_BUILD_PARALLEL_LEVEL` on smaller
  machines.

  The pinned commit builds with a limited range of compilers. On macOS:

  | Compiler | Result |
  |---|---|
  | Command-line tools 16.2 (Apple clang 16) | fails: `no member named 'atomic_ref' in namespace 'std'` |
  | Homebrew LLVM 19.1.7 | builds, with the command below |
  | Homebrew LLVM 23.1.5 | fails: `no template named 'vector' in namespace 'std'` (in a dependency) |

  With command-line tools older than 16.3, install LLVM 19 (`brew install llvm@19`; it is not put on the
  PATH and does not replace Apple's compiler) and build `wildmeshing` with it:

  ```
  L=$(brew --prefix llvm@19); SDK=$(xcrun --show-sdk-path)
  SDKROOT=$SDK CMAKE_ARGS="-DCMAKE_OSX_SYSROOT=$SDK -DCMAKE_CXX_SCAN_FOR_MODULES=OFF" \
  CC=$L/bin/clang CXX=$L/bin/clang++ LDFLAGS="-L$L/lib/c++ -Wl,-rpath,$L/lib/c++" \
  pip install "$(grep '^wildmeshing @' requirements.txt)"
  pip install -r requirements.txt
  ```

  (Without the two SDK settings, the build of the Python module stops with `'unistd.h' file not found`.)
  The first `pip install` builds only `wildmeshing`, the line of `requirements.txt` that names it; the
  second installs the rest with the default compiler.

## Run

Inputs are CAD models in the ABS HDF5 format ([abs-hdf5](https://pypi.org/project/abs-hdf5/)).
`data/simple/` has 15 small test models.

```
python main.py data/simple/Box.hdf5 out/Box --error_rel 0.05
python remeshing/remesh.py out/Box --len_rel 0.05
python check_topology.py data/simple/Box.hdf5 out/Box
```

`--error_rel` is the forward stage's target geometric error (default 1e-3) and `--len_rel` the remeshing
target edge length, both relative to the bounding box diagonal. They are different quantities and are set
independently: the forward stage has no edge length to match. Without `--len_rel`, the remeshing is given
no target and uses the average edge length of the fused mesh, so it keeps the forward mesh's resolution;
`--len_rel 0.05`, as in the example, coarsens it.

`remesh.py --cleanup_angle A` also repairs, after every remeshing iteration, the triangles whose smallest
angle is under A degrees (wildmeshing-toolkit's `cleanup_angle`). It is off by default: the repair is
serial and, while a fine forward mesh is coarsened, visits most of the mesh, which made the remeshing 20
to over 100 times slower on assemblies of 0.5 to 4 million forward triangles.

`main.py` meshes the faces in parallel, one process per CPU by default; `--workers N` sets the number of
processes (`--workers 1` runs serially). The output is the same for any number of workers.

`check_topology.py` checks that `fused.pkl` and, if present, `remeshed.vtu` match the B-rep's topology:
every B-rep vertex is one mesh vertex, every B-rep edge a chain of mesh edges between its two corners, and
every B-rep face a patch bordered by the chains of its own edges. It exits with status 1 on a mismatch.

## Output

The forward stage writes `out/<model>/fused.pkl`, a pickled tuple
`(V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids)`:

- `V`, `F`: vertices and triangles of the fused mesh
- `patch_to_fids`: B-rep face id to its triangle ids
- `feature_edge_id_to_vids`: B-rep edge id to its chain of vertex ids
- `corner_id_to_vids`: B-rep vertex id to its mesh vertex id

plus `result.json` (per-face status and timings). The remeshing stage adds:

- `remeshed.vtu`: the remeshed triangles, with cell data `patch_id` (B-rep face id) and point data
  `vertex_id` (B-rep vertex id of the corner vertices, -1 elsewhere)
- `remeshed_edges.vtu`: the same points and one line per edge of a feature chain, with cell data `edge_id`
  (B-rep edge id); zero-length B-rep edges are only their corner vertex
- `remeshed_report.json`: statistics of the remeshing

## Evaluation scripts

- `get_distance.py`: Hausdorff distance from points sampled on the CAD model to `remeshed.vtu`
- `get_metrics.py`: per-triangle quality (minimum angle, radius ratio)

For example, on the outputs of all models in `data/simple` (each in `out/<model>`):

```
python get_distance.py -i out --cad-root data/simple -o out/distance.csv
python get_metrics.py -i out -o out/metrics.csv
```

## Tests

```
pytest tests
```

CI (`.github/workflows/ci.yml`) runs the tests on Linux, macOS and Windows, and both stages, the topology
check and the evaluation scripts on every model in `data/simple`.

## License

MIT, see `LICENSE`. The C++ tools the pipeline calls are third-party software under their own licenses
(Mmg is LGPL-3.0); see `third_party/README.md`.
