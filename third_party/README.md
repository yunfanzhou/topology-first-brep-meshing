# Third-party software

`remeshing/convert_pkl_to_wmtk.py` is copied from wildmeshing-toolkit (MIT, `wildmeshing-toolkit.LICENSE`).

The two C++ tools the pipeline calls are not part of this repository and keep their own licenses:

- `mmg2d_O3` (forward stage) is Mmg 5.8.0 (LGPL-3.0, https://github.com/MmgTools/mmg), installed with the
  mmgpy package from `requirements.txt`.
- `remeshing_app` (remeshing stage) is built from source by `CMakeLists.txt` from wildmeshing-toolkit, branch
  `fpir` (MIT, https://github.com/wildmeshing/wildmeshing-toolkit), which downloads its dependencies, each under
  its own license.
