# Third-party software

The pipeline calls two C++ tools, installed by `requirements.txt` and kept under their own licenses:

- `mmg2d_O3` (forward stage): Mmg 5.8.0 (LGPL-3.0, https://github.com/MmgTools/mmg), from the mmgpy package.
- Isotropic remeshing (remeshing stage): wildmeshing-toolkit (MIT,
  https://github.com/wildmeshing/wildmeshing-toolkit), from its Python package `wildmeshing`, built from
  source; it downloads its dependencies, each under its own license.
