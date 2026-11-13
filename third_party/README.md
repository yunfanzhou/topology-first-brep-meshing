# Third-party software

The pipeline calls two C++ tools, installed by `requirements.txt` and kept under their own licenses:

- Mmg 5.8.0 (forward stage; LGPL-3.0, https://github.com/MmgTools/mmg), from the mmgpy package: its library
  through mmgpy, or with `--mmg binary` its `mmg2d_O3` executable.
- Isotropic remeshing (remeshing stage): wildmeshing-toolkit (MIT,
  https://github.com/wildmeshing/wildmeshing-toolkit), from its Python package `wildmeshing`, built from
  source; it downloads its dependencies, each under its own license.
