# Test models

35 models for testing the pipeline on more than the samples in `data/simple`, in the ABS HDF5 format.
`python run_models.py` (at the repository root) runs both stages and the topology check on all of them.

These files are not covered by this repository's MIT license: the ABC and Fusion 360 Gallery models
belong to their datasets and are under those datasets' terms.

## ABC dataset (27 models)

From the first chunk of the ABC dataset (Koch et al., *ABC: A Big CAD Model Dataset For Geometric Deep
Learning*, CVPR 2019), converted to ABS HDF5. The file name is the model id.

| File | ABC model | Size |
|---|---|---|
| `00003020.hdf5` | `00003020_623a6f12d0724d62a351e2a1_step_007` | 0.5 MB |
| `00004898.hdf5` | `00004898_29ce3eb9259f4becb9a5e526_step_144` | 0.7 MB |
| `00005122.hdf5` | `00005122_c4185a5498ce4a2f86e6f014_step_001` | 16.0 MB |
| `00007345.hdf5` | `00007345_a54f2a03e07d4a4497551c69_step_000` | 1.4 MB |
| `00008126.hdf5` | `00008126_5d720e0ec69f4a628d58af42_step_001` | 0.1 MB |
| `00009595.hdf5` | `00009595_2e821dba51fe4b79ba24bba7_step_000` | 0.3 MB |
| `00001105.hdf5` | `00001105_d1fef3edb4e34b88b70c6e85_step_000` | 4.2 MB |
| `00001269.hdf5` | `00001269_b53ece83d8964b44bbf1f8ed_step_005` | 2.2 MB |
| `00001630.hdf5` | `00001630_c1de5979807b4b6a8c5b0e49_step_001` | 0.4 MB |
| `00002712.hdf5` | `00002712_6aaf2a59f56d47e4a209d1b5_step_000` | 0.3 MB |
| `00004066.hdf5` | `00004066_410ebb8fc2b24e52babbd0e7_step_000` | 1.3 MB |
| `00004095.hdf5` | `00004095_682d0d68eb404db58e39f511_step_008` | 3.7 MB |
| `00005112.hdf5` | `00005112_a4d13a0ea546408bb7e08479_step_005` | 0.2 MB |
| `00005179.hdf5` | `00005179_e85b98fbe73143b9b7ce7b73_step_003` | 0.4 MB |
| `00005714.hdf5` | `00005714_cafe9c2a889d4198a6f9e326_step_012` | 2.2 MB |
| `00006220.hdf5` | `00006220_4eaf4323c5af4d758a782423_step_030` | 0.2 MB |
| `00006254.hdf5` | `00006254_2590200b97d0443d8bcdeee8_step_004` | 6.7 MB |
| `00006259.hdf5` | `00006259_67017e8898d4497fa2034765_step_000` | 4.7 MB |
| `00007076.hdf5` | `00007076_eb58cc5c6b8c44a2993e347e_step_002` | 7.8 MB |
| `00007494.hdf5` | `00007494_4fda1bdadb5d4d44a0276363_step_000` | 3.0 MB |
| `00007924.hdf5` | `00007924_614c0c7b35b74258ba545689_step_001` | 5.5 MB |
| `00008818.hdf5` | `00008818_3caba8b60eb44a53b3a3d8ea_step_002` | 8.3 MB |
| `00009002.hdf5` | `00009002_f85d4dda0c184332be3f2d6b_step_000` | 2.3 MB |
| `00009227.hdf5` | `00009227_f6e2761b81fd49c5be4b1b65_step_001` | 1.2 MB |
| `00009839.hdf5` | `00009839_e4e824f6666f4c1f8cb59a45_step_000` | 4.4 MB |
| `00009994.hdf5` | `00009994_ea83e2f413614058bcfa17e1_step_000` | 3.2 MB |
| `00003039.hdf5` | `00003039_623a6f12d0724d62a351e2a1_step_026` | 17.6 MB |

## Fusion 360 Gallery assembly dataset (4 models)

From the Fusion 360 Gallery Assembly dataset (Willis et al., *JoinABLe: Learning Bottom-up Assembly of
Parametric CAD Joints*, CVPR 2022), converted to ABS HDF5: the `assembly.hdf5` of each model.

| File | Size |
|---|---|
| `22162_439af02a.hdf5` | 10.0 MB |
| `22832_731d3b9f.hdf5` | 21.6 MB |
| `20451_557fe5cd.hdf5` | 3.0 MB |
| `20451_84d8b290.hdf5` | 2.9 MB |

## Toy models (4)

| File | Size |
|---|---|
| `Block_hole.hdf5` | 0.4 MB |
| `cusp.hdf5` | 0.1 MB |
| `Multiblock2.hdf5` | 0.3 MB |
| `SingleSolidSphere.hdf5` | 0.1 MB |
