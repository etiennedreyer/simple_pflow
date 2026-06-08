[![DOI](https://zenodo.org/badge/1188512931.svg)](https://doi.org/10.5281/zenodo.20588829)

## PyCaloFlash

A pythonic, GPU-accelerated implementation of the CaloFlash fast EM calorimeter shower model from [Grindhammer & Peters (2000)](https://arxiv.org/abs/hep-ex/0001020v1).

### Quick Start

```bash
pip install -e .
```

### Branches:

- **`b_torch` (default branch)**: PyTorch implementation, including dataset for training models
- **`b_jax` branch**: JAX implementation

### Usage

See the `notebooks/` directory for examples:
- `calo_flash_demo.ipynb` — Directly interfacing the shower model
- `dataset_demo.ipynb` — Generating datasets
