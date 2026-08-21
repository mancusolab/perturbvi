---
hide:
  - toc
---

# Single-cell perturbation analysis

PerturbVI learns latent gene programs and perturbation effects from
single-cell perturbation screens.

## Before you start

PerturbVI relates gene expression to the perturbations applied in a screen. It
uses two matrices containing the same cells in the same order:

- `X` contains one expression profile per cell, with genes in columns.
- `G` records the perturbation design, with perturbations in columns. Control
  cells have all-zero rows, single-perturbation cells have one active column,
  and combination screens may have several.

For H5AD data, `load_screen` can construct `G` from a perturbation label in
`obs` or read a prepared matrix from `obsm`. A control label is required only
when `G` is constructed from labels. For 10x H5 or MEX data, the loader reads
gene expression and aligns it with a barcode-indexed table of prepared
perturbation assignments. Covariates are optional and are needed only when
residualization is requested.

## CLI

```bash
perturbvi fit screen.h5ad \
  --guide-key perturbation \
  --control-label control \
  --covariates batch log_total_counts percent_mito \
  --categorical-covariates batch \
  --output results \
  --z-dim 12 --l-dim 400 --tau 50
```

Current 10x H5 and MEX inputs use a barcode-indexed metadata table:

```bash
perturbvi fit filtered_feature_bc_matrix.h5 \
  --format 10x-h5 \
  --metadata cell_metadata.tsv \
  --guide-key perturbation \
  --control-label control \
  --covariates batch log_total_counts percent_mito \
  --categorical-covariates batch \
  --output results \
  --z-dim 12 --l-dim 400 --tau 50
```

```bash
perturbvi analyze results --output results/analysis
```

Residualization is optional. When it is combined with standardization,
PerturbVI residualizes first and standardizes the residuals. Skip
residualization for expression values that have already been residualized.

## Python

```python
from perturbvi import analyze, fit_screen, load_screen, residualize_screen

screen = load_screen(
    "screen.h5ad",
    guide_key="perturbation",
    control_label="control",
    covariates=["batch", "log_total_counts", "percent_mito"],
)
screen = residualize_screen(screen, categorical_covariates=["batch"])
result = fit_screen(screen, z_dim=12, l_dim=400, tau=50, seed=1)
tables = analyze(
    result,
    gene_names=screen.gene_names,
    perturbation_names=screen.perturbation_names,
)
```

## Formats

| Input | CLI | Python |
|---|---:|---:|
| H5AD | yes | yes |
| AnnData object | no | yes |
| CSV/TSV | yes | yes |
| Current 10x H5/MEX | yes | yes |

CSV and TSV are intended for small or medium dense matrices. Zarr is not a
first-class input format; save AnnData as H5AD before loading it.

Fit and analysis filenames are listed in the
[API reference](api.md#saved-files).

See the [cookbook](cookbook.md) for complete analyses and the [API
reference](api.md) for functions and saved outputs.
