[![Documentation-webpage](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/perturbvi/)
[![PyPI-Server](https://img.shields.io/pypi/v/perturbvi.svg)](https://pypi.org/project/perturbvi/)
[![Github](https://img.shields.io/github/stars/mancusolab/perturbvi?style=social)](https://github.com/mancusolab/perturbvi)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project generated with Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

# perturbVI
`perturbvi` is a scalable variational inference method for learning regulatory modules from single-cell
Perturb-seq data. It fits a sparse latent factor model that jointly infers gene programs (factors) and
the perturbation effects that drive them.

  [**Installation**](#installation)
  | [**Example**](#current-python-api)
  | [**Notes**](#notes)
  | [**Support**](#support)
  | [**Other Software**](#other-software)

------------------

## Installation

``` bash
pip install perturbvi
```

Or from source:

``` bash
git clone https://github.com/mancusolab/perturbvi.git
cd perturbvi
pip install .
```

## CLI

```bash
# Fit from AnnData
perturbvi fit screen.h5ad \
  --output results \
  --z-dim 12 --l-dim 400 --tau 800 \
  --guide-key perturbation \
  --control-label non-targeting

# with covariate residualization (h5ad only)
perturbvi fit screen.h5ad \
  --output results \
  --z-dim 12 --l-dim 400 --tau 800 \
  --guide-key perturbation \
  --covariates batch percent_mito n_counts \
  --categoricals batch

# Fit from 10x feature-barcode H5
perturbvi fit filtered_feature_bc_matrix.h5 \
  --format 10x-h5 \
  --output results \
  --z-dim 12 --l-dim 400 --tau 800

# Analyze saved results
perturbvi analyze results \
  --gene-names genes.csv \
  --perturbation-names perturbations.csv \
  --output results/analysis

# Analyze with LFSR (expensive)
perturbvi analyze results \
  --gene-names genes.csv \
  --perturbation-names perturbations.csv \
  --compute-lfsr --lfsr-iters 2000
```

## Python API

The workflow always has 3 stages, run in order: **fit → save → analyze**.
`gene_names` / `perturbation_names` are optional labels for the output tables — pass `None`
(or omit them) to get integer index labels instead.

### 1. Fit

```python
from perturbvi import load_screen, residualize_screen, fit_screen

screen = load_screen(
    "screen.h5ad",
    guide_key="perturbation",
    control_label="non-targeting",
    covariates=["batch", "percent_mito", "n_counts"],  # optional
)

# Optional — skip this line entirely if your expression is already clean.
# Residualization happens first; standardization (in fit_screen) applies after.
screen = residualize_screen(screen, categoricals=["batch"])

fitted = fit_screen(screen, z_dim=12, l_dim=400, tau=800)
# `fitted` is an InferResults object (params, elbo, pve, pip)
```

### 2. Save

```python
from perturbvi import save_results

save_results(fitted, path="results/")
```

### 3. Analyze

```python
from perturbvi import analyze

results = analyze(
    fitted,
    gene_names=screen.gene_names,                  # optional
    perturbation_names=screen.perturbation_names,  # optional
    compute_lfsr=False,                            # optional, default: False
)
```

`results` is a dict of DataFrames: `pip_df`, `pve_df`, `beta_df`, `p_hat_df`, `overall_effect_df`.

Analyzing results saved in an earlier session, without refitting — load, then analyze:

```python
from perturbvi import load_results, analyze

fitted = load_results("results/")
results = analyze(
    fitted,
    gene_names=screen.gene_names,                  # optional
    perturbation_names=screen.perturbation_names,  # optional
    compute_lfsr=False,                            # optional, default: False
)
```

> [!IMPORTANT]
> `compute_lfsr=True` triggers an expensive Monte Carlo computation (`lfsr_iters`, default
> `2000`). It only runs when explicitly requested. Turn it on to also get `results["lfsr_df"]`:
> ```python
> results = analyze(fitted, compute_lfsr=True, lfsr_iters=2000)
> ```

> [!WARNING]
> `gene_names`/`perturbation_names`, if given, must have the same length as the corresponding
> dimension in `fitted` — there's currently no friendly error message for a mismatch, just a
> raw pandas `ValueError`.

Low-level array API (unchanged):

```python
from perturbvi import infer
results = infer(X, z_dim=12, l_dim=400, G=G, tau=800)
```

## Supported Input Formats

### `.h5ad` (AnnData)

Guides come from an `obs` column (`--guide-key`) or an existing `obsm` matrix (`--guide-obsm`):

```bash
perturbvi fit screen.h5ad \
  --output results \
  --z-dim 12 --l-dim 400 --tau 800 \
  --guide-key perturbation
```

### `10x-h5` (10x feature-barcode HDF5)

Guides are extracted from the file by feature type:

```bash
perturbvi fit filtered_feature_bc_matrix.h5 \
  --format 10x-h5 \
  --output results \
  --z-dim 12 --l-dim 400 --tau 800 \
  --expression-feature-type "Gene Expression" \
  --guide-feature-type "CRISPR Guide Capture"
```

### `10x-mex` (10x MEX directory)

Directory must contain `matrix.mtx`, `features.tsv`, and `barcodes.tsv`:

```bash
perturbvi fit path/to/mex_dir/ \
  --format 10x-mex \
  --output results \
  --z-dim 12 --l-dim 400 --tau 800 \
  --expression-feature-type "Gene Expression" \
  --guide-feature-type "CRISPR Guide Capture"
```

### CSV/TSV

No CLI support yet — use the low-level `infer()` API directly. `G` can be dense or sparse
(`jax.experimental.sparse`); drop any non-targeting/control columns from the guide matrix
before converting it. Expression is expected to already be residualized/QC'd — there's no
`residualize_screen` step in this path, unlike `.h5ad`.

```python
import jax.numpy as jnp
import pandas as pd
from jax.experimental import sparse
from perturbvi import infer, save_results, analyze

# Expression: cells x genes. Assumed already residualized/preprocessed.
X = jnp.asarray(pd.read_csv("expression.csv", index_col=0), dtype=jnp.float64)

# Guides: cells x perturbations. Drop non-targeting/control columns first.
guide = pd.read_csv("guides.csv", index_col=0)
guide = guide.drop(columns=["non-targeting"], errors="ignore")
G = sparse.bcoo_fromdense(jnp.asarray(guide, dtype=jnp.float64))

fitted = infer(
    X, z_dim=12, l_dim=500, G=G,
    p_prior=0.1, standardize=True, tau=50,
    init="pca", max_iter=1000, tol=1e-2,
)

save_results(fitted, path="results/")
results = analyze(fitted, perturbation_names=guide.columns.tolist())
```

> [!NOTE]
> Zarr is not yet first-class. Covariate residualization for 10x formats is planned; use the
> Python API to construct `ScreenData.covariates` manually for now.

## Notes

-   `perturbvi` uses [JAX](https://github.com/google/jax) with [Just In
    Time](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
    compilation to achieve high-speed computation. However, there are
    some [issues](https://github.com/google/jax/issues/5501) for JAX
    with Mac M1 chip. To solve this, users need to initiate conda using
    [miniforge](https://github.com/conda-forge/miniforge), and then
    install `perturbvi` using `pip` in the desired environment.

## Support

Please report any bugs or feature requests in the [Issue
Tracker](https://github.com/mancusolab/perturbvi/issues). If users have
any questions or comments, please contact Dong Yuan (<dongyuan@usc.edu>)
and Nicholas Mancuso (<nmancuso@usc.edu>).

## Other Software

Feel free to use other software developed by [Mancuso
Lab](https://www.mancusolab.com/):

-   [SuShiE](https://github.com/mancusolab/sushie): a Bayesian
    fine-mapping framework for molecular QTL data across multiple
    ancestries.
-   [MA-FOCUS](https://github.com/mancusolab/ma-focus): a Bayesian
    fine-mapping framework using
    [TWAS](https://www.nature.com/articles/ng.3506) statistics across
    multiple ancestries to identify the causal genes for complex traits.
-   [SuSiE-PCA](https://github.com/mancusolab/susiepca): a scalable
    Bayesian variable selection technique for sparse principal component
    analysis
-   [twas_sim](https://github.com/mancusolab/twas_sim): a Python
    software to simulate [TWAS](https://www.nature.com/articles/ng.3506)
    statistics.
-   [FactorGo](https://github.com/mancusolab/factorgo): a scalable
    variational factor analysis model that learns pleiotropic factors
    from GWAS summary statistics.
-   [HAMSTA](https://github.com/tszfungc/hamsta): a Python software to
    estimate heritability explained by local ancestry data from
    admixture mapping summary statistics.

------------------------------------------------------------------------

`perturbvi` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.


------------------------------------------------------------------------

This project has been set up using Hatch. For details and usage
information on Hatch see <https://github.com/pypa/hatch>.
