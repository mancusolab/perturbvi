[![Documentation-webpage](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/perturbvi/)
[![PyPI-Server](https://img.shields.io/pypi/v/perturbvi.svg)](https://pypi.org/project/perturbvi/)
[![Github](https://img.shields.io/github/stars/mancusolab/perturbvi?style=social)](https://github.com/mancusolab/perturbvi)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project generated with Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

# perturbVI
`perturbvi` is a scalable approach to infer regulatory modules through informative latent component model in the single-cell Perturb-seq data.

  [**Installation**](#installation)
  | [**Example**](#cli)
  | [**Notes**](#notes)
  | [**Support**](#support)
  | [**Other Software**](#other-software)

------------------

## Installation

``` bash
# install perturbvi
uv pip install perturbvi

# help
perturbvi --help
```


## CLI

Fit from an AnnData `.h5ad` file:

```bash
perturbvi fit screen.h5ad \
  --output results \
  --z-dim 12 \
  --l-dim 400 \
  --tau 800 \
  --guide-key perturbation \
  --control-label non-targeting \
  --p-prior 0.1 \
  --seed 1

# with covariate residualization
perturbvi fit screen.h5ad \
  --output results \
  --z-dim 12 --l-dim 400 --tau 800 \
  --guide-key perturbation \
  --covariates batch percent_mito n_counts \
  --categoricals batch
```

Fit from a 10x feature-barcode H5 file:

```bash
perturbvi fit filtered_feature_bc_matrix.h5 \
  --format 10x-h5 \
  --output results \
  --z-dim 12 \
  --l-dim 400 \
  --tau 800
```

Analyze saved results:

```bash
perturbvi analyze results \
  --genes genes.csv \
  --perturbations perturbations.csv \
  --output results/analysis

# add --compute-lfsr to run the (expensive) LFSR step
perturbvi analyze results \
  --genes genes.csv \
  --perturbations perturbations.csv \
  --compute-lfsr \
  --lfsr-iters 2000
```

## Python API

Workflow: **fit → analyze → save**. `genes` and `perturbations` are optional labels.
Omit them to get integer index labels instead.

```python
from perturbvi import load_screen, residualize_screen, fit_screen, analyze, save_results

screen = load_screen("screen.h5ad", guide_key="perturbation", control_label="non-targeting")
screen = residualize_screen(screen, categoricals=["batch"])  # optional, skip if already clean

fitted = fit_screen(screen, z_dim=12, l_dim=400, tau=800)  # InferResults: params, elbo, pve, pip

save_results(fitted, path="results")  # W.txt, pip.txt, pve.txt, params_file.pkl

results = analyze(
    fitted,
    genes=screen.genes,
    perturbations=screen.perturbations,
    compute_lfsr=True,
    path="results",  # also writes results/lfsr.csv
)
# dict of DataFrames: pip_df, pve_df, beta_df, p_hat_df, overall_effect_df, lfsr
```

> [!IMPORTANT]
> `compute_lfsr` runs an expensive Monte Carlo step (`lfsr_iters`, default `2000`).
> Call it when you need it.

To reopen later, without refitting or recomputing LFSR:

```python
from perturbvi import load_results, analyze

fitted = load_results("results")
results = analyze(fitted, path="results")  # reuses results/lfsr.csv if present, else lfsr is None
```

There is also a low-level `infer()` API. See [CSV/TSV](#csvtsv) below for a full example.

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

No CLI support yet. Use the low-level `infer()` API directly. `G` can be dense or sparse
(`jax.experimental.sparse`). Drop non-targeting or control columns from the guide matrix
before converting it. Expression should already be residualized and QC'd. Unlike `.h5ad`,
this path has no `residualize_screen` step.

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

save_results(fitted, path="results")
results = analyze(fitted, perturbations=guide.columns.tolist())
```

> [!NOTE]
> Zarr is not yet first-class. Covariate residualization also works for 10x formats: pass
> `--covariate-file` (a barcode-indexed CSV/TSV) alongside `--covariates`.


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
-   [jaxQTL](https://github.com/mancusolab/jaxqtl): a scalable software 
    for large-scale eQTL mapping using count-based models.
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
-   [traceax](https://github.com/mancusolab/traceax): a Python
    software to perform stochastic trace estimation for linear operators.
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
