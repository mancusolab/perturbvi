[![Documentation-webpage](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/perturbvi/)
[![PyPI-Server](https://img.shields.io/pypi/v/perturbvi.svg)](https://pypi.org/project/perturbvi/)
[![Github](https://img.shields.io/github/stars/mancusolab/perturbvi?style=social)](https://github.com/mancusolab/perturbvi)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project generated with Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

# perturbVI
`perturbvi` is a scalable approach to infer regulatory modules through informative latent component model in the single-cell Perturb-seq data.

  [**Installation**](#installation)
  | [**Example**](#get-started-with-example)
  | [**Notes**](#notes)
  | [**Version**](#version-history)
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
  --gene-names genes.csv \
  --perturbation-names perturbations.csv \
  --output results/analysis

# add --compute-lfsr to run the (expensive) LFSR step
perturbvi analyze results \
  --gene-names genes.csv \
  --perturbation-names perturbations.csv \
  --compute-lfsr \
  --lfsr-iters 2000
```

## Python API

The workflow always has 3 stages, run in order: **fit → save → analyze**.
`gene_names` / `perturbation_names` are optional labels for the output tables — pass `None` (or omit them) to get integer index labels instead.

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
screen = residualize_screen(screen, categoricals=["batch"])

fitted = fit_screen(screen, z_dim=12, l_dim=400, tau=800)
# `fitted` is an InferResults object (params, elbo, pve, pip)
```

### 2. Save

```python
from perturbvi import save_results

save_results(fitted, path="results/")
# writes: W.txt, pip.txt, pve.txt, params_file.pkl
```

### 3. Analyze

Two ways to get to the same analysis tables, depending on whether you still have
`fitted` in memory:

| You have...                                          | Call                                     |
|--------------------------------------------------------|---------------------------------------------|
| `fitted` in memory (just fit it)                        | `analyze(fitted, ...)`                       |
| only a `results/` directory (new session, no refit)      | `analyze_saved("results/", ...)`             |

Both accept the *same* optional keyword arguments and return the *same* dict of tables — the
only difference is what you pass in first (an object vs. a path).

```python
from perturbvi import analyze

results = analyze(
    fitted,
    gene_names=screen.gene_names,          # optional
    perturbation_names=screen.perturbation_names,  # optional
)
```

```python
# Equivalent, but starting fresh from disk instead of the in-memory `fitted`
# (e.g. in a new script the next day):
from perturbvi import analyze_saved

results = analyze_saved(
    "results/",
    gene_names=screen.gene_names,          # optional
    perturbation_names=screen.perturbation_names,  # optional
)
```

`results` is a dict of DataFrames: `pip_df`, `pve_df`, `beta_df`, `p_hat_df`, `overall_effect_df`.

**LFSR** is a separate, expensive computation and is *off by default* for both functions above.
Turn it on explicitly, and check `results["lfsr_df"]` only after doing so:

```python
results = analyze(fitted, compute_lfsr=True, lfsr_iters=2000)
# now results also has "lfsr_df"
```

> **Note:** `gene_names`/`perturbation_names`, if given, must have the same length as the
> corresponding dimension in `fitted` — there's currently no friendly error message for a
> mismatch, just a raw pandas `ValueError`.

Low-level array API (unchanged):

```python
from perturbvi import infer, save_results

results = infer(X, z_dim=12, l_dim=400, G=G, tau=800)
save_results(results, path="results/")
```

## Supported Input Formats

| Format | Description |
|---|---|
| `.h5ad` | AnnData file. Use `--guide-key` (obs column) or `--guide-obsm` (obsm key) to specify guides. |
| `10x-h5` | 10x feature-barcode HDF5 file. Guides extracted by feature type. |
| `10x-mex` | 10x MEX directory (`matrix.mtx`, `features.tsv`, `barcodes.tsv`). |
| CSV/TSV | Use the low-level `infer()` API directly for small files and tests. |

Zarr is not yet first-class. Covariate residualization for 10x formats is planned; use the Python API to construct `ScreenData.covariates` manually for now.


## Notes

-   `perturbvi` uses [JAX](https://github.com/google/jax) with [Just In
    Time](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
    compilation to achieve high-speed computation. However, there are
    some [issues](https://github.com/google/jax/issues/5501) for JAX
    with Mac M1 chip. To solve this, users need to initiate conda using
    [miniforge](https://github.com/conda-forge/miniforge), and then
    install `perturbvi` using `pip` in the desired environment.

## Version History

TBD

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
