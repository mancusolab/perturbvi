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
  | [**Version**](#version-history)
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

```python
from perturbvi import load_screen, residualize_screen, fit_screen, save_results, analyze_results

# Load from file
screen = load_screen(
    "screen.h5ad",
    guide_key="perturbation",
    control_label="non-targeting",
    covariates=["batch", "percent_mito", "n_counts"],  # optional
)

# Optional: regress covariates out before fitting
# Residualization happens first; standardization (in fit_screen) applies after
screen = residualize_screen(screen, categoricals=["batch"])

# Fit
results = fit_screen(screen, z_dim=12, l_dim=400, tau=800)

# Save
save_results(results, path="results/")

# Analyze (LFSR off by default)
tables = analyze_results(
    results,
    gene_names=screen.gene_names,
    perturbation_names=screen.perturbation_names,
)
# tables["pip_df"], tables["pve_df"], tables["overall_effect_df"], ...
```

Low-level array API (unchanged):

```python
from perturbvi import infer
results = infer(X, z_dim=12, l_dim=400, G=G, tau=800)
```

## Supported Input Formats

| Format | How to use |
|---|---|
| `.h5ad` | `load_screen("file.h5ad", guide_key="col")` or `guide_obsm="key"` |
| `10x-h5` | `load_screen("matrix.h5", format="10x-h5")` |
| `10x-mex` | `load_screen("mex_dir/", format="10x-mex")` |
| CSV/TSV | Use `infer()` directly after loading arrays manually |

Zarr is not yet first-class. Covariate residualization for 10x formats is planned; use the Python API to populate `ScreenData.covariates` manually for now.

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
