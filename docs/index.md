---
hide:
  - toc
---

# PerturbVI

PerturbVI infers latent gene programs and their perturbation effects from
single-cell Perturb-seq data.

## Install

```bash
uv pip install perturbvi
```

## Quick start

Prepare an H5AD file with transformed expression in `adata.X` and a binary,
named perturbation DataFrame in `adata.obsm["G"]`.

```python
from pathlib import Path
from perturbvi import fit_screen, load_screen, save_results

result_dir = Path("results/my_screen")
data = load_screen(
    "data/screen.h5ad",
)
fit = fit_screen(
    data,
    z_dim=12,
    l_dim=100,
)
save_results(
    fit,
    result_dir,
)
```

The result directory contains the fitted posterior and labeled CSVs for
loadings, perturbation effects, inclusion probabilities, and variance summaries.

## Set up plotting

Install Matplotlib for plotting:

```bash
uv pip install matplotlib
```

```python
from perturbvi import plotting as pp

fig = pp.plot_factor_effects(
    fit.B,
    scale="asinh",
)
fig.savefig(
    result_dir / "factor_effects.png",
    dpi=300,
)
```

## Guides

- [LUHMES Analysis with PerturbVI](luhmes.md): fit the LUHMES screen, plot
  factor and gene effects, and examine neuronal GO enrichment.
- [Using PerturbVI with Your Data](workflow.md): prepare CSVs or AnnData,
  encode controls and target pairs, select covariates, fit, and save results.
- [API](api.md): function arguments, result matrices, and CLI reference.

## Support

Please report bugs or feature requests in the
[issue tracker](https://github.com/mancusolab/perturbvi/issues). For questions
or comments, contact Abdullah Al Nahid (<alnahid@usc.edu>) or Nicholas Mancuso
(<nmancuso@usc.edu>).

Developed by the [Mancuso Lab](https://www.mancusolab.com/).
Distributed under the [MIT license](license.md).
