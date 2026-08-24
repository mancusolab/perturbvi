[![Documentation](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/perturbvi/)
[![PyPI](https://img.shields.io/pypi/v/perturbvi.svg)](https://pypi.org/project/perturbvi/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# PerturbVI

PerturbVI finds gene programs and estimates how perturbations affect them in
single-cell screens.

## Install

```bash
uv pip install perturbvi
```

## Quick start

`PerturbData` keeps expression, genetic perturbations, and covariates aligned:

| Argument | Shape | Contents |
|---|---|---|
| `X` | cells × genes | Normalized, scaled, or transformed expression |
| `G` | cells × perturbations | Binary guide or target assignments |
| `covariates` | cells × covariates | Variables whose effects should be removed from expression |

`X` and `G` are required. Include covariates when their effects should be
removed before the model is fit.

```python
from perturbvi import PerturbData, fit_screen, residualize_screen

# expression, G, and covariates are aligned tables prepared as described below.
data = PerturbData(
    X=expression,
    G=G,
    covariates=covariates,
)
data = residualize_screen(data)
fit = fit_screen(data, z_dim=12, l_dim=400, tau=50)
```

All inputs must describe the same cells in the same row order. `PerturbData`
checks DataFrame row alignment immediately; the remaining shape, name, missing
value, and binary `G` checks run before fitting. `residualize_screen()` removes
the supplied covariate effects once; the resulting data can be reused for
multiple fits. `fit_screen()` then centers expression; pass
`standardize=True` to also scale each gene to unit variance.

See the [Workflow](https://mancusolab.github.io/perturbvi/workflow/)
for complete input and analysis guidance and the
[Input structure](https://mancusolab.github.io/perturbvi/input_structure/)
page for where each piece of a screen lives in an AnnData file. The
[Cookbook](https://mancusolab.github.io/perturbvi/cookbook/#3-real-genetic-screens)
for real Datlinger, Norman, and Adamson screens.

## AnnData and CLI

For AnnData, prepare the file with transformed expression in `adata.X` and the
binary perturbation matrix in `adata.obsm["G"]`, then load and fit:

```python
from perturbvi import load_screen

data = load_screen("screen.h5ad")
```

If `adata.obsm["G"]` includes the reference column, name it so the loader drops
it before fitting:

```python
data = load_screen("screen.h5ad", control="control")
```

The CLI uses the same convention:

```bash
perturbvi fit screen.h5ad \
  --output results \
  --z-dim 12 --l-dim 400 --tau 50
```

Add `--control control` when `G` keeps its reference column.

Create result tables after fitting:

```bash
perturbvi analyze results
```

## Documentation

- [Workflow](https://mancusolab.github.io/perturbvi/workflow/): constructing
  `X` and `G`, names, covariates, fitting, saving, and analysis.
- [Input structure](https://mancusolab.github.io/perturbvi/input_structure/):
  AnnData layout for `X`, `G`, and covariates.
- [Cookbook](https://mancusolab.github.io/perturbvi/cookbook/): real LUHMES,
  Datlinger, Adamson, Norman, and A375 10x examples.
- [API](https://mancusolab.github.io/perturbvi/api/): Python functions, CLI
  options, result tables, and saved files.

## Support

Please report bugs or feature requests in the
[issue tracker](https://github.com/mancusolab/perturbvi/issues). For questions
or comments, contact Abdullah Al Nahid (<alnahid@usc.edu>) or Nicholas Mancuso
(<nmancuso@usc.edu>).

## Other Software

Other software developed by the [Mancuso Lab](https://www.mancusolab.com/):

- [SuShiE](https://github.com/mancusolab/sushie): a Bayesian fine-mapping
  framework for molecular QTL data across multiple ancestries.
- [jaxQTL](https://github.com/mancusolab/jaxqtl): scalable, count-based
  large-scale eQTL mapping.
- [MA-FOCUS](https://github.com/mancusolab/ma-focus): a Bayesian fine-mapping
  framework using [TWAS](https://www.nature.com/articles/ng.3506) statistics
  across multiple ancestries to identify causal genes for complex traits.
- [SuSiE-PCA](https://github.com/mancusolab/susiepca): scalable Bayesian
  variable selection for sparse principal component analysis.
- [twas_sim](https://github.com/mancusolab/twas_sim): simulation of
  [TWAS](https://www.nature.com/articles/ng.3506) statistics.
- [traceax](https://github.com/mancusolab/traceax): stochastic trace
  estimation for linear operators.
- [FactorGo](https://github.com/mancusolab/factorgo): scalable variational
  factor analysis for learning pleiotropic factors from GWAS summary
  statistics.
- [HAMSTA](https://github.com/tszfungc/hamsta): estimation of heritability
  explained by local ancestry data from admixture mapping summary statistics.

---

PerturbVI is distributed under the terms of the
[MIT license](https://spdx.org/licenses/MIT.html).
