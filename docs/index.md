---
hide:
  - toc
---

# Single-cell Perturbation Analysis with PerturbVI

PerturbVI is a scalable approach to infer regulatory modules from single-cell Perturb-seq data.


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

See the [Workflow](workflow.md) for complete input and analysis guidance and
the [Input structure](input_structure.md) for where each piece of a screen
lives in an AnnData file. The [Cookbook](cookbook.md#3-real-genetic-screens)
covers real Datlinger, Norman, and Adamson screens.

## AnnData and CLI

For AnnData, `load_screen()` reads transformed expression from `adata.X` and
combines it with the binary `G` matrix stored in the file at `obsm["G"]` (key
selectable via `g_key=`). Prepare the file with `adata.obsm["G"]` set to a
named DataFrame of binary columns, then fit:

```bash
perturbvi fit screen.h5ad \
  --output results \
  --z-dim 12 --l-dim 400 --tau 50
```

If `G` includes the reference column, pass `--control control`; the loader
drops it before fitting.

Create result tables after fitting:

```bash
perturbvi analyze results
```

## Read next

- [Workflow](workflow.md): constructing `X` and `G`, names, covariates,
  fitting, saving, and analysis.
- [Input structure](input_structure.md): AnnData layout for `X`, `G`, and
  covariates.
- [Cookbook](cookbook.md): real LUHMES, Datlinger, Adamson, Norman, and A375
  10x examples.
- [API](api.md): Python functions, CLI options, result tables, and saved files.

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
