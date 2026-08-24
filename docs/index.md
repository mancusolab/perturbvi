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

### From AnnData (recommended)

Prepare the file with transformed expression in `adata.X` and the binary
perturbation matrix in `adata.obsm["G"]`, then load and fit:

```python
from perturbvi import fit_screen, load_screen, residualize_screen

data = load_screen("screen.h5ad")  # control=None (default): G has no reference column
data = load_screen("screen.h5ad", control="Nontargeting")  # G keeps its reference column

data = residualize_screen(data)  # optional; only if you loaded covariates
fit = fit_screen(data, z_dim=12, l_dim=400, tau=50)
```

Same workflow from the CLI:

```bash
perturbvi fit screen.h5ad \
  --output results \
  --z-dim 12 --l-dim 400 --tau 50
```

Omit `--control` when `G` is baseline-free; add `--control Nontargeting` when
`G` keeps its reference column.

```bash
perturbvi analyze results
```

### Already have `X` and `G`? (arrays or CSV)

`PerturbData` keeps expression, perturbations, and covariates aligned:

| Argument | Shape | Contents |
|---|---|---|
| `X` | cells × genes | Normalized, scaled, or transformed expression |
| `G` | cells × perturbations | Binary guide or target assignments |
| `covariates` | cells × covariates | Variables whose effects should be removed from expression |

```python
from perturbvi import PerturbData, fit_screen, residualize_screen

# control= drops the reference column; omit it when G is baseline-free
data = PerturbData(X=expression, G=G, covariates=covariates, control="Nontargeting")
data = residualize_screen(data)  # optional
fit = fit_screen(data, z_dim=12, l_dim=400, tau=50)
```

`X` and `G` are required and must list the same cells in the same row order.
Read CSV/TSV with pandas first, then pass the DataFrames. `fit_screen()`
centers expression; pass `standardize=True` to also scale each gene to unit
variance. `control=` names the reference column to drop from `G`; omit it when
`G` is already baseline-free.

See the [Workflow](workflow.md) for complete input and analysis guidance and
the [Input structure](input_structure.md) for where each piece of a screen
lives in an AnnData file. The [Cookbook](cookbook.md#3-real-genetic-screens)
covers real Datlinger, Norman, and Adamson screens.


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
