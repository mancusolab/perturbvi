[![Documentation](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/perturbvi/)
[![PyPI](https://img.shields.io/pypi/v/perturbvi.svg)](https://pypi.org/project/perturbvi/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# PerturbVI

Perturbvi is a scalable approach to infer regulatory modules through informative latent component model in the single-cell Perturb-seq data.

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

data = load_screen(
    "screen.h5ad",
    x_key=None,  # None (Default) = adata.X
    g_key="G",  # "G" (Default) = adata.obsm["G"]
    control=None,
)

data = load_screen(
    "screen.h5ad",
    x_key="transformed",  # adata.layers["transformed"]
    g_key="G",  # "G" (Default) = adata.obsm["G"]
    control=None,
)

data = load_screen(
    "screen.h5ad",
    x_key="counts",  # adata.layers["counts"]
    g_key="perturbations",  # adata.obsm["perturbations"]
    control="Nontargeting",  # drop the reference column
)

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
| `control` | label | Reference column to drop from `G` (default: none) |

```python
from perturbvi import PerturbData, fit_screen, residualize_screen

# control= drops the reference column; omit it when G is baseline-free
data = PerturbData(
    X=expression,
    G=G,
    covariates=covariates,
    control="Nontargeting",
)

data = residualize_screen(data)  # optional

fit = fit_screen(data, z_dim=12, l_dim=400, tau=50)
```

`X` and `G` are both required, and their rows must refer to the same cells in
the same order. Read CSV/TSV files with pandas first, then pass the resulting
DataFrames so gene and perturbation names stay aligned.

`fit_screen()` always centers each gene across cells. If your expression is
not already scaled, pass `standardize=True` to also divide each gene by its
standard deviation, giving every gene unit variance.

`control=` names a reference column in `G` to drop (for example
`"Nontargeting"`). Omit it when `G` is already baseline-free.

See the [Workflow](https://mancusolab.github.io/perturbvi/workflow/)
for complete input and analysis guidance and the
[Input structure](https://mancusolab.github.io/perturbvi/input_structure/)
page for where each piece of a screen lives in an AnnData file. The
[Cookbook](https://mancusolab.github.io/perturbvi/cookbook/#3-real-genetic-screens)
for real Datlinger, Norman, and Adamson screens.

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
