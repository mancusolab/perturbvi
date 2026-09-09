[![Documentation](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/perturbvi/)
[![PyPI](https://img.shields.io/pypi/v/perturbvi.svg)](https://pypi.org/project/perturbvi/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# PerturbVI

PerturbVI infers latent gene programs and their perturbation effects from
single-cell Perturb-seq data.

## Install

```bash
uv pip install perturbvi
```

## LUHMES: fit and interpret

This example uses the processed LUHMES expression and perturbation matrices
in `luhmes/luhmes_exp.csv` and `luhmes/luhmes_G.csv`.
Expression has cells on rows and Ensembl gene IDs on columns; `G` has matching
cell rows and binary condition columns. Adjust these paths to your files.
The [LUHMES Analysis with PerturbVI tutorial](https://mancusolab.github.io/perturbvi/luhmes/)
explains the selections and shows the resulting figures and neuronal GO enrichment.

### Fit the model

The documentation example uses 20 factors, 1,000 single-effect loading
components, and PCA initialization. The expression is already processed and
scaled; `fit_screen()` centers genes. All 15 condition columns, including
`Nontargeting`, are retained to match that example.

```python
from pathlib import Path

import jax
import pandas as pd

from perturbvi import PerturbData, fit_screen, save_results

jax.config.update("jax_enable_x64", True)

data_dir = Path("luhmes")
result_dir = Path("perturbvi_results")
expression = pd.read_csv(data_dir / "luhmes_exp.csv", index_col=0)
G = pd.read_csv(data_dir / "luhmes_G.csv", index_col=0)
data = PerturbData(
    X=expression,
    G=G,
)
fit = fit_screen(
    data,
    z_dim=20,
    l_dim=1000,
    init="pca",
    tau=100,
    standardize=False,
    max_iter=500,
    tol=0.001,
    seed=0,
    verbose=True,
)
save_results(
    fit,
    result_dir,
)
```

Saving writes `model.pkl` and six labeled CSVs: `W`, `PIP_W`, `B`, `PIP_B`,
`BW`, and `PVE`. `W` is factors by genes, `B` is perturbations by factors,
and `BW = B @ W` is perturbations by genes. PIP matrices describe inclusion
probabilities; effect matrices describe magnitude and direction.

### Plot perturbation effects on factors

Install Matplotlib for plotting:

```bash
uv pip install matplotlib
```

If you already have saved results, start here and point `result_dir` to that
folder. Read only the matrix needed for each plot. Each function returns a
Matplotlib figure that you can save or customize.

```python
from pathlib import Path

import pandas as pd
from perturbvi import plotting as pp

result_dir = Path("perturbvi_results")
B = pd.read_csv(result_dir / "B.csv", index_col=0)
fig = pp.plot_factor_effects(
    B,
    perturbations=["ADNP", "ARID1B", "ASH1L", "CHD2", "PTEN", "SETD5"],
    factors=["factor_1", "factor_2", "factor_3", "factor_4", "factor_6", "factor_8",
             "factor_9", "factor_11", "factor_12", "factor_14", "factor_16"],
    show_significance=False,
    scale="asinh",
)
fig.savefig(
    result_dir / "factor_effects.png",
    dpi=300,
    bbox_inches="tight",
)
```

Omit `perturbations` and `factors` to show the full matrix. Factor IDs are
zero-based (`factor_0`); displayed labels start at Factor 1. The selections
above match the compact LUHMES example.

### Plot gene loadings on factors

Supply a `gene_annotations.csv` with three columns: `gene_ID`, `gene_name`,
and `annotation`. Place it in the results folder. `gene_ID` must match the
fitted gene IDs; the other columns supply display names and biological groups.
Genes sharing an annotation are automatically grouped together, even if their
rows are separated in the CSV. The LUHMES tutorial uses 30 selected markers
and regulators; your own annotations and selections can address any tissue
or biological question.

```python
annotations = pd.read_csv(result_dir / "gene_annotations.csv")
genes = annotations["gene_ID"].tolist()
W = pd.read_csv(result_dir / "W.csv", index_col=0)
fig = pp.plot_gene_loadings(
    W,
    genes=genes,
    factors=["factor_1", "factor_2", "factor_3", "factor_4", "factor_6", "factor_8",
             "factor_9", "factor_11", "factor_12", "factor_14", "factor_16"],
    gene_annotations=annotations,
    show_significance=False,
    scale="asinh",
)
fig.savefig(
    result_dir / "gene_loadings.png",
    dpi=300,
    bbox_inches="tight",
)
```

### Plot overall perturbation effects on genes

Overall effects combine contributions through all fitted factors, including
factors omitted from a displayed subset.

```python
BW = pd.read_csv(result_dir / "BW.csv", index_col=0)
fig = pp.plot_gene_effects(
    BW,
    genes=genes,
    perturbations=["ADNP", "ARID1B", "ASH1L", "CHD2", "PTEN", "SETD5"],
    gene_annotations=annotations,
    show_significance=False,
    scale="asinh",
)
fig.savefig(
    result_dir / "gene_effects.png",
    dpi=300,
    bbox_inches="tight",
)
```

Pass `fit.B`, `fit.W`, or `fit.BW` directly if the fit is in memory.
Each plot returns a Matplotlib figure. Asinh colorbars retain original effect
units; colors alone do not indicate significance.

### Optional overall-effect significance

LFSR requires posterior sampling. It is needed for DEG counts or optional
significance dots, but not for the heatmaps above. Compute it separately:

```bash
perturbvi lfsr perturbvi_results --draws 2000 --seed 2026
```

This writes `LFSR_BW.csv`. For significance dots, supply `lfsr=LFSR_BW`
for gene effects or `pip=PIP_B` / `pip=PIP_W` for factor effects / loadings,
and set `show_significance=True`. See the tutorial for counts and enrichment.

## Documentation

- [LUHMES Analysis with PerturbVI](https://mancusolab.github.io/perturbvi/luhmes/):
  a complete fit, plotting, and neuronal enrichment tutorial with figures.
- [Using PerturbVI with Your Data](https://mancusolab.github.io/perturbvi/workflow/):
  CSV and AnnData inputs, controls, covariates, and other screen examples.
- [API](https://mancusolab.github.io/perturbvi/api/): functions, result matrices, and CLI.

## Support

Please report bugs or feature requests in the
[issue tracker](https://github.com/mancusolab/perturbvi/issues). For questions
or comments, contact Abdullah Al Nahid (<alnahid@usc.edu>) or Nicholas Mancuso
(<nmancuso@usc.edu>).

Developed by the [Mancuso Lab](https://www.mancusolab.com/).
Distributed under the [MIT license](https://spdx.org/licenses/MIT.html).
