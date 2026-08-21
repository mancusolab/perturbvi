[![Documentation](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/perturbvi/)
[![PyPI](https://img.shields.io/pypi/v/perturbvi.svg)](https://pypi.org/project/perturbvi/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# PerturbVI

PerturbVI learns latent gene programs and perturbation effects from
single-cell perturbation screens.

## Install

```bash
uv pip install perturbvi
```

## Before you start

PerturbVI uses two matrices containing the same cells in the same order:

- `X` contains expression values, with genes in columns.
- `G` records the perturbation design, with perturbations in columns. Controls
  are all-zero rows; combination perturbations have more than one active
  column.

For H5AD data, `load_screen` can construct `G` from a perturbation label in
`obs` or read a prepared matrix from `obsm`. For 10x H5 or MEX data, provide a
barcode-indexed metadata table containing the prepared assignments. Raw guide
counts must be converted to assignments before fitting. Covariates are
optional and are used only when residualization is requested.

## CLI

### H5AD

```bash
perturbvi fit screen.h5ad \
  --guide-key perturbation \
  --control-label control \
  --covariates batch log_total_counts percent_mito \
  --categorical-covariates batch \
  --output results \
  --z-dim 12 --l-dim 400 --tau 50
```

`--guide-key` names the observation column containing prepared perturbation
labels. Use `--guide-obsm guide_matrix` when AnnData already contains a named
binary or multi-hot `G` matrix.

### Current 10x H5 or MEX

```bash
perturbvi fit filtered_feature_bc_matrix.h5 \
  --format 10x-h5 \
  --metadata cell_metadata.tsv \
  --guide-key perturbation \
  --control-label control \
  --covariates batch log_total_counts percent_mito \
  --categorical-covariates batch \
  --output results \
  --z-dim 12 --l-dim 400 --tau 50
```

The metadata index must match the 10x barcodes and contain prepared
perturbation assignments, such as Cell Ranger protospacer calls. PerturbVI
selects gene-expression features but does not threshold raw guide features.
Use `--format 10x-mex` for a current Cell Ranger MEX directory.

### Analyze

```bash
perturbvi analyze results --output results/analysis
```

When covariates and `--standardize` are both requested, PerturbVI residualizes
expression first and standardizes the residuals. Skip residualization when the
input expression is already residualized.

## Python

```python
from perturbvi import analyze, fit_screen, load_screen, residualize_screen, save_results

screen = load_screen(
    "screen.h5ad",
    guide_key="perturbation",
    control_label="control",
    covariates=["batch", "log_total_counts", "percent_mito"],
)
screen = residualize_screen(screen, categorical_covariates=["batch"])
result = fit_screen(screen, z_dim=12, l_dim=400, tau=50, seed=1)

save_results(
    result,
    "results",
    gene_names=screen.gene_names,
    perturbation_names=screen.perturbation_names,
)
tables = analyze(
    result,
    gene_names=screen.gene_names,
    perturbation_names=screen.perturbation_names,
)
```

`load_screen` also accepts an AnnData object.

## Formats

| Input | CLI | Python |
|---|---:|---:|
| H5AD | yes | yes |
| AnnData object | no | yes |
| CSV/TSV | yes | yes |
| Current 10x H5/MEX | yes | yes |

CSV and TSV are intended for small or medium dense matrices. H5AD and 10x
loading preserves sparse expression.

Zarr is not a first-class input format. Save an AnnData object as H5AD before
loading it with PerturbVI.

## Outputs

Every saved fit contains `W.txt`, `pip.txt`, `pve.txt`, and
`params_file.pkl`. Named fits also contain `gene_names.txt` and
`perturbation_names.txt`; CLI fits add `run_config.json` and
`input_summary.json`.

By default, `perturbvi analyze` writes `pip.csv`, `pve.csv`, `beta.csv`,
`p_hat.csv`, `overall_effect.csv`, `pip_significant.csv`, and
`pip_summary.csv`. With `--compute-lfsr`, it also writes `lfsr.csv` and
`lfsr_significant.csv`.

See the [output reference](docs/api.md#saved-files) for details.

## Direct array use

Use `infer` when `X` and `G` are already aligned and prepared. Controls are
all-zero rows in `G`; do not add a control column.

```python
from perturbvi import infer

result = infer(
    X,
    z_dim=12,
    l_dim=400,
    G=G,
    tau=50,
    seed=1,
)
```

See the [cookbook](examples/cookbook.md) and the
[API documentation](https://mancusolab.github.io/perturbvi/) for additional
options.

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

---

This project has been set up using Hatch. See the
[Hatch documentation](https://hatch.pypa.io/) for usage information.
