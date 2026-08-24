# From a screen to PerturbVI

PerturbVI learns gene programs from transformed single-cell expression and a
binary genetic perturbation design. This page follows the five decisions that
matter: prepare the screen, build `X` and `G`, keep names aligned, choose
covariates, and fit and inspect the results.

## 1. Start with a model-ready screen

PerturbVI starts after upstream single-cell processing. Raw RNA and guide
measurements must first be turned into:

- retained cells and genes;
- normalized, scaled, or transformed expression;
- final guide or target calls for each cell;
- any measured covariates selected for removal.

PerturbVI does not perform guide calling or decide the experiment's QC and
expression transformation. The [cookbook's raw-count
example](cookbook.md#2-a-basic-raw-count-transformation) shows one compact
Scanpy transformation; its [LUHMES example](cookbook.md#31-luhmes-crop-seq-processed-x-and-g)
starts from expression already prepared by the GSFA authors. The [A375 10x
example](cookbook.md#35-a375-10x-crispr-h5-matrix-plus-separate-calls) shows how
to convert a 10x H5 file outside PerturbVI and then load the resulting AnnData.

The high-level input is `PerturbData`:

| Argument | Shape | Contents |
|---|---|---|
| `X` | cells × genes | Transformed expression |
| `G` | cells × perturbations | Binary guide or target assignments |
| `covariates` | cells × covariates | Variables whose effects will be removed from expression |
| `control` | label | Reference column to drop from `G` (default: none) |

`X` and `G` are required. Covariates are included when the analysis calls for
them. Every input must describe the same cells in the same row order.

## 2. Build and label X and G

See [Input structure](input_structure.md) for where each piece of a screen
lives in an AnnData file and how `load_screen()` reads it.

### Expression: `X`

Each row of `X` is a cell and each column is a gene. Its values must already be
normalized, scaled, or transformed rather than raw UMI counts.

This example has four cells and three genes, so `X.shape == (4, 3)`:

| cell | STAT1 | IRF1 | CXCL10 |
|---|---:|---:|---:|
| cell_001 | 0.3 | -1.1 | 0.5 |
| cell_002 | 1.2 | -0.4 | 0.8 |
| cell_003 | -0.7 | 1.4 | -0.2 |
| cell_004 | 1.7 | 1.0 | 0.4 |

`fit_screen()` centers every gene. Set `standardize=True` when PerturbVI should
also scale genes to unit variance.
For a complete raw-count example, see
[Shared preprocessing](cookbook.md#2-a-basic-raw-count-transformation).

### Perturbations: `G`

Each row of `G` is the same cell as the corresponding row of `X`. Each column
is a modeled guide or target, and every value is `0` or `1`.

This example has the same four cells and two targets, so `G.shape == (4, 2)`:

| cell | A | B |
|---|---:|---:|
| cell_001 | 0 | 0 |
| cell_002 | 1 | 0 |
| cell_003 | 0 | 1 |
| cell_004 | 1 | 1 |

A `1` means that target is present. All-zero rows are allowed. A row may also
contain more than one target, as in `cell_004`. Every column must contain at
least one assigned cell.

Choose the resolution of `G` before fitting. Its columns may represent
individual guides or target genes. PerturbVI does not collapse guides to genes
or split target-pair labels on its own. See the [Adamson guide-to-target
example](cookbook.md#33-adamson-crispri-guides-collapsed-to-targets) and the
[Norman target-pair example](cookbook.md#34-norman-crispra-single-targets-and-target-pairs).

### Names for genes and perturbations

Names label the returned loadings, probabilities, and perturbation effects.
They do not change the numerical fit, but they must be unique and remain in the
same order as the matrix columns.

DataFrame columns supply the names automatically:

```python
import pandas as pd

from perturbvi import PerturbData, residualize_screen

expression = pd.read_csv("expression.csv", index_col=0)
G = pd.read_csv("perturbations.csv", index_col=0)

# control= drops the reference column; omit it when G is baseline-free
data = PerturbData(X=expression, G=G, control="Nontargeting")
```

`PerturbData` accepts the resulting DataFrames; it does not guess how CSV or
TSV paths should be interpreted. Read delimited files with pandas first so you
can choose the index, headers, and alignment explicitly.

The two DataFrame indexes must contain the same cells in the same order. For
NumPy or sparse matrices, pass the column names explicitly:

```python
data = PerturbData(
    X=expression_array,
    G=G_array,
    gene_names=gene_names,
    perturbation_names=perturbation_names,
)
```

### AnnData, H5AD, and Zarr

`load_screen()` reads an AnnData object, H5AD file, or AnnData Zarr folder.
Expression comes from `adata.X` unless `x_key=...` selects another layer.
Gene names come from `adata.var_names`. The binary perturbation matrix is read
from `adata.obsm[g_key]` (default `"G"`) and must be a named pandas DataFrame
with one row per cell.

Store the matrix during preparation. For a screen with one final target label
per cell:

```python
from perturbvi import load_screen

adata.obsm["G"] = (
    adata.obs["target"].astype("string").str.get_dummies().astype(int)
)

data = load_screen(
    adata,
    control="control",
    covariates=["replicate", "percent_mito"],
)
```

The stored frame may keep the reference column; `control=` names it and the
loader drops it before fitting. Baseline-free frames simply omit the argument.
The [Datlinger code block](cookbook.md#32-datlinger-crop-seq-one-target-per-cell)
shows the preparation with real `obs` rows and the resulting `G`; the [Norman
code block](cookbook.md#34-norman-crispra-single-targets-and-target-pairs) shows
how real target-pair labels become a multi-target `G` (built directly in that
example).

## 3. Choose covariates deliberately

Covariates are measured cell-level variables whose linear effects the analysis
has chosen to remove from expression. They are stored with `X` and `G`:

```python
covariates = pd.read_csv("covariates.csv", index_col=0)
data = PerturbData(X=expression, G=G, covariates=covariates)
data = residualize_screen(data)
```

Their rows must match `X` and `G`. Numeric columns are treated as continuous
measurements. Text, categorical, and boolean columns are represented as
groups. With AnnData, pass their `obs` names:

```python
adata.obs["replicate"] = adata.obs["replicate"].astype("category")

data = load_screen(
    adata,
    control="control",
    covariates=["replicate", "percent_mito"],
)
data = residualize_screen(data)
```

There is no universal covariate list. The choice follows the experimental
design and scientific question. Before passing a column, determine whether its
effect is intended to be removed, whether it overlaps with `G`, and whether
the same correction was already applied to `X`. Shared effects cannot be
separated under perfect confounding, and applying the same correction twice
is unnecessary. The [Scanpy regression
documentation](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.regress_out.html)
also notes that regression can overcorrect.

`residualize_screen()` regresses all supplied covariates from every gene
together and clears `data.covariates`. Run it once, then reuse the residualized
data for one or more fits. `fit_screen()` centers that expression and can
optionally scale it with `standardize=True`.
The [dataset examples](cookbook.md#3-real-genetic-screens) show how covariates
are selected and passed alongside each screen.

## 4. Fit the model

```python
from perturbvi import fit_screen

fit = fit_screen(
    data,
    z_dim=12,
    l_dim=100,
    tau=800.0,
    max_iter=500,
    seed=1,
    verbose=True,
)
```

The main model settings are:

| Setting | Meaning |
|---|---|
| `z_dim` | Number of gene programs |
| `l_dim` | Number of single-gene effects available to construct each program |
| `tau` | Starting inverse noise level; it is updated during fitting |

Before fitting, PerturbVI checks matrix dimensions, aligned DataFrame indexes,
finite expression, binary `G`, nonempty perturbation columns, unique names, and
complete covariates. The [API](api.md#covariates-and-fitting) documents all fit
arguments, and the cookbook's [common fit block](cookbook.md#4-fit-save-and-analyze-an-example)
shows the same call after a real dataset has been prepared.

## 5. Save and analyze

`save_results()` writes the fitted parameters and core model matrices.
`analyze()` returns labeled result tables. Local false sign rates are computed
only when requested.

```python
from pathlib import Path

from perturbvi import save_results
from perturbvi.utils import analyze

output = Path("results")
save_results(fit, str(output))

tables = analyze(
    fit,
    compute_lfsr=True,
    lfsr_iters=2_000,
    seed=1,
)

for name, table in tables.items():
    table.to_csv(output / f"{name}.csv")
```

The Python `analyze()` function returns tables; the loop above chooses to save
them as CSV files. See [Saved files](api.md#saved-files) for the files written
by `save_results()` and [Save and analyze](api.md#save-and-analyze) for the
contents of every analysis table. The cookbook's [fit, save, and analyze block](cookbook.md#4-fit-save-and-analyze-an-example)
is a runnable version of this final step.
