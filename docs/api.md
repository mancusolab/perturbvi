# API

PerturbVI fits transformed cell-by-gene expression `X` together with a `0`/`1`
cell-by-perturbation matrix `G`. See the [general guide](workflow.md) for a
full example of both matrices.

## PerturbData

::: perturbvi.PerturbData

Use `PerturbData` when `X` and `G` are already DataFrames or arrays. DataFrame
column names are used automatically:

```python
from perturbvi import PerturbData

# control= drops the reference column; omit it when G is baseline-free
data = PerturbData(
    X=expression,
    G=G,
    covariates=covariates,
    control="Nontargeting",
)
```

CSV and TSV paths are not parsed automatically. Load them into pandas first,
then pass the DataFrames to `PerturbData` so row indexes and headers are under
your control. Pass `control=` when `G` keeps its reference column; omit it when
`G` is already baseline-free.

`X`, `G`, and covariates must contain the same cells in the same row order.
NumPy and sparse arrays have no column names, so provide them separately:

```python
data = PerturbData(
    X=expression_array,
    G=G_array,
    gene_names=gene_names,
    perturbation_names=perturbation_names,
)
```

`PerturbData` checks DataFrame row alignment when it is created. Full matrix,
name, value, and covariate checks run before fitting; `load_screen()` also runs
them before returning.

## Loading AnnData

::: perturbvi.load_screen

`load_screen()` reads an AnnData object, H5AD file, or AnnData Zarr folder and
returns `PerturbData`. Expression comes from `adata.X` by default; `x_key=`
selects a named layer. The binary perturbation matrix is read from
`adata.obsm[g_key]` (default `"G"`) and must be a named pandas DataFrame whose
rows match the expression matrix. See
[AnnData inputs](workflow.md#anndata) for the full AnnData layout.

```python
from perturbvi import load_screen

data = load_screen(
    "screen.h5ad",
    covariates=["batch"],
)
```

`obsm["G"]` is a PerturbVI storage convention. AnnData reserves no `obsm` keys;
`G` is an n_obs × perturbations cell-level annotation, and its DataFrame column
names become `perturbation_names`. Column order is preserved exactly:
`perturbation_names[i]` labels `G[:, i]`.

If the stored G includes the reference column, pass its name to `control=` and
the loader drops it before building `PerturbData`:

```python
data = load_screen(
    adata,
    control="control",
)
```

`control=` is drop-only: the named column must exist or the loader raises. When
the stored G is already baseline-free (reference rows are all zero), omit
`control=` and G passes through unchanged. PerturbVI does not verify that
all-zero rows are biological controls.

## Covariates and fitting

::: perturbvi.residualize_screen

::: perturbvi.FitResults

::: perturbvi.fit_screen

Pass the covariates you want removed from expression. With AnnData, use
their `obs` column names. `fit_screen()` regresses them out before fitting:

```python
data = load_screen(
    adata,
    control="control",
    covariates=["batch", "percent_mito"],
)

fit = fit_screen(
    data,
    z_dim=12,
    l_dim=100,
    tau=100,
)
```

The main model settings are:

| Setting | Meaning |
|---|---|
| `z_dim` | Number of gene programs to fit |
| `l_dim` | Number of single-gene effects available to build each program |
| `tau` | Starting inverse noise level for expression; PerturbVI updates it during fitting |

Numeric covariates are treated as measurements. Text, categorical, and boolean
covariates are treated as groups. If a covariate overlaps with `G`, shared
signal may be removed; perfectly confounded effects cannot be separated.
To reuse corrected expression across fits, call `residualize_screen(data)`
once and pass its result to `fit_screen()`. `fit_screen()` centers
every gene. Set `standardize=True` to also scale genes to unit variance. It does not
perform raw-count QC, normalization, gene selection, or guide calling. See
[Covariates](workflow.md#covariates) for the full behavior.

## Save results and compute LFSR

::: perturbvi.save_results

### Labeled matrices in memory

`fit_screen()` returns a `FitResults` object. Access each matrix directly;
there is no separate analysis step or dictionary of tables.

| Property / CSV stem | Rows × columns | Meaning |
|---|---|---|
| `fit.W` / `W` | factors × genes | Inclusion-weighted posterior mean loadings |
| `fit.PIP_W` / `PIP_W` | factors × genes | Gene-loading inclusion probabilities |
| `fit.B` / `B` | perturbations × factors | Inclusion-weighted mean effects on factors |
| `fit.PIP_B` / `PIP_B` | perturbations × factors | Coefficient inclusion probabilities |
| `fit.BW` / `BW` | perturbations × genes | Overall effects, `B @ W` |
| `fit.PVE` / `PVE` | factors × 1 | Per-factor expression variance summary |

These properties return ordinary pandas DataFrames. They compute the requested
matrix on access; they do not sample or write files. Store a matrix in a variable
if you will reuse it. Raw arrays remain available through `fit.inference`.

```python
from perturbvi import plotting as pp

fig = pp.plot_factor_effects(
    fit.B,
    perturbations=["ADNP", "PTEN", "SETD5"],
    show_significance=False,
    scale="asinh",
)
```

### Saved files

`save_results(fit, "results")` writes `model.pkl` and the six CSVs listed above.
It does not sample LFSR or write duplicate TXT summaries. The saved model retains
the fitted posterior and gene/perturbation names; plotting the CSVs does not
require it. The fitting CLI also records fit arguments in `run_config.json`
and input information in `input_summary.json`.

To regenerate summaries from a saved posterior without refitting:

```python
from perturbvi import save_results

save_results("results")
```

This refreshes the six CSVs and leaves an existing `model.pkl` and `LFSR_BW.csv`
untouched. Older `params_file.pkl` fits are accepted internally. Files without
saved labels use positional identifiers; the original gene and perturbation
order is needed to relabel them.

### Optional LFSR

::: perturbvi.estimate_lfsr

```python
from perturbvi import estimate_lfsr

LFSR_BW = estimate_lfsr(
    "results",
    draws=2_000,
    seed=1,
)
LFSR_BW.to_csv("results/LFSR_BW.csv")
```

Replace `"results"` with `fit` to use the in-memory posterior. This samples
only overall-effect sign uncertainty, returning perturbations on rows and
genes on columns. It does not return or regenerate the six summary matrices.
If LFSR has already been computed for this fit, read its CSV instead.

## CLI

Fit a prepared file whose binary matrix lives at `obsm["G"]`:

```bash
perturbvi fit screen.h5ad \
  --output results --z-dim 12 --l-dim 100 --tau 100
```

If `G` includes the reference column, pass `--control control`; the loader
drops it before fitting. Expression can be selected with `--x-key <layer>`, and
the perturbation key with `--g-key <obsm_key>` (default `"G"`).

Fitting already writes the six result tables. Compute LFSR separately:

```bash
perturbvi lfsr results --draws 2000 --seed 1
```

This command writes only `LFSR_BW.csv`.

## Read result tables

```python
import pandas as pd

BW = pd.read_csv("results/BW.csv", index_col=0)
BW.head()
```

The CLI writes the same CSVs directly. Read `LFSR_BW.csv` only when LFSR was
computed for this fit. Load only the matrices needed for the plot; each plotting
function receives an individual DataFrame. If identifiers such as
`001` or `NA` must remain strings, use pandas options such as `dtype={0: str}`
and `keep_default_na=False` for the first column.

## Plot interpretation tables

Install Matplotlib for plotting:

```bash
uv pip install matplotlib
```

Each function accepts one labeled DataFrame and returns a Matplotlib Figure.
Read the corresponding CSV directly, or pass `fit.B`, `fit.W`, or `fit.BW`.
Full matrices and explicit subsets use the same functions. See the
[LUHMES Analysis with PerturbVI](luhmes.md).

Appearance options are `scale` (`"linear"` or `"asinh"`), `cmap`, and
`colorbar_ticks`. Typography, spacing, and italic gene labels are automatic.
By default, each colorbar has five markers evenly spaced along the displayed
scale, including zero and both limits. Labels retain original units and are
rounded to one decimal place. The displayed data determine the symmetric
range. For an explicit override, supplied `colorbar_ticks` define the range
using their largest absolute value. Use `ax` to compose plots and
ordinary Matplotlib commands to customize the returned figure or axis labels.

For gene heatmaps, `gene_annotations` accepts a DataFrame read directly from a
CSV with `gene_ID`, `gene_name`, and `annotation` columns. No ordering index is
needed. Genes are automatically grouped by annotation, with groups in first-seen
CSV order and unannotated genes last. Supply
`genes=annotations["gene_ID"].tolist()` to preserve CSV order within groups, or
another gene list to choose the within-group order. Repeated annotations share
a color and appear once in the two-column legend below the plot; annotation
text is displayed verbatim.

| Function | Required DataFrame | Optional uncertainty input | Significance rule |
|---|---|---|---|
| `plot_factor_effects(B, ...)` | `B`: perturbations × factors | `pip=PIP_B` | PIP > 0.95 |
| `plot_gene_loadings(W, ...)` | `W`: factors × genes | `pip=PIP_W` | PIP > 0.95 |
| `plot_gene_effects(BW, ...)` | `BW`: perturbations × genes | `lfsr=LFSR_BW` | LFSR < 0.05 |

Dots are off by default (`show_significance=False`). Set the flag to `True`
and supply the corresponding uncertainty matrix to show them. The effect and
uncertainty matrices must have the same row and column identifiers; their
order may differ. The plotting functions align them and handle display
transposes internally. No model loading, file loading, or sampling occurs.

```python
from perturbvi import plotting as pp

fig = pp.plot_gene_effects(
    BW,
    genes=genes,
    perturbations=["ADNP", "PTEN", "SETD5"],
    gene_annotations=annotations,
    show_significance=False,
    scale="asinh",
)
```

::: perturbvi.plotting.plot_factor_effects

::: perturbvi.plotting.plot_gene_loadings

::: perturbvi.plotting.plot_gene_effects

Selected matrices and color settings are recorded in `fig.perturbvi_data`,
with one record per heatmap. Biological labels/groups are optional user-supplied
tables. Enrichment and its visualization use direct R code in the tutorial.

## Using arrays directly

To call the core model without `PerturbData`, pass `X` and `G` directly:

```python
results = infer(
    X,
    G,
    z_dim=20,
    l_dim=10,
    tau=10.0,
)
```

::: perturbvi.infer.infer

::: perturbvi.infer.compute_elbo

::: perturbvi.infer.compute_pip

::: perturbvi.infer.compute_pve
