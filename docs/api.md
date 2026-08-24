# API

PerturbVI fits transformed cell-by-gene expression `X` together with a `0`/`1`
cell-by-perturbation matrix `G`. See the [Workflow](workflow.md) for a
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
[Input structure](input_structure.md) for the full AnnData layout.

```python
from perturbvi import load_screen

data = load_screen(adata)                     # adata.X + adata.obsm["G"]
data = load_screen("screen.h5ad", covariates=["batch"], g_key="G")
```

`obsm["G"]` is a PerturbVI storage convention. AnnData reserves no `obsm` keys;
`G` is an n_obs × perturbations cell-level annotation, and its DataFrame column
names become `perturbation_names`. Column order is preserved exactly:
`perturbation_names[i]` labels `G[:, i]`.

If the stored G includes the reference column, pass its name to `control=` and
the loader drops it before building `PerturbData`:

```python
data = load_screen(adata, control="control")
```

`control=` is drop-only: the named column must exist or the loader raises. When
the stored G is already baseline-free (reference rows are all zero), omit
`control=` and G passes through unchanged. PerturbVI does not verify that
all-zero rows are biological controls.

## Covariates and fitting

::: perturbvi.residualize_screen

::: perturbvi.FitResults

::: perturbvi.fit_screen

Passing covariates records the columns to regress out. PerturbVI does not
choose them. With AnnData, pass their `obs` names, then residualize once before
fitting:

```python
data = load_screen(
    adata,
    control="control",
    covariates=["batch", "percent_mito"],
)

data = residualize_screen(data)
fit = fit_screen(data, z_dim=12, l_dim=400, tau=50)
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
The residualized `data` can be reused for multiple fits. `fit_screen()` centers
every gene. Set `standardize=True` to also scale genes to unit variance. It does not
perform raw-count QC, normalization, gene selection, or guide calling. See
[Workflow](workflow.md#3-choose-covariates-deliberately) for the full behavior.

## Save and analyze

::: perturbvi.save_results

::: perturbvi.load_results

::: perturbvi.utils.analyze

```python
save_results(fit, "results")
tables = analyze(fit)
```

`analyze()` returns five DataFrames with row and column names. It does not
write files:

| Key | Shape | Meaning |
|---|---|---|
| `pip` | genes × programs | Probability that each gene contributes to each program |
| `pve` | programs × 1 | Share of expression variation explained by each program |
| `perturbation_effect` | perturbations × programs | Estimated effect of each perturbation on each program |
| `perturbation_pip` | perturbations × programs | Probability that each perturbation affects each program |
| `gene_effect` | genes × perturbations | Estimated effect of each perturbation on each gene |

Local false sign rates (LFSR) can take substantial time, so they are computed
only when requested:

```python
tables = analyze(fit, compute_lfsr=True, lfsr_iters=2_000, seed=1)
```

This adds one `lfsr` table. `analyze()` does not read an old LFSR file or
compute LFSR unless `compute_lfsr=True`.

### Saved files

`save_results()` writes four files:

| File | Contents |
|---|---|
| `W.txt` | Gene loadings for each program |
| `pip.txt` | Probability that each gene contributes to each program |
| `pve.txt` | Expression variation explained by each program |
| `params_file.pkl` | Fitted parameters and labels |

The CLI additionally writes `run_config.json` (the fit arguments, including
covariates and the columns treated as categorical) and `input_summary.json`
(expression and perturbation shapes plus gene and perturbation names) into the
same directory for reproducible runs.

`load_results()` reloads the fit from `params_file.pkl`. The three text files
are provided for inspection and are not needed when reloading.

## CLI

Fit a prepared file whose binary matrix lives at `obsm["G"]`:

```bash
perturbvi fit screen.h5ad \
  --output results --z-dim 12 --l-dim 400 --tau 50
```

If `G` includes the reference column, pass `--control control`; the loader
drops it before fitting. Expression can be selected with `--x-key <layer>`, and
the perturbation key with `--g-key <obsm_key>` (default `"G"`).

Write the five result tables, optionally adding LFSR:

```bash
perturbvi analyze results
perturbvi analyze results --compute-lfsr --lfsr-iters 2000
```

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
