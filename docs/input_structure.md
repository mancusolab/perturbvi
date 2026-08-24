# PerturbVI input structure

This document describes where each part of a PerturbVI screen lives in an
AnnData file, how `load_screen()` reads it, and how covariates are handled
during residualization.

## What PerturbVI needs

PerturbVI expects processed expression, a designed binary perturbation matrix,
and optional covariates. It does not read raw counts, call guides, or
normalize expression.

| PerturbVI input | Shape | Contents |
|---|---|---|
| `X` | cells × genes | Normalized, scaled, or transformed expression |
| `G` | cells × perturbations | Binary guide or target assignments |
| `covariates` (optional) | cells × covariates | Variables whose effects are removed before fitting |

## Where each input lives in AnnData

AnnData has no dedicated slots for "perturbations" or "covariates". PerturbVI
uses standard AnnData containers for cell-level annotations:

| PerturbVI input | AnnData slot | Selected with | Notes |
|---|---|---|---|
| `X` | `adata.X` or `adata.layers[x_key]` | `x_key=` (default `None` → `adata.X`) | A layer named `"X"` is reachable as `x_key="X"`; only `None` selects the default slot |
| `G` | `adata.obsm[g_key]` | `g_key=` (default `"G"`) | A named pandas DataFrame of binary columns; this is a documented PerturbVI storage convention, since AnnData reserves no `obsm` keys |
| `covariates` | `adata.obs` columns | `covariates=["batch", ...]` | Cell-level metadata; the standard AnnData home for covariates |
| gene names | `adata.var_names` | automatic | Become `gene_names` |
| perturbation names | `adata.obsm[g_key]` DataFrame columns | automatic | Become `perturbation_names`; column order is preserved |
| cell names | `adata.obs_names` | automatic | Used for row alignment; not stored in `PerturbData` |

## How names and covariates are extracted from AnnData

`load_screen()` reads each piece from its AnnData slot (`src/perturbvi/loaders.py`):

```python
gene_names = tuple(str(name) for name in adata.var_names)    # every var column, in X-column order
perturbation_names = tuple(str(name) for name in G_frame.columns)  # obsm[g_key] columns, after control drop
covariate_frame = adata.obs[covariate_names].copy()          # only the requested obs columns
```

### Gene names

- One name per expression column, taken from `adata.var_names` in the same
  order as `X` columns.
- Duplicate or empty names are rejected by validation before fitting.

### Perturbation names

- One name per `G` column, taken from the `adata.obsm[g_key]` DataFrame's
  columns.
- Column order is preserved: `perturbation_names[i]` labels `G[:, i]`.
- If `control=` is given, that column is dropped first, so it does not appear
  in `perturbation_names` or reach the model.
- Duplicate or empty names are rejected by validation; every stored column must
  contain at least one assigned cell.

### Covariates

- Only the columns explicitly requested via `covariates=["batch", ...]` are
  extracted; the loader does not infer covariates from other `obs` columns.
- They are copied from `adata.obs`, preserving column order and dtype, so
  numeric versus categorical encoding happens later in `residualize_screen`.
- An unknown column name raises immediately; missing values are rejected by
  validation or residualization.

### Cell names

Cell names (`adata.obs_names`) are used for row alignment and are not model
data.

- AnnData keys `X` rows, `obs` rows, and `obsm` rows to `obs_names`. Assigning
  a DataFrame to `obsm` checks that its index equals `obs_names`, so rows
  remain aligned.
- In the direct `PerturbData` path, DataFrame indexes are checked for equality
  between `X`, `G`, and covariates at construction.
- `PerturbData` does not store cell names after loading.

### Expression: `X`

`adata.X` is the default source. To select a named layer instead:

```python
from perturbvi import load_screen

data = load_screen(adata, x_key="normalized")
```

### Perturbations: `G` at `obsm["G"]`

The binary perturbation matrix is stored as a named pandas DataFrame in
`adata.obsm["G"]`:

```python
adata.obsm["G"] = adata.obs["target"].astype("string").str.get_dummies().astype(int)
```

The stored frame must:

- be a pandas DataFrame with named, non-empty columns;
- have exactly one row per cell (AnnData enforces alignment to `obs_names`);
- contain binary values (`0`/`1`);
- have at least one assigned cell per column;
- preserve column order exactly: `perturbation_names[i]` labels `G[:, i]`;
- allow all-zero rows, which define the model reference.

### The reference (control) column

`obsm[g_key]` may include the reference column. When it does, `control=` names
it and the loader drops it before building `PerturbData`:

```python
data = load_screen(adata, control="control", covariates=["batch", "percent_mito"])
```

The loader handles `control=` as follows (`src/perturbvi/loaders.py`):

```python
# inside load_screen()
if control is not None:
    control = str(control)
    if control not in G_frame.columns:
        raise ValueError(
            f"control column {control!r} is not present in adata.obsm[{g_key!r}]; "
            "pass control=None when G is already baseline-free"
        )
    G_frame = G_frame.drop(columns=control)
```

`control=` is drop-only: the named column must exist, otherwise the loader
raises an error. When the stored G is already baseline-free (reference rows
are all zero), omit `control=` and the frame is passed through unchanged.
PerturbVI does not verify that all-zero rows are biological controls.

## Covariates and residualization

Covariates are cell-level variables whose linear effects the analysis has
chosen to remove. They live in `adata.obs` as ordinary columns.

```python
data = load_screen(
    adata,
    control="control",
    covariates=["batch", "percent_mito"],
)
data = residualize_screen(data)
```

`load_screen` copies the requested `obs` columns into
`PerturbData.covariates`. Alignment is maintained by AnnData: `obs` rows and
`X` rows are both keyed to `obs_names`.

`residualize_screen` (or `fit_screen` internally, when covariates are present)
then:

- builds a design matrix with an intercept;
- centers numeric covariates;
- one-hot encodes categorical, string, and boolean covariates with a reference
  level dropped;
- rejects missing or non-finite covariate values;
- projects each gene onto the orthogonal complement of the design using a
  rank-aware SVD (no `C.T @ C` inversion);
- clears `PerturbData.covariates` after the regression.

Numeric and categorical columns are distinguished by pandas dtype. Whether a
column should be a covariate is a scientific decision; only the columns named
in `covariates=` are used.

## The full load call

```python
from perturbvi import load_screen, residualize_screen, fit_screen

data = load_screen(
    "screen.h5ad",
    covariates=["batch", "percent_mito"],
    control="control",
)
data = residualize_screen(data)
fit = fit_screen(data, z_dim=12, l_dim=400, tau=50)
```

If the screen is already residualized, omit the covariates argument.

## What the loader validates

Before returning, `load_screen` runs the same checks used before fitting:

- expression is a 2-D matrix with no non-finite values;
- `G` rows match expression rows and `G` columns are named;
- `G` is binary with no all-zero columns;
- gene and perturbation names are present, unique, and match matrix widths;
- requested covariate columns exist in `obs` and have no missing values.

## Direct `PerturbData` path (non-AnnData)

When the data is not in AnnData, build `PerturbData` directly from aligned
tables. DataFrame columns supply the names automatically; arrays need explicit
names:

```python
from perturbvi import PerturbData

data = PerturbData(
    X=expression,            # DataFrame or array
    G=G,                     # DataFrame or binary array
    covariates=covariates,   # DataFrame or None
)
```

For NumPy or sparse matrices, pass `gene_names=` and `perturbation_names=`
separately. CSV/TSV files are read with pandas first; paths are not parsed
automatically.

## Reproducibility metadata

When run through the CLI, the fit directory records what was loaded and
residualized:

- `run_config.json`: fit arguments plus `covariates` and the columns treated
  as categorical;
- `input_summary.json`: `X_shape`, `G_shape`, gene and perturbation names, and
  whether residualization was applied.

## Scope

PerturbVI consumes processed screens. Raw 10x feature-barcode H5/MEX output,
guide calling, QC, and expression normalization are performed upstream. The
[A375 cookbook example](cookbook.md#35-a375-10x-crispr-h5-matrix-plus-separate-calls)
shows how to read the matrix and calls with Scanpy/Cell Ranger, store the
designed `G` at `obsm["G"]`, and then load it.

See the
[Workflow](workflow.md) and
[API](api.md) for the complete behavior of `load_screen`,
`residualize_screen`, and `fit_screen`.
