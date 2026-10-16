# Using PerturbVI with Your Data

PerturbVI uses transformed gene expression and the perturbations assigned to
each cell:

| Input | Contains | Rows × columns |
|---|---|---|
| `X` | Transformed gene-expression values | Cells × genes |
| `G` | Perturbation assignments (`1` = assigned, `0` = not assigned) | Cells × perturbations |

Rows in `X` and `G` must refer to the same cells in the same order. A cell
assigned to multiple perturbations has multiple `1`s in its row of `G`.
To preprocess raw counts, use an established method such as
[SCTransform v2](https://satijalab.org/seurat/archive/v4.3/sctransform_v2_vignette)
and pass the resulting transformed expression matrix as `X`. PerturbVI does
not select genes or call guides.

**In this workflow**

1. [Install](#install)
2. [Create the `PerturbData` object](#create-the-perturbdata-object)
3. [Fit and save](#fit-and-save)

## Install

```bash
uv pip install perturbvi
```

## Create the `PerturbData` object

Before fitting, create a `PerturbData` object and pass it to `fit_screen()`.
Choose one of these input routes:

| Input format | Create the object with |
|---|---|
| Expression and perturbation CSV or TSV files | `PerturbData(X=X, G=G)` |
| AnnData object, H5AD file, or Zarr store | `load_screen(...)` |

Both calls return a `PerturbData` object for `fit_screen()`.

### CSV or TSV files

Format the files like this:

- One row per cell; put cell IDs in the first column of both files.
- Use genes as columns of `X` and perturbations as columns of `G`.
- Load cell IDs with `index_col=0`; `PerturbData` checks IDs and row order.

If `G` has a `Nontargeting` column, pass `control="Nontargeting"`:

```python
from pathlib import Path

import pandas as pd
from perturbvi import PerturbData

data_dir = Path("data")
X = pd.read_csv(data_dir / "expression.csv", index_col=0)
G = pd.read_csv(data_dir / "perturbations.csv", index_col=0)

screen = PerturbData(X=X, G=G, control="Nontargeting")
```

`control="Nontargeting"` removes that reference column. Drop any other columns
you do not want to model directly from `G`, for example:

```python
G = G.drop(columns=["unused_column"])
```

Cells assigned only to the non-targeting control then have all-zero rows in
the remaining `G`. If you already removed the control column, omit `control`:

```python
screen = PerturbData(X=X, G=G)
```

<details markdown="1">
<summary>Add covariates</summary>

Load covariates with the same cell IDs and row order as `X` and `G`, then
pass the DataFrame to `PerturbData`:

Numeric columns are treated as continuous values; text, boolean, and pandas
`category` columns are treated as groups. Convert numeric labels, such as batch
IDs, to `category`. `fit_screen()` regresses selected covariates out of
expression before fitting. See the
[API reference](api.md#covariates-and-fitting) for details.

```python
covariates = pd.read_csv(data_dir / "covariates.csv", index_col=0)
covariates["batch"] = covariates["batch"].astype("category")

screen = PerturbData(
    X=X,
    G=G,
    control="Nontargeting",
    covariates=covariates,
)
```
</details>

??? example "Load from AnnData"
    Use `load_screen()` to create a `PerturbData` object from an AnnData
    object, H5AD file, or Zarr store. By default, it reads expression from
    `adata.X` and perturbation assignments from `adata.obsm["G"]`.

    Store `G` in `adata.obsm["G"]` as a named pandas DataFrame, with one row
    per cell in AnnData order and one column per perturbation. To use other
    data, set `x_key` to an expression layer, `g_key` to another `obsm` entry,
    or `covariates` to columns in `adata.obs`.

    ```python
    from perturbvi import load_screen

    screen = load_screen(
        "data/screen.h5ad",
        x_key="transformed",
        control="Nontargeting",
        covariates=["batch"],
    )
    ```

    Omit `control=` if `G` already has all-zero control rows and no control
    column.

## Fit and save

After preparing `screen` from either input format, fit the model and save the
results:

```python
from perturbvi import fit_screen, save_results

fit = fit_screen(screen, z_dim=20, l_dim=1000, init="pca")
save_results(fit, "results")
```

`z_dim` sets the number of latent factors. `l_dim` sets the number of
single-effect components available to each factor. `init` can be `"pca"` or
`"random"`. Genes are centered by default; set `standardize=True` to also
scale them to unit variance. See the [API reference](api.md#covariates-and-fitting)
for the remaining options.

`save_results()` writes the fitted model and labeled result matrices,
including `B`, `W`, and `BW`.

??? example "Replogle K562: genome-scale fit"
    Place the prepared files in a `data` directory. `X` contains transformed
    expression (cells × genes); `G` contains binary guide assignments for the
    same cells in the same row order (cells × perturbations). The gene ID list
    must match the columns of `X` in order. Since genome-scale `G` is sparse,
    the example converts it to JAX's [sparse BCOO format](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.BCOO.html)
    before fitting.

    ```python
    from pathlib import Path

    import anndata as ad
    import jax.numpy as jnp
    import pandas as pd
    from jax import config
    from jax.experimental import sparse

    from perturbvi import PerturbData, estimate_lfsr, fit_screen, save_results

    config.update("jax_enable_x64", True)
    config.update("jax_default_matmul_precision", "highest")

    Z_DIM, L_DIM, TAU, INIT = 20, 1000, 1, "pca"

    DATA = Path("data")
    MATRIX = DATA / "K562_essential_resid.h5ad"
    GUIDE = DATA / "wide_df.csv"
    BACKGROUND = DATA / "K562_essential_downstream_gene.tsv"
    OUTPUT = Path("results")
    DROP_COLS = ["non-targeting", "cell_barcode"]

    adata = ad.read_h5ad(MATRIX)
    guides = pd.read_csv(GUIDE, index_col=0)
    guides = guides.drop(columns=DROP_COLS, errors="ignore")
    background_genes = pd.read_csv(BACKGROUND, sep="\t")["gene_id"].tolist()

    # Pass X and G as numeric arrays; provide their column labels separately.
    screen = PerturbData(
        X=jnp.asarray(adata.X, dtype=jnp.float64),
        G=sparse.bcoo_fromdense(jnp.asarray(guides.to_numpy(), dtype=jnp.float64)),
        gene_names=background_genes,
        perturbation_names=guides.columns.tolist(),
    )

    fit = fit_screen(
        screen,
        z_dim=Z_DIM,
        l_dim=L_DIM,
        tau=TAU,
        init=INIT,
        p_prior=0.1,
        standardize=True,
        tol=1e-2,
        max_iter=1000,
    )

    save_results(fit, OUTPUT)
    lfsr_bw = estimate_lfsr(OUTPUT)
    lfsr_bw.to_csv(OUTPUT / "LFSR_BW.csv")
    ```

The [LUHMES notebook](luhmes.ipynb) shows how to plot and interpret the saved
matrices and run biological-process enrichment.
