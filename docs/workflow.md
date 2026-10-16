# Using PerturbVI with Your Data

PerturbVI fits transformed expression together with a binary perturbation
assignment matrix. It does not preprocess raw counts, select genes, or call
guides; prepare those inputs before fitting.

For raw-count data, perform quality control, normalization, and gene selection
before creating `X`. SCTransform v2 is one option for count normalization and
variance stabilization; see the
[SCTransform v2 guide](https://satijalab.org/seurat/archive/v4.3/sctransform_v2_vignette).

## Install

```bash
uv pip install perturbvi
```

## Prepare a screen

`X` has cells on rows and genes on columns. `G` has the same cells, in the
same order, on rows and perturbations on columns. Each entry of `G` is `0` or
`1`; a cell with multiple perturbations has a `1` in each corresponding
column. Give genes and perturbations unique column names.

Control cells should have all-zero rows in `G`. If `G` has a control column,
pass its name as `control` so PerturbVI drops it before fitting. If controls
are already represented by all-zero rows, leave `control` unset.

### CSV or TSV files

Use cell IDs as the row index in each file. For TSV files, add `sep="\t"` to
`read_csv()`.

```python
from pathlib import Path

import pandas as pd
from perturbvi import PerturbData

data_dir = Path("data")
X = pd.read_csv(data_dir / "expression.csv", index_col=0)
G = pd.read_csv(data_dir / "perturbations.csv", index_col=0)

data = PerturbData(X=X, G=G, control="Nontargeting")
```

Omit `control=` when `G` is already baseline-free.

??? example "Add covariates"
    Load covariates with the same cell IDs and row order as `X` and `G`, then
    pass the DataFrame to `PerturbData`:

    ```python
    covariates = pd.read_csv(data_dir / "covariates.csv", index_col=0)
    covariates["batch"] = covariates["batch"].astype("category")

    data = PerturbData(
        X=X,
        G=G,
        control="Nontargeting",
        covariates=covariates,
    )
    ```

??? example "Load data from AnnData"
    `load_screen()` reads an AnnData object, an H5AD file, or a Zarr store. It
    uses `adata.X` for expression and `adata.obsm["G"]` for perturbation
    assignments by default. Store `G` there as a named pandas DataFrame with
    one row per cell in AnnData order and one column per perturbation. Use
    `x_key` to read expression from a layer, `g_key` to select another
    `obsm` entry, and `covariates` to select columns from `adata.obs`.

    ```python
    from perturbvi import load_screen

    data = load_screen(
        "data/screen.h5ad",
        x_key="transformed",
        control="Nontargeting",
        covariates=["batch"],
    )
    ```

    Omit `control=` if `G` already has all-zero control rows and no control
    column.

### Covariates

Numeric covariates are treated as continuous values. Text and boolean columns,
and columns with pandas' `category` dtype, are treated as groups. If numbers
label groups rather than measure amounts (for example, batch IDs), convert
that column with `.astype("category")`; otherwise PerturbVI treats the numbers
as a continuous trend. `fit_screen()` regresses selected covariates out of
expression before fitting. See the
[API reference](api.md#covariates-and-fitting) for details.

## Fit and save

Both input routes create a `PerturbData` object called `data`. Fit and save it
like this:

```python
from perturbvi import fit_screen, save_results

fit = fit_screen(data, z_dim=20, l_dim=1000, init="pca")
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
    Place the prepared files in an `input` directory. `X` contains transformed
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
