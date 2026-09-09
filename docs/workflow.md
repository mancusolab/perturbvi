# Using PerturbVI with Your Data

Start with an expression matrix and perturbation assignments for the same cells.

## 1. Set up

```bash
uv pip install perturbvi matplotlib
```

```python
from pathlib import Path

import pandas as pd
from perturbvi import PerturbData, fit_screen, load_screen, save_results

data_dir = Path("data")
result_dir = Path("results")
```

## 2. Prepare the inputs

| Input | Rows × columns | Contents |
|---|---|---|
| `X` | Cells × genes | Transformed expression |
| `G` | Cells × perturbations | Binary target or guide assignments |
| `covariates`, optional | Cells × covariates | Variables to regress out of expression |

Keep cells in the same order in every input. Use unique gene and perturbation
names for the columns. If you have gene annotations, their IDs must match
the expression matrix.

Before fitting, filter low-quality cells and genes, normalize expression,
select genes, and assign perturbations to cells. PerturbVI expects these steps
to be complete.

### Controls and multiple perturbations

In `G`, use 1 when a cell has a perturbation and 0 otherwise. Cells with
multiple perturbations have multiple 1s:

| Cell assignment | `G[CEBPE]` | `G[RUNX1T1]` |
|---|---:|---:|
| Control | 0 | 0 |
| CEBPE | 1 | 0 |
| RUNX1T1 | 0 | 1 |
| CEBPE + RUNX1T1 | 1 | 1 |

If `G` has a control column, `control="Nontargeting"` removes that column
while keeping the control cells. Without this option, the control column is
fitted like any other perturbation. If control cells already have all-zero
rows, leave `control` unset.

### CSV or TSV

Use cell IDs as row labels and gene or perturbation names as column headers.
For TSV files, add `sep="\t"` to `pd.read_csv()`.

```python
X = pd.read_csv(data_dir / "expression.csv", index_col=0)
G = pd.read_csv(data_dir / "perturbations.csv", index_col=0)

data = PerturbData(X=X, G=G)
```

??? example "NumPy or sparse arrays"
    With arrays, provide the gene and perturbation names separately.

    ```python
    data = PerturbData(
        X=X_arr,
        G=G_arr,
        gene_names=gene_names,
        perturbation_names=perturbation_names,
    )
    ```

### AnnData

`load_screen()` reads AnnData objects, H5AD files, and AnnData Zarr folders.
It expects:

| Location | Contents |
|---|---|
| `adata.X` | Transformed expression; alternatively use a named layer |
| `adata.obs_names` | Cell identifiers |
| `adata.var_names` | Gene identifiers |
| `adata.obsm["G"]` | Binary pandas DataFrame, indexed by cell and with named columns |
| `adata.obs`, optional | Covariate columns |

```python
data = load_screen(data_dir / "screen.h5ad")
```

You can also pass an already loaded AnnData object. To read expression from
`adata.layers["transformed"]`, set `x_key="transformed"`. To read assignments
from `adata.obsm["perturbations"]`, set `g_key="perturbations"`.

??? example "Build G from one target label per cell"
    Here `adata.obs["target"]` contains target names and the label `control`.

    ```python
    adata.obsm["G"] = adata.obs["target"].str.get_dummies().astype(int)

    data = load_screen(adata, control="control")
    ```

### Covariates

Pass covariates such as batch or mitochondrial percentage to adjust expression
before fitting. `fit_screen()` performs the regression automatically.

Numeric columns are treated as continuous values; text, categorical, and
boolean columns define groups. For batches numbered 1, 2, 3, and so on,
convert the column to `category`.

??? example "Covariates from AnnData or a CSV"
    With AnnData, provide column names from `adata.obs`:

    ```python
    data = load_screen(
        data_dir / "screen.h5ad",
        covariates=["batch", "percent_mito"],
    )
    ```

    With DataFrames, provide an aligned covariate table:

    ```python
    covariates = pd.read_csv(data_dir / "covariates.csv", index_col=0)
    covariates["batch"] = covariates["batch"].astype("category")

    data = PerturbData(
        X=expression,
        G=G,
        covariates=covariates,
    )
    ```

If a covariate is confounded with perturbation, correcting for it can also
remove the perturbation signal. Skip correction if it has already been done.

## 3. Dataset examples

Choose an example that matches your screen, then continue to
[fit and save](#4-fit-and-save). Each example creates `data` and `result_dir`.

Download the Datlinger, Adamson, or Norman H5AD files from
[scPerturb](https://github.com/sanderlab/scPerturb) into `data_dir`.

??? example "Preprocess raw counts"
    Install Scanpy and define this function before running an example:

    ```bash
    uv pip install scanpy
    ```

    ```python
    import anndata as ad
    import scanpy as sc

    def transform_counts(
        adata: ad.AnnData,
        n_top_genes: int = 6_000,
        min_genes: int = 200,
        min_cells: int = 3,
        max_pct_mt: float = 20.0,
    ) -> ad.AnnData:
        adata = adata.copy()
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        sc.pp.filter_cells(adata, min_genes=min_genes)
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt].copy()
        sc.pp.filter_genes(adata, min_cells=min_cells)

        sc.experimental.pp.highly_variable_genes(
            adata,
            flavor="pearson_residuals",
            n_top_genes=min(n_top_genes, adata.n_vars),
            subset=True,
        )
        sc.experimental.pp.normalize_pearson_residuals(adata)
        return adata
    ```

    This filters cells and genes, selects up to 6,000 genes by Pearson-residual
    variance, and replaces `adata.X` with Pearson residuals. Adjust QC thresholds
    for your screen. The `MT-` rule assumes human gene symbols in `var_names`.

??? example "Datlinger CROP-seq: one target per cell"
    In the [Datlinger dataset](https://www.nature.com/articles/nmeth.4177),
    missing target labels identify controls. Replicate is categorical.

    ```python
    adata = ad.read_h5ad(data_dir / "DatlingerBock2017.h5ad")
    adata.obs["target"] = adata.obs["target"].astype("string").fillna("control")
    adata.obs["replicate"] = adata.obs["replicate"].astype("category")
    adata = transform_counts(adata)

    adata.obsm["G"] = adata.obs["target"].str.get_dummies().astype(int)

    data = load_screen(
        adata,
        control="control",
        covariates=["replicate", "percent_mito"],
    )

    result_dir = Path("results/datlinger")
    ```

??? example "Adamson CRISPRi: collapse guides to target genes"
    The [Adamson dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC5315571/)
    stores labels such as `CREB1_pDS269`. Use the text before `_` as the target;
    `62(mod)` denotes controls and `*` labels are excluded.

    ```python
    adata = ad.read_h5ad(
        data_dir / "AdamsonWeissman2016_GSM2406675_10X001.h5ad"
    )
    labels = adata.obs["perturbation"].astype("string").str.split("_", n=1).str[0]
    adata.obs["target"] = labels.replace({"62(mod)": "control"}).mask(labels == "*")
    adata = adata[adata.obs["target"].notna()].copy()
    adata = transform_counts(adata)

    adata.obsm["G"] = adata.obs["target"].str.get_dummies().astype(int)
    data = load_screen(
        adata,
        control="control",
        covariates=["percent_ribo"],
    )

    result_dir = Path("results/adamson")
    ```

??? example "Norman CRISPRa: single targets and target pairs"
    In the [Norman dataset](https://doi.org/10.1126/science.aax4438), split
    target-pair labels such as `CEBPE_RUNX1T1` into two active columns.
    Controls have all-zero rows, as in the table above. This models additive
    target effects; it does not add interaction terms.

    ```python
    adata = ad.read_h5ad(data_dir / "NormanWeissman2019_filtered.h5ad")
    adata.obs["gemgroup"] = adata.obs["gemgroup"].astype("category")
    adata = transform_counts(adata)

    labels = adata.obs["perturbation"].astype("string")
    G = labels.mask(labels == "control", "").str.get_dummies(sep="_").astype("int8")
    assert (G.sum(axis=1).to_numpy() == adata.obs["nperts"].to_numpy()).all()

    data = PerturbData(
        X=adata.X,
        G=G,
        gene_names=adata.var_names,
        covariates=adata.obs[["gemgroup", "percent_mito"]],
    )

    result_dir = Path("results/norman")
    ```

??? example "A375 10x CRISPR: count matrix and a separate calls table"
    This example uses a control/RAB1A subset. Prepare a barcode-indexed TSV
    containing `perturbation`, `log_total_counts`, and `percent_mito`.
    The H5 file supplies counts; final perturbation calls come from your
    upstream calling pipeline.

    ```python
    matrix = sc.read_10x_h5(
        data_dir / "a375_1k_filtered_feature_bc_matrix.h5",
        gex_only=False,
    )
    adata = matrix[:, matrix.var["feature_types"] == "Gene Expression"].copy()
    adata.var_names_make_unique()

    calls = pd.read_csv(data_dir / "a375_10x_h5_metadata.tsv", sep="\t", index_col=0)
    calls.index = calls.index.astype(str)
    calls = calls[calls["perturbation"].isin(["control", "RAB1A"])]
    adata = adata[calls.index].copy()
    adata.obs = adata.obs.join(calls)
    adata.obs["perturbation"] = adata.obs["perturbation"].astype("category")
    adata = transform_counts(adata)

    adata.obsm["G"] = (adata.obs["perturbation"] == "RAB1A").astype(int).to_frame(name="RAB1A")
    data = load_screen(
        adata,
        covariates=["log_total_counts", "percent_mito"],
    )

    result_dir = Path("results/a375")
    ```

## 4. Fit and save

Fit the model using the `data` prepared above:

```python
fit = fit_screen(
    data,
    z_dim=20,
    l_dim=1000,
    init="pca"
)

save_results(fit, result_dir)
```

| Setting | Meaning |
|---|---|
| `z_dim` | Number of latent factors |
| `l_dim` | Number of single-effect loading components per factor |
| `init` | Initialization: `"pca"` or `"random"` |

Adjust the number of factors and loading components for your analysis.
Genes are centered automatically; set `standardize=True` to also scale each
gene to unit variance.

See the [API reference](api.md#covariates-and-fitting) for all fitting options.

??? example "Fit from the command line"
    For an H5AD file with transformed expression in `X` and perturbation
    assignments in `obsm["G"]`:

    ```bash
    perturbvi fit data/screen.h5ad --output results --z-dim 20 --l-dim 1000
    ```

    Add `--control Nontargeting` if that reference column should be dropped.
    Run `perturbvi fit --help` for layer and covariate options.

??? note "Reuse covariate-corrected expression across fits"
    When fitting the same data several times, you can correct expression once:

    ```python
    from perturbvi import residualize_screen

    resid = residualize_screen(data)

    fit = fit_screen(
        resid,
        z_dim=20,
        l_dim=1000,
        init="pca"
    )
    ```

## 5. Plot and interpret results

The [LUHMES notebook](luhmes.ipynb) shows how to plot and interpret the results,
count significant genes, and run GO enrichment.
