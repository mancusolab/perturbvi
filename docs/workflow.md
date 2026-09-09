# Using PerturbVI with Your Data

Prepare expression and perturbation assignments, fit latent factors, then read
the results with pandas or plot them. This guide covers CSVs, AnnData, and
common perturbation designs. For an analysis with figures, see
[LUHMES Analysis with PerturbVI](luhmes.md).

## 1. Set up

```bash
uv pip install perturbvi
```

```python
from pathlib import Path

import pandas as pd
from perturbvi import PerturbData, fit_screen, load_screen, save_results

data_dir = Path("data")
result_dir = Path("results/my_screen")
```

## 2. Prepare the inputs

| Input | Rows × columns | Contents |
|---|---|---|
| `X` | Cells × genes | Transformed expression |
| `G` | Cells × perturbations | Binary target or guide assignments |
| `covariates`, optional | Cells × covariates | Variables to regress out of expression |

All inputs must contain the same cells in the same order. DataFrame columns
supply gene and perturbation names. Use unique identifiers; gene IDs should
also match the annotations you plan to use downstream.

PerturbVI centers each gene. It does not perform raw-count QC, normalization,
gene selection, or guide calling. Prepare those upstream, for example with
Scanpy and your experiment's guide-calling pipeline.

### Controls and multiple perturbations

Each column of `G` represents one modeled perturbation. A cell can have more
than one active column:

| Cell assignment | `G[CEBPE]` | `G[RUNX1T1]` |
|---|---:|---:|
| Control | 0 | 0 |
| CEBPE | 1 | 0 |
| RUNX1T1 | 0 | 1 |
| CEBPE + RUNX1T1 | 1 | 1 |

If `G` contains a separate control column, pass its name as
`control="Nontargeting"` to drop that column. Control cells remain in the data.
Omit `control` when controls already have all-zero rows. Omitting it when a
control column exists retains that column as a modeled condition, as in the
[LUHMES example](luhmes.md).

### CSV or TSV

Put cell identifiers in the first column and gene or perturbation names in
the header. For TSV files, add `sep="\t"` to the reads.

```python
expression = pd.read_csv(data_dir / "expression.csv", index_col=0)
G = pd.read_csv(data_dir / "perturbations.csv", index_col=0)
data = PerturbData(
    X=expression,
    G=G,
)
```

??? example "NumPy or sparse arrays"
    Arrays have no column labels, so supply both name lists explicitly.

    ```python
    data = PerturbData(
        X=expression_array,
        G=G_array,
        gene_names=gene_names,
        perturbation_names=perturbation_names,
    )
    ```

### AnnData

Use this layout for an AnnData object, H5AD file, or AnnData Zarr folder:

| Location | Contents |
|---|---|
| `adata.X` | Transformed expression; alternatively use a named layer |
| `adata.obs_names` | Cell identifiers |
| `adata.var_names` | Gene identifiers |
| `adata.obsm["G"]` | Binary pandas DataFrame, indexed by cell and with named columns |
| `adata.obs`, optional | Covariate columns |

```python
data = load_screen(
    data_dir / "screen.h5ad",
)
```

Pass an AnnData object instead of a path if it is already loaded. Use
`x_key="transformed"` for `adata.layers["transformed"]`, or
`g_key="perturbations"` if `G` is stored under another `obsm` key.

??? example "Build G from one target label per cell"
    Here `adata.obs["target"]` contains target names and the label `control`.

    ```python
    adata.obsm["G"] = adata.obs["target"].str.get_dummies().astype(int)
    data = load_screen(
        adata,
        control="control",
    )
    ```

### Covariates

Pass the variables you want removed from expression. `fit_screen()` regresses
them out before fitting; a separate correction call is unnecessary.
Numeric columns are continuous covariates. Text, categorical, and boolean
columns are treated as groups. Convert numeric batch labels to `category`.

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

Choose covariates deliberately: a variable confounded with perturbation can
remove the signal of interest. Skip this step if expression is already corrected.

## 3. Dataset examples

These optional recipes replace the input-loading step above. Each creates
`data` and selects a `result_dir`; then continue to [fitting](#4-fit-and-save).
The Datlinger, Adamson, and Norman examples use H5AD files distributed by
[scPerturb](https://github.com/sanderlab/scPerturb). Put them in `data_dir`.

??? example "Raw-count preprocessing used by these recipes"
    Install Scanpy, then run this setup before any of the four recipes:

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

After choosing one input route:

```python
fit = fit_screen(
    data,
    z_dim=12,
    l_dim=100,
    init="pca",
    tau=100,
    max_iter=500,
    seed=0,
)
save_results(
    fit,
    result_dir,
)
```

| Setting | Meaning |
|---|---|
| `z_dim` | Number of latent factors |
| `l_dim` | Number of single-effect loading components per factor |
| `init` | Initialization: `"pca"` or `"random"` |
| `tau` | Initial expression noise precision, updated during fitting |

These are example settings, not values selected for every dataset.
Fitting centers genes; use `standardize=True` to also scale them to unit
variance. See the [API](api.md#covariates-and-fitting) for all settings.

??? example "Fit from the command line"
    For a prepared H5AD with transformed expression in `X` and named `obsm["G"]`:

    ```bash
    perturbvi fit data/screen.h5ad --output results/my_screen --z-dim 12 --l-dim 100 --tau 100
    ```

    Add `--control Nontargeting` if that reference column should be dropped.
    Run `perturbvi fit --help` for layer and covariate options.

??? note "Reuse covariate-corrected expression across fits"
    If you supplied covariates and plan several fits, correct expression once:

    ```python
    from perturbvi import residualize_screen

    corrected_data = residualize_screen(data)
    fit = fit_screen(
        corrected_data,
        z_dim=12,
        l_dim=100,
    )
    ```

## 5. Read and plot results

Install Matplotlib for plotting:

```bash
uv pip install matplotlib
```

Saving writes `model.pkl` and six labeled CSVs:

| File | Rows × columns | Contents |
|---|---|---|
| `W.csv` | Factors × genes | Inclusion-weighted mean loadings |
| `PIP_W.csv` | Factors × genes | Loading inclusion probabilities |
| `B.csv` | Perturbations × factors | Inclusion-weighted mean effects on factors |
| `PIP_B.csv` | Perturbations × factors | Effect inclusion probabilities |
| `BW.csv` | Perturbations × genes | Overall effects, `B @ W` |
| `PVE.csv` | Factors × 1 | Expression variance summary |

Effect matrices describe magnitude and direction; PIP matrices describe
inclusion probability. Read only the matrix you need:

```python
from perturbvi import plotting as pp

B = pd.read_csv(result_dir / "B.csv", index_col=0)
fig = pp.plot_factor_effects(
    B,
    show_significance=False,
    scale="asinh",
)
fig.savefig(
    result_dir / "factor_effects.png",
    dpi=300,
)
```

After fitting in Python, `fit.B`, `fit.W`, and `fit.BW` are already DataFrames.
You can pass them directly to plotting functions. The
[LUHMES analysis](luhmes.md#5-perturbation-effects-on-factors) demonstrates
subsets, gene annotations, gene heatmaps, and enrichment.

## 6. Optional overall-effect significance

LFSR measures uncertainty in the sign of an overall gene effect. Compute it
when you need DEG counts or significance dots; it is not needed for effect
heatmaps or factor enrichment.

```python
from perturbvi import estimate_lfsr

LFSR_BW = estimate_lfsr(
    result_dir,
    draws=2000,
    seed=2026,
)
LFSR_BW.to_csv(result_dir / "LFSR_BW.csv")
```

For an existing LFSR file, read it directly:

```python
LFSR_BW = pd.read_csv(result_dir / "LFSR_BW.csv", index_col=0)
deg_counts = (LFSR_BW < 0.05).sum(axis=1)
```

??? example "Compute LFSR from the command line"
    ```bash
    perturbvi lfsr results/my_screen --draws 2000 --seed 2026
    ```

    This writes `LFSR_BW.csv` in the saved result directory.
