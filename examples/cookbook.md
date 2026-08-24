# PerturbVI cookbook

## 1. About these examples

This cookbook applies PerturbVI to real genetic perturbation screens. The
[Workflow](workflow.md) is the main explanation of `X`, `G`, names,
covariates, fitting, saving, and analysis. Here, each example focuses on the
dataset-specific work needed to create `PerturbData`.

The examples cover:

- processed LUHMES CROP-seq tables from the GSFA authors;
- Datlinger CROP-seq with one target label per cell;
- Adamson CRISPRi with guide identifiers collapsed to target genes;
- Norman CRISPRa with controls, single targets, and target pairs;
- A375 10x CRISPR with a barcode-matched calls table.

Raw sequencing output is not a PerturbVI input. RNA counts must be processed,
and guide measurements must already have been converted into final guide or
target calls. For the AnnData layout of a prepared screen, see
[Input structure](input_structure.md).

## 2. A basic raw-count transformation

The downloaded [scPerturb](https://github.com/sanderlab/scPerturb) H5AD files
used below contain raw RNA counts and published perturbation labels.

The function below performs basic Scanpy QC, selects genes by Pearson-residual
variance, and replaces `adata.X` with analytic Pearson residuals. It is a
compact example, not a complete QC policy for every experiment.

```python
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

from perturbvi import PerturbData, load_screen, residualize_screen


def transform_counts(
    adata: ad.AnnData,
    n_top_genes: int = 6_000,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mt: float = 20.0,
) -> ad.AnnData:
    adata = adata.copy()
    # Scanpy reads and writes the expression matrix in adata.X by default.
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

Pearson-residual normalization expects raw counts. The transformed values stay
in `adata.X`, which is the matrix that `load_screen()` reads by default. This
function does not call guides or alter the published perturbation labels.

## 3. Real genetic screens

### 3.1 LUHMES CROP-seq: processed X and G

The [LUHMES study](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE142078)
targeted neurodevelopmental genes with CRISPR knockdown in neural progenitor
cells. This example uses the processed matrices from the
[GSFA LUHMES workflow](https://xinhe-lab.github.io/GSFA_paper/preprocess_and_gsfa_LUHMES.html),
not the raw Cell Ranger files.

```python
expression = pd.read_csv("luhmes_exp.csv", index_col=0)
G = pd.read_csv("luhmes_G.csv", index_col=0).drop(columns="Nontargeting")
gene_names = (
    pd.read_csv("luhmes_gene_symbol.csv", header=None)
    .iloc[:, 0]
    .astype(str)
    .tolist()
)

data = PerturbData(
    X=expression.set_axis(gene_names, axis="columns"),
    G=G,
)

output = Path("work/output/cookbook/luhmes")
```

The expression table was already transformed, selected, corrected, and scaled
in the published workflow. The non-targeting column is removed from `G`, so
control cells have all-zero rows. No covariates are passed because the
published expression table was already corrected.

### 3.2 Datlinger CROP-seq: one target per cell

The [Datlinger CROP-seq study](https://www.nature.com/articles/nmeth.4177)
provides a final target gene for each perturbed cell. Missing targets identify
published controls. These are ten real rows from
`DatlingerBock2017.h5ad` before the example's QC:

```text
cell          perturbation          perturbation_2  replicate  target  ncounts  ngenes  percent_mito  nperts
TACTTGACCCCN  control               stimulated          1  NaN        8696    2722          0.29       1
TTACAGCTGAAC  Tcrlibrary_JUND_2     stimulated          1  JUND       3198    1581          5.07       3
CTAAGGCCCTTA  Tcrlibrary_BACH2_3    stimulated          1  BACH2      8137    2856          4.35       3
CTTGACGCAGGT  Tcrlibrary_NFKB2_3    stimulated          1  NFKB2      7051    2687          9.03       3
TAACCCGTACGC  Tcrlibrary_JUN_1      stimulated          1  JUN        5453    2122         12.14       3
ATCTAGATACNN  control               stimulated          1  NaN        2465    1336          2.84       1
CTATCGTTCTTN  Tcrlibrary_NFKB1_1    stimulated          1  NFKB1      1033     589         22.65       3
GTATTGCGAGCN  Tcrlibrary_JUND_3     stimulated          1  JUND      27397    5618          3.15       3
GTACTGTGTTAN  Tcrlibrary_JUNB_1     stimulated          1  JUNB      14008    3739          4.41       3
CGTCTTTCANNN  Tcrlibrary_NFKB2_2    stimulated          1  NFKB2      6491    2726          5.18       3
```

Replace missing targets with the control label, transform expression, and store
the one-hot target matrix at `obsm["G"]`; `control=` tells the loader to drop
the reference column. This example treats replicate and mitochondrial
percentage as covariates:

```python
adata = ad.read_h5ad(Path("work/data/raw/DatlingerBock2017.h5ad"))
adata.obs["target"] = adata.obs["target"].astype("string").fillna("control")
adata.obs["replicate"] = adata.obs["replicate"].astype("category")
adata = transform_counts(adata)

adata.obsm["G"] = adata.obs["target"].str.get_dummies().astype(int)
data = load_screen(
    adata,
    control="control",
    covariates=["replicate", "percent_mito"],
)
data = residualize_screen(data)

output = Path("work/output/cookbook/datlinger_2017")
```

The target labels map to binary columns like this. Cells removed by the chosen
QC are not included in the fitted `G`:

```text
cell          G[JUND]  G[BACH2]  G[NFKB2]  G[JUN]  G[NFKB1]  G[JUNB]
TACTTGACCCCN        0         0         0       0         0        0
TTACAGCTGAAC        1         0         0       0         0        0
CTAAGGCCCTTA        0         1         0       0         0        0
CTTGACGCAGGT        0         0         1       0         0        0
TAACCCGTACGC        0         0         0       1         0        0
ATCTAGATACNN        0         0         0       0         0        0
CTATCGTTCTTN        0         0         0       0         1        0
GTATTGCGAGCN        1         0         0       0         0        0
GTACTGTGTTAN        0         0         0       0         0        1
CGTCTTTCANNN        0         0         1       0         0        0
```

### 3.3 Adamson CRISPRi: guides collapsed to targets

The [Adamson Perturb-seq study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5315571/)
used CRISPR interference in K562 cells. The downloaded file stores a target
and guide identifier together in `obs["perturbation"]`:

```text
cell            perturbation      read count  UMI count  ncounts  ngenes  percent_mito  percent_ribo
AAACATACACCGAT  CREB1_pDS269            1286         98     8138    2412          0.00         34.04
AAACATACAGAGAT  SNAI1_pDS266             296         19     8980    2386          0.00         40.01
AAACATACCAGAAA  62(mod)_pBA581           1829        162    28610    4404          0.00         40.00
AAACATACGTTGAC  EP300_pDS268            1580         98    11346    2815          0.00         35.18
AAACATACTGTTCT  62(mod)_pBA581            748         51     9864    2584          0.00         35.82
AAACCGTGCAGCTA  ZNF326_pDS262             789         55    10470    2575          0.00         36.69
AAACCGTGCCTGAA  62(mod)_pBA581            802         56    10649    2660          0.00         40.14
AAACCGTGCGGAGA  62(mod)_pBA581            300         18     9293    2769          0.00         27.29
AAACCGTGGAACTC  CREB1_pDS269              275         17     3857    1369          0.00         38.89
AAACCGTGTGGCAT  62(mod)_pBA581           1275         81    11813    2689          0.00         38.62
```

Use the text before `_` as the target. In this file, `62(mod)` is the control
and `*` labels are excluded. This example treats ribosomal percentage as a
covariate:

```python
adata = ad.read_h5ad(
    Path("work/data/raw/AdamsonWeissman2016_GSM2406675_10X001.h5ad")
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
data = residualize_screen(data)

output = Path("work/output/cookbook/adamson_2016")
```

The first ten target labels produce this excerpt from `G`:

```text
cell            G[CREB1]  G[SNAI1]  G[EP300]  G[ZNF326]
AAACATACACCGAT         1         0         0          0
AAACATACAGAGAT         0         1         0          0
AAACATACCAGAAA         0         0         0          0
AAACATACGTTGAC         0         0         1          0
AAACATACTGTTCT         0         0         0          0
AAACCGTGCAGCTA         0         0         0          1
AAACCGTGCCTGAA         0         0         0          0
AAACCGTGCGGAGA         0         0         0          0
AAACCGTGGAACTC         1         0         0          0
AAACCGTGTGGCAT         0         0         0          0
```

Multiple guide identifiers can therefore contribute cells to the same target
column.

### 3.4 Norman CRISPRa: single targets and target pairs

The [Norman study](https://doi.org/10.1126/science.aax4438) used a dual-sgRNA
CRISPRa library in K562 cells. The downloaded H5AD contains controls,
single-target cells, and target-pair cells:

```text
cell              guide_id                                      gemgroup  perturbation    nperts  ncounts  ngenes  percent_mito
AGACGTTGTCTAGCGC  NegCtrl10_NegCtrl0;NegCtrl10_NegCtrl0                2  control              0    21172    4312          7.69
ACGGAGACACACCGCA  NegCtrl0_CEBPE;NegCtrl0_CEBPE                        7  CEBPE                1    11928    2770          5.53
ACCAGTATCGGAGGTA  NegCtrl0_RUNX1T1;NegCtrl0_RUNX1T1                    2  RUNX1T1              1    12984    3067          7.23
AACTCAGAGACTGGGT  CEBPE_RUNX1T1;CEBPE_RUNX1T1                          5  CEBPE_RUNX1T1        2    22491    4026          3.06
ATGAGGGCATTGGCGC  SET_NegCtrl0;SET_NegCtrl0                            2  SET                  1    16385    3501          3.75
GTACTTTTCAGTTGAC  KLF1_NegCtrl0;KLF1_NegCtrl0                          7  KLF1                 1    62356    6515          6.66
CCTTCCCTCCGTCATC  SET_KLF1;SET_KLF1                                    4  SET_KLF1             2    38454    5385          4.34
CGAGAAGTCTTGAGGT  CBL_NegCtrl0;CBL_NegCtrl0                            5  CBL                  1    20423    4005          3.36
GTTACAGCAATGTTGC  CNN1_NegCtrl0;CNN1_NegCtrl0                          8  CNN1                 1    21875    4138          5.55
TGGCTGGTCTTGAGAC  CBL_CNN1;CBL_CNN1                                    5  CBL_CNN1             2    18730    3981          5.03
```

Here `G` models target presence. Split the published target-pair labels at
`_`, while controls remain all-zero rows. This example treats gem group and
mitochondrial percentage as covariates:

```python
adata = ad.read_h5ad(Path("work/data/raw/NormanWeissman2019_filtered.h5ad"))
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
data = residualize_screen(data)

output = Path("work/output/cookbook/norman_2019")
```

The same label conversion gives controls zero active columns, single targets
one, and target pairs two:

```text
cell              G[CEBPE]  G[RUNX1T1]  G[SET]  G[KLF1]  G[CBL]  G[CNN1]
AGACGTTGTCTAGCGC         0           0       0        0       0        0
ACGGAGACACACCGCA         1           0       0        0       0        0
ACCAGTATCGGAGGTA         0           1       0        0       0        0
AACTCAGAGACTGGGT         1           1       0        0       0        0
ATGAGGGCATTGGCGC         0           0       1        0       0        0
GTACTTTTCAGTTGAC         0           0       0        1       0        0
CCTTCCCTCCGTCATC         0           0       1        1       0        0
CGAGAAGTCTTGAGGT         0           0       0        0       1        0
GTTACAGCAATGTTGC         0           0       0        0       0        1
TGGCTGGTCTTGAGAC         0           0       0        0       1        1
```

PerturbVI uses this additive target matrix; it does not reproduce the original
paper's separate genetic-interaction regression.

### 3.5 A375 10x CRISPR: H5 matrix plus separate calls

The 10x H5 file contains feature counts and barcodes. It does not contain an
AnnData `obs` table or confident biological calls. The companion calls table
below is the explicit upstream Cell Ranger call result supplied with this
example; it matches all 48 retained barcodes. Its first ten rows are:

```text
cell                  perturbation  log_total_counts  percent_mito
AAGCAAATCAGAGAAC-1    control                 7.754          4.376
AAGCCAACACCACTAG-1    control                 9.932          4.009
AAGCTCATCACAACGC-1    control                 9.691          0.946
AATCCTCTCATTATCT-1    RAB1A                   10.392          3.834
ACAAGAGGTCGTAATA-1    control                 9.036          4.022
ACAATTATCCCCATGC-1    control                 9.401          5.163
ACACGGTTCAGTTGGC-1    control                10.227          3.966
ACACGGTTCCTTCGCA-1    control                 9.873          4.493
ACCACATCACCACGGT-1    control                10.079          4.262
ACCTACGCAACACGGG-1    control                 9.820          3.647
```

Read the 10x matrix and the separate calls table with Scanpy and pandas, keep
only gene-expression features, and join the calls by barcode. Scanpy is used
here as an external file reader; `load_screen()` still receives an AnnData
object. If you have only the H5 matrix, this step cannot construct `G`: you
must first obtain confident guide or target calls from Cell Ranger or another
documented calling method.

```python
matrix = sc.read_10x_h5(
    Path("work/data/raw/a375_1k_filtered_feature_bc_matrix.h5"),
    gex_only=False,
)
adata = matrix[:, matrix.var["feature_types"] == "Gene Expression"].copy()
adata.var_names_make_unique()

calls = pd.read_csv(
    Path("work/data/prepared/a375_10x_h5_metadata.tsv"),
    sep="\t",
    index_col=0,
)
# These are already called perturbations, not guide-count thresholds.
calls.index = calls.index.astype(str)
unknown = calls.index.difference(adata.obs_names)
if len(unknown):
    raise ValueError(f"Calls contain unknown barcodes: {unknown[:5].tolist()}")

adata = adata[calls.index].copy()
adata.obs = adata.obs.join(calls)
adata.obs["perturbation"] = adata.obs["perturbation"].astype("category")
# transform_counts transforms the gene-expression matrix in adata.X only.
adata = transform_counts(adata)

adata.obsm["G"] = (adata.obs["perturbation"] == "RAB1A").astype(int).to_frame()
data = load_screen(
    adata,
    covariates=["log_total_counts", "percent_mito"],
)
data = residualize_screen(data)

output = Path("work/output/cookbook/a375_10x")
```

The resulting `G` has one modeled target column, `RAB1A`:

```text
cell                  G[RAB1A]
AAGCAAATCAGAGAAC-1           0
AAGCCAACACCACTAG-1           0
AAGCTCATCACAACGC-1           0
AATCCTCTCATTATCT-1           1
ACAAGAGGTCGTAATA-1           0
ACAATTATCCCCATGC-1           0
ACACGGTTCAGTTGGC-1           0
ACACGGTTCCTTCGCA-1           0
ACCACATCACCACGGT-1           0
ACCTACGCAACACGGG-1           0
```

This preview is sorted by barcode, not by perturbation, so it happens to show
mostly controls. In the complete 48-cell subset there are 32 control cells and
16 RAB1A cells. Because the control is the reference rather than a modeled
column, `G[RAB1A] = 0` means control and `G[RAB1A] = 1` means RAB1A.

Do not threshold the guide-capture counts in the H5 file and call the result
`G` without an explicit calling method. Here, `G` comes from the matched calls
table, while expression comes from the 10x gene-expression features.

## 4. Fit, save, and analyze an example

Each dataset section defines `data` and `output`. After running one of them:

```python
from perturbvi import fit_screen, save_results
from perturbvi.utils import analyze

fit = fit_screen(
    data,
    z_dim=12,
    l_dim=100,
    tau=800.0,
    max_iter=500,
    seed=1,
    verbose=True,
)

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

The [Workflow](workflow.md#5-save-and-analyze) explains what each stage does,
and the [API](api.md) lists every model setting and result table.
