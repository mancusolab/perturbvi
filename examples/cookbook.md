# PerturbVI cookbook

These examples use one function for every PerturbVI run. Each dataset section
gives its biological background, download source, and runnable analysis.

## Using the examples

PerturbVI uses a cell-by-gene expression matrix `X` and a matching
cell-by-perturbation design matrix `G`. Controls are all-zero rows in `G`;
combination perturbations have more than one active column.

`load_screen` can build `G` from a perturbation label in AnnData `obs` or read a
prepared matrix from `obsm`. For 10x data, it aligns expression with prepared
assignments by barcode. Covariates are optional and are used only when
residualization is requested.

## Shared function

Define this function once before running any of the dataset examples.

```python
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import pandas as pd

from perturbvi import analyze, fit_screen, load_screen, residualize_screen, save_results


def run_perturbvi(
    source: Any,
    output_dir: Union[str, Path],
    *,
    load_kwargs: Mapping[str, Any],
    categorical_covariates: Sequence[str] = (),
    z_dim: int = 3,
    l_dim: int = 2,
    tau: float = 1.0,
    max_iter: int = 10,
    seed: int = 1,
    verbose: bool = False,
):
    """Load, residualize, fit, save, reload, and analyze one screen."""
    loader_options = dict(load_kwargs)
    covariates = list(loader_options.get("covariates") or [])
    categorical_covariates = list(categorical_covariates)
    unknown = [name for name in categorical_covariates if name not in covariates]
    if unknown:
        raise ValueError(f"Categorical covariates must also be loaded as covariates: {unknown}")

    if verbose:
        print(f"input_type: {type(source).__name__}")
        print(f"load_kwargs: {loader_options}")
    screen = load_screen(source, **loader_options)
    if verbose:
        print(f"loaded_X_shape: {screen.X.shape}")
        print(f"loaded_G_shape: {screen.G.shape}")
        print(f"loaded_covariates: {screen.covariate_names}")
        print(f"source_details: {screen.source}")
    if covariates:
        screen = residualize_screen(
            screen,
            covariates=covariates,
            categorical_covariates=categorical_covariates,
        )
        if verbose:
            print(f"residualized_covariates: {covariates}")
            print(f"categorical_covariates: {categorical_covariates}")

    results = fit_screen(
        screen,
        z_dim=z_dim,
        l_dim=l_dim,
        tau=tau,
        max_iter=max_iter,
        seed=seed,
        verbose=verbose,
    )

    output = Path(output_dir)
    save_results(
        results,
        str(output),
        gene_names=screen.gene_names,
        perturbation_names=screen.perturbation_names,
    )
    tables = analyze(str(output))
    analysis_dir = output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(analysis_dir / f"{name}.csv")
            if verbose:
                print(f"analysis_table: {name}, shape={table.shape}")

    return screen, results, tables
```

The defaults above keep these examples short. Set `z_dim`, `l_dim`,
`max_iter`, and `tau` for the analysis you intend to run.

## Preparing raw expression

If `adata.X` contains raw counts, the following code calculates two commonly
used technical covariates and creates a log-normalized expression matrix.

```python
import numpy as np
import scanpy as sc

total_counts = np.asarray(adata.X.sum(axis=1)).reshape(-1)
mitochondrial = adata.var_names.astype(str).str.upper().str.startswith("MT-")
mitochondrial_counts = np.asarray(adata[:, mitochondrial].X.sum(axis=1)).reshape(-1)

adata.obs["log_total_counts"] = np.log1p(total_counts)
adata.obs["percent_mito"] = np.divide(
    mitochondrial_counts * 100,
    total_counts,
    out=np.zeros_like(total_counts, dtype=float),
    where=total_counts > 0,
)

sc.pp.normalize_total(adata, target_sum=10_000)
sc.pp.log1p(adata)
```

Do not repeat this step when the selected expression matrix is already
normalized. Perturbation labels and controls must come from the experiment's
prepared assignments. Change the mitochondrial-gene mask if the annotation
does not use `MT-` symbols.

Download each dataset from the source linked in its section. Set `DATA` to the
directory containing your prepared files and change the example filenames if
needed. Results are written beneath `OUTPUT`.

```python
from pathlib import Path

DATA = Path("/path/to/your/data")
OUTPUT = Path("perturbvi_results")
```

## LUHMES CSV fit and analysis

This human neuronal differentiation screen used dCas9-based transcriptional
repression to study autism and neurodevelopmental disease genes. LUHMES cells
received a library of 47 sgRNAs targeting 14 genes or serving as non-targeting
controls. Low-multiplicity transduction enriched for one perturbation per cell;
the cells were then differentiated for eight days and profiled with 10x
Chromium. CROP-seq recovered the sgRNA assignments from polyadenylated RNA.

The data are available from GEO as
[GSE142078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE142078). The
experiment is described by [Lalli et al., Genome Research
2020](https://pubmed.ncbi.nlm.nih.gov/32887689/).

This workflow uses a cell-by-gene expression table, a cell-by-guide table, and
a one-column gene-symbol file. `Nontargeting` is the control column in
`luhmes_G.csv`; PerturbVI removes that column from `G` and treats its cells as
controls. The expression and guide tables must use the same unique cell names.

```bash
perturbvi fit luhmes_exp.csv \
  --guide-matrix luhmes_G.csv \
  --control-label Nontargeting \
  --z-dim 12 \
  --l-dim 400 \
  --tau 800 \
  --output results \
  --verbose

perturbvi analyze results \
  --gene-names luhmes_gene_symbol.csv \
  --pip-threshold 0.9 \
  --lfsr-threshold 0.05 \
  --compute-lfsr \
  --verbose
```

The fit is saved under `results`, including `params_file.pkl`. The analysis
tables are saved under `results/analysis`.

```python
from pathlib import Path

import pandas as pd

analysis = Path("results/analysis")

pip_summary = pd.read_csv(analysis / "pip_summary.csv", index_col=0)
num_deg_per_w = pip_summary.set_index("factor")["n_pip_significant"]

lfsr_hits = pd.read_csv(analysis / "lfsr_significant.csv", index_col=0)
perturbations = pd.read_csv(analysis / "beta.csv", index_col=0).index
num_deg_per_perturbed_gene = (
    lfsr_hits.groupby("perturbation")
    .size()
    .reindex(perturbations, fill_value=0)
)

print(num_deg_per_w)
print(num_deg_per_perturbed_gene)
```

## scPerturb datasets

[scPerturb](https://www.sanderlab.org/scPerturb/) distributes harmonized H5AD
files with expression in `X` and perturbation metadata in `obs`. The examples
below use its published assignments; no guide calls are inferred by PerturbVI.
The covariates and experimental designs differ, so each has its own short
block.

### Adamson et al. 2016 Perturb-seq

This pilot Perturb-seq experiment used CRISPR interference in K562 leukemia
cells to measure the transcriptional effects of repressing seven transcription
factors. Each cell has an assigned sgRNA target or the negative-control label.
The experiment was part of the study that introduced Perturb-seq and used the
method to dissect the mammalian unfolded protein response.

The source file, `AdamsonWeissman2016_GSM2406675_10X001.h5ad`, comes from the
[scPerturb collection on Zenodo](https://zenodo.org/records/7278143).
The experiment is described in the original [Cell
paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5315571/).

#### H5AD

This file has one perturbation label per cell and one numeric covariate.

```python
import anndata as ad

path = DATA / "adamson_2016.h5ad"
adata = ad.read_h5ad(path, backed="r")
print(adata.shape)
print(adata.obs["perturbation"].value_counts().head())
print(adata.obs[["log_total_counts"]].describe())
adata.file.close()

screen, results, tables = run_perturbvi(
    path,
    OUTPUT / "adamson_h5ad",
    load_kwargs={
        "guide_key": "perturbation",
        "control_label": "control",
        "covariates": ["log_total_counts"],
    },
)
```

`log_total_counts` is numeric, so it is not listed under
`categorical_covariates`.

#### CSV

The same Adamson cells are also exported as separate expression, guide, and
metadata tables. All three tables must contain the same cells in the same
order. Controls are the all-zero rows of `guides.csv`.

```python
import pandas as pd

directory = DATA / "delimited"
expression_path = directory / "expression.csv"
guide_path = directory / "guides.csv"
metadata_path = directory / "metadata.csv"

expression = pd.read_csv(expression_path, index_col=0)
guides = pd.read_csv(guide_path, index_col=0)
metadata = pd.read_csv(metadata_path, index_col=0)

assert expression.index.equals(guides.index)
assert expression.index.equals(metadata.index)
print(expression.shape, guides.shape)
print(guides.sum().sort_values(ascending=False).head())
print(metadata.describe())

screen, results, tables = run_perturbvi(
    expression_path,
    OUTPUT / "adamson_csv",
    load_kwargs={
        "format": "csv",
        "guide_path": str(guide_path),
        "metadata_path": str(metadata_path),
        "covariates": ["log_total_counts"],
    },
)
```

### Datlinger et al. 2017 CROP-seq

This CROP-seq experiment used CRISPR knockout in Jurkat T cells to study
regulators of T-cell receptor signaling. Single-cell RNA sequencing linked each
guide to its transcriptional effect in stimulated and unstimulated cells.

The source file, `DatlingerBock2017.h5ad`, was downloaded from the [scPerturb
collection on Zenodo](https://zenodo.org/records/13350497). The experiment is
described in the original [Nature Methods
paper](https://www.nature.com/articles/nmeth.4177).

`replicate` identifies experimental replicates and is treated as categorical.

```python
import anndata as ad

adata = ad.read_h5ad(DATA / "datlinger_2017.h5ad")
print(adata.shape)
print(adata.obs["perturbation"].value_counts().head())
print(adata.obs[["replicate", "log_total_counts", "percent_mito"]].head())

screen, results, tables = run_perturbvi(
    adata,
    OUTPUT / "datlinger_anndata",
    load_kwargs={
        "guide_key": "perturbation",
        "control_label": "control",
        "covariates": ["replicate", "log_total_counts", "percent_mito"],
    },
    categorical_covariates=["replicate"],
)
```

### Norman et al. 2019 CRISPRa

This Perturb-seq experiment activated single genes and gene pairs in K562
leukemia cells using CRISPRa. The study used the resulting single-cell
transcriptional phenotypes to map genetic interactions and cell-state changes.

The source file, `NormanWeissman2019_filtered.h5ad`, was downloaded from the
[scPerturb collection on Zenodo](https://zenodo.org/records/13350497). The
experiment is described in the original [Science
paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/).

The named matrix in `adata.obsm["guide_matrix"]` has one column per target.
Control cells are its all-zero rows.

```python
import anndata as ad
import numpy as np
import pandas as pd

adata = ad.read_h5ad(DATA / "norman_2019.h5ad")
guides = adata.obsm["guide_matrix"]
assignments_per_cell = np.asarray(guides.sum(axis=1)).reshape(-1)

print(adata.shape, guides.shape)
print(pd.Series(assignments_per_cell).value_counts().sort_index())
print(adata.obs[["gemgroup", "log_total_counts", "percent_mito"]].head())

screen, results, tables = run_perturbvi(
    adata,
    OUTPUT / "norman_multiguide",
    load_kwargs={
        "guide_obsm": "guide_matrix",
        "covariates": ["gemgroup", "log_total_counts", "percent_mito"],
    },
    categorical_covariates=["gemgroup"],
)
```

### Srivatsan et al. 2020 sci-Plex 2

This sci-Plex experiment measured dose-dependent transcriptional responses in
A549 lung adenocarcinoma cells treated with BMS-345541, dexamethasone,
nutlin-3a, SAHA, or vehicle. Cells were assigned to treatments by nuclear
hashing before single-cell RNA sequencing.

The source file, `SrivatsanTrapnell2020_sciplex2.h5ad`, was downloaded from the
[scPerturb collection on Zenodo](https://zenodo.org/records/13350497). The
experiment is described in the original [Science
paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7289078/).

Drug and dose are combined into the prepared `perturbation` label. Sequencing
depth and mitochondrial percentage are numeric residualization covariates.

```python
import anndata as ad

path = DATA / "sciplex2_2020.h5ad"
adata = ad.read_h5ad(path, backed="r")
print(adata.shape)
print(adata.obs["perturbation"].value_counts().head())
print(adata.obs[["log_total_counts", "percent_mito"]].describe())
adata.file.close()

screen, results, tables = run_perturbvi(
    path,
    OUTPUT / "sciplex_h5ad",
    load_kwargs={
        "guide_key": "perturbation",
        "control_label": "control",
        "covariates": ["log_total_counts", "percent_mito"],
    },
)
```

## 10x Genomics A375 CRISPR

This dataset contains A375 melanoma cells stably expressing dCas9 and
separately transduced with either a RAB1A-targeting sgRNA or a non-targeting
sgRNA. Gene-expression and CRISPR libraries were profiled with the Chromium
GEM-X Single Cell 5' assay.

The filtered feature-barcode matrix and Cell Ranger CRISPR calls were downloaded
from the [10x Genomics A375 dataset
page](https://www.10xgenomics.com/datasets/1k-CRISPR-5p-gemx).

The 10x matrix supplies expression and barcodes. The metadata table supplies
the confident target/control calls and covariates.

```python
import pandas as pd
import scanpy as sc

matrix_path = DATA / "a375_10x_h5.h5"
metadata_path = DATA / "a375_10x_h5_metadata.tsv"

matrix = sc.read_10x_h5(matrix_path, gex_only=False)
metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

print(matrix.shape)
print(matrix.var["feature_types"].value_counts())
print(metadata["perturbation"].value_counts().head())
print(metadata[["log_total_counts", "percent_mito"]].describe())

screen, results, tables = run_perturbvi(
    matrix_path,
    OUTPUT / "a375_10x_h5",
    load_kwargs={
        "format": "10x-h5",
        "metadata_path": str(metadata_path),
        "guide_key": "perturbation",
        "control_label": "control",
        "covariates": ["log_total_counts", "percent_mito"],
    },
)
```

## 10x Genomics A549 CRISPR

This dataset contains A549 lung carcinoma cells expressing dCas9-KRAB and
transduced with a pool of 93 sgRNAs: 90 guides targeting 45 genes and three
non-targeting controls. The experiment couples 3' gene expression with direct
capture of the sgRNAs.

The feature-barcode matrix and Cell Ranger CRISPR calls were downloaded from
the [10x Genomics A549 dataset
page](https://www.10xgenomics.com/datasets/5-k-a-549-lung-carcinoma-cells-no-treatment-transduced-with-a-crispr-pool-3-1-standard-6-0-0).

Current MEX uses `matrix.mtx`, three-column `features.tsv`, and
`barcodes.tsv`; compressed files are accepted.

```python
import pandas as pd
import scanpy as sc

matrix_path = DATA / "a549_10x_mex_mex"
metadata_path = DATA / "a549_10x_mex_metadata.tsv"

matrix = sc.read_10x_mtx(matrix_path, gex_only=False)
metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

print(matrix.shape)
print(matrix.var["feature_types"].value_counts())
print(metadata["perturbation"].value_counts().head())
print(metadata[["log_total_counts", "percent_mito"]].describe())

screen, results, tables = run_perturbvi(
    matrix_path,
    OUTPUT / "a549_10x_mex",
    load_kwargs={
        "format": "10x-mex",
        "metadata_path": str(metadata_path),
        "guide_key": "perturbation",
        "control_label": "control",
        "covariates": ["log_total_counts", "percent_mito"],
    },
)
```
