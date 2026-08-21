import anndata as ad
import numpy as np
import pandas as pd
import pytest

from jax.experimental import sparse as jax_sparse

from perturbvi import load_screen
from perturbvi.loaders import _load_10x

from .helpers import write_10x_h5, write_10x_mex


def _make_10x_adata(n=30, n_exp=10, n_assay=3, seed=0):
    """Minimal current Cell Ranger feature-barcode matrix."""
    rng = np.random.default_rng(seed)
    n_vars = n_exp + n_assay
    X = rng.poisson(3, (n, n_vars)).astype(float)
    var = pd.DataFrame(
        {"feature_types": ["Gene Expression"] * n_exp + ["CRISPR Guide Capture"] * n_assay},
        index=[f"gene_{i}" for i in range(n_exp)] + [f"assay_{i}" for i in range(n_assay)],
    )
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n)])
    return ad.AnnData(X=X, obs=obs, var=var)


def _write_metadata(path, cells):
    labels = ["control", "target_a", "target_b"]
    pd.DataFrame(
        {
            "target": [labels[index % len(labels)] for index in range(len(cells))],
            "batch": [f"b{index % 2}" for index in range(len(cells))],
        },
        index=cells,
    ).to_csv(path)
    return path


def test_load_10x_selects_expression_and_uses_metadata_labels(tmp_path):
    adata = _make_10x_adata(n_exp=10, n_assay=3)
    metadata = _write_metadata(tmp_path / "metadata.csv", adata.obs_names)
    screen = _load_10x(
        adata,
        expression_feature_type="Gene Expression",
        source_format="10x-h5",
        source_path="fake.h5",
        metadata_path=str(metadata),
        guide_key="target",
        control_label="control",
    )

    assert screen.X.shape == (30, 10)
    assert screen.G.shape == (30, 2)
    assert screen.gene_names == [f"gene_{index}" for index in range(10)]
    assert screen.perturbation_names == ["target_a", "target_b"]
    assert screen.source["guide_assignment"] == "metadata"
    assert screen.source["n_total_vars"] == 13


def test_load_10x_missing_expression_type(tmp_path):
    adata = _make_10x_adata()
    metadata = _write_metadata(tmp_path / "metadata.csv", adata.obs_names)
    with pytest.raises(ValueError, match="No features with feature_type='Bad Type'"):
        _load_10x(
            adata,
            expression_feature_type="Bad Type",
            source_format="10x-h5",
            source_path="fake.h5",
            metadata_path=str(metadata),
            guide_key="target",
        )


def test_load_10x_missing_feature_types_column(tmp_path):
    adata = _make_10x_adata()
    adata.var = adata.var.drop(columns=["feature_types"])
    metadata = _write_metadata(tmp_path / "metadata.csv", adata.obs_names)
    with pytest.raises(ValueError, match="feature_types"):
        _load_10x(
            adata,
            expression_feature_type="Gene Expression",
            source_format="10x-h5",
            source_path="fake.h5",
            metadata_path=str(metadata),
            guide_key="target",
        )


def test_load_screen_10x_requires_metadata():
    with pytest.raises(ValueError, match="requires one barcode-indexed metadata_path"):
        load_screen("fake.h5", format="10x-h5")


def test_load_screen_10x_requires_guide_key_with_metadata(tmp_path):
    metadata = _write_metadata(tmp_path / "metadata.csv", ["cell_0"])
    with pytest.raises(ValueError, match="guide_key is required"):
        load_screen("fake.h5", format="10x-h5", metadata_path=str(metadata))


@pytest.mark.parametrize("option", [{"layer": "counts"}, {"guide_obsm": "guides"}, {"guide_path": "g.csv"}])
def test_load_screen_10x_rejects_inapplicable_options(option):
    with pytest.raises(ValueError, match="not applicable|only applicable"):
        load_screen("fake.h5", format="10x-h5", **option)


def test_load_10x_metadata_supplies_subset_and_covariates(tmp_path):
    adata = _make_10x_adata(n=8, n_exp=4, n_assay=2)
    selected = ["cell_6", "cell_2", "cell_4", "cell_0"]
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "target": ["control", "target_b", "target_a", "target_a"],
            "batch": ["b2", "b1", "b2", "b1"],
            "depth": [6.0, 2.0, 4.0, 0.0],
        },
        index=selected,
    ).to_csv(metadata_path)

    screen = _load_10x(
        adata,
        expression_feature_type="Gene Expression",
        source_format="10x-h5",
        source_path="fake.h5",
        metadata_path=str(metadata_path),
        guide_key="target",
        control_label="control",
        covariates=["batch", "depth"],
    )

    assert screen.cell_names == ["cell_0", "cell_2", "cell_4", "cell_6"]
    assert screen.perturbation_names == ["target_a", "target_b"]
    assert screen.X.shape == (4, 4)
    assert screen.G.shape == (4, 2)
    assert screen.covariate_names == ["batch", "depth"]
    assert screen.source["n_input_obs"] == 8


def test_load_10x_metadata_rejects_unknown_barcode(tmp_path):
    adata = _make_10x_adata(n=4, n_exp=3, n_assay=2)
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"target": ["target_a"]}, index=["not_in_matrix"]).to_csv(metadata_path)

    with pytest.raises(ValueError, match="not present in the 10x matrix"):
        _load_10x(
            adata,
            expression_feature_type="Gene Expression",
            source_format="10x-h5",
            source_path="fake.h5",
            metadata_path=str(metadata_path),
            guide_key="target",
        )


def test_load_10x_h5_adapter_keeps_expression_sparse(tmp_path):
    path = write_10x_h5(tmp_path / "screen.h5")
    metadata = _write_metadata(tmp_path / "metadata.csv", [f"cell_{index}" for index in range(6)])
    screen = load_screen(
        str(path),
        format="10x-h5",
        metadata_path=str(metadata),
        guide_key="target",
        control_label="control",
    )

    assert isinstance(screen.X, jax_sparse.JAXSparse)
    assert screen.X.shape == (6, 3)
    assert screen.G.shape == (6, 2)
    assert screen.gene_names == ["gene_1", "gene_2", "gene_3"]
    assert screen.perturbation_names == ["target_a", "target_b"]


def test_load_10x_mex_adapter_keeps_expression_sparse(tmp_path):
    path = write_10x_mex(tmp_path / "mex")
    metadata = _write_metadata(tmp_path / "metadata.csv", [f"cell_{index}" for index in range(6)])
    screen = load_screen(
        str(path),
        format="10x-mex",
        metadata_path=str(metadata),
        guide_key="target",
        control_label="control",
    )

    assert isinstance(screen.X, jax_sparse.JAXSparse)
    assert screen.X.shape == (6, 3)
    assert screen.G.shape == (6, 2)
