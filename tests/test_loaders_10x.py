import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from perturbvi.loaders import _load_10x


def _make_10x_adata(n=30, n_exp=10, n_guide=3, seed=0):
    """Minimal AnnData with feature_types column, like Cell Ranger output."""
    rng = np.random.default_rng(seed)
    n_vars = n_exp + n_guide
    X = rng.poisson(3, (n, n_vars)).astype(float)
    var = pd.DataFrame(
        {"feature_types": ["Gene Expression"] * n_exp + ["CRISPR Guide Capture"] * n_guide},
        index=[f"gene_{i}" for i in range(n_exp)] + [f"guide_{i}" for i in range(n_guide)],
    )
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_load_10x_splits_features():
    adata = _make_10x_adata(n_exp=10, n_guide=3)
    screen = _load_10x(
        adata,
        expression_feature_type="Gene Expression",
        guide_feature_type="CRISPR Guide Capture",
        guide_threshold=1,
        multi_guide="allow",
        source_format="10x-h5",
        source_path="fake.h5",
        covariate_file=None,
        covariate_keys=None,
    )
    assert screen.X.shape == (30, 10)
    assert screen.G.shape == (30, 3)
    assert len(screen.genes) == 10
    assert len(screen.perturbations) == 3


def test_load_10x_missing_expression_type():
    adata = _make_10x_adata()
    with pytest.raises(ValueError, match="No features with feature_type='Bad Type'"):
        _load_10x(
            adata,
            expression_feature_type="Bad Type",
            guide_feature_type="CRISPR Guide Capture",
            guide_threshold=1,
            multi_guide="allow",
            source_format="10x-h5",
            source_path="fake.h5",
            covariate_file=None,
            covariate_keys=None,
        )


def test_load_10x_missing_guide_type():
    adata = _make_10x_adata()
    with pytest.raises(ValueError, match="No features with feature_type='Bad Guide'"):
        _load_10x(
            adata,
            expression_feature_type="Gene Expression",
            guide_feature_type="Bad Guide",
            guide_threshold=1,
            multi_guide="allow",
            source_format="10x-h5",
            source_path="fake.h5",
            covariate_file=None,
            covariate_keys=None,
        )


def test_load_10x_missing_feature_types_column():
    adata = _make_10x_adata()
    adata.var = adata.var.drop(columns=["feature_types"])
    with pytest.raises(ValueError, match="feature_types"):
        _load_10x(
            adata,
            expression_feature_type="Gene Expression",
            guide_feature_type="CRISPR Guide Capture",
            guide_threshold=1,
            multi_guide="allow",
            source_format="10x-h5",
            source_path="fake.h5",
            covariate_file=None,
            covariate_keys=None,
        )


def test_load_10x_binarizes_at_threshold():
    adata = _make_10x_adata(n=10, n_exp=5, n_guide=2)
    # Set guide counts to known values
    adata.X[:, 5] = 0   # guide_0 — all zeros
    adata.X[:, 6] = 3   # guide_1 — all above threshold
    adata.X[0, 5] = 2   # cell_0 guide_0 = 2 (above threshold 1)

    screen = _load_10x(
        adata,
        expression_feature_type="Gene Expression",
        guide_feature_type="CRISPR Guide Capture",
        guide_threshold=1,
        multi_guide="allow",
        source_format="10x-h5",
        source_path="fake.h5",
        covariate_file=None,
        covariate_keys=None,
    )
    assert screen.G[0, 0] == 1.0   # cell_0, guide_0 → above threshold
    assert screen.G[1, 1] == 1.0   # cell_1, guide_1 → above threshold


def test_load_10x_multi_guide_warn():
    adata = _make_10x_adata(n=10, n_exp=5, n_guide=3)
    # Force cell_0 to have all guides assigned
    adata.X[0, 5] = 5
    adata.X[0, 6] = 5
    adata.X[0, 7] = 5

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _load_10x(
            adata,
            expression_feature_type="Gene Expression",
            guide_feature_type="CRISPR Guide Capture",
            guide_threshold=1,
            multi_guide="warn",
            source_format="10x-h5",
            source_path="fake.h5",
            covariate_file=None,
            covariate_keys=None,
        )
    assert any("more than one guide" in str(w.message) for w in caught)


def test_load_10x_multi_guide_error():
    adata = _make_10x_adata(n=10, n_exp=5, n_guide=3)
    adata.X[0, 5] = 5
    adata.X[0, 6] = 5
    adata.X[0, 7] = 5

    with pytest.raises(ValueError, match="more than one guide"):
        _load_10x(
            adata,
            expression_feature_type="Gene Expression",
            guide_feature_type="CRISPR Guide Capture",
            guide_threshold=1,
            multi_guide="error",
            source_format="10x-h5",
            source_path="fake.h5",
            covariate_file=None,
            covariate_keys=None,
        )


def test_load_10x_multi_guide_allow():
    adata = _make_10x_adata(n=10, n_exp=5, n_guide=3)
    adata.X[0, 5] = 5
    adata.X[0, 6] = 5
    adata.X[0, 7] = 5

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        screen = _load_10x(
            adata,
            expression_feature_type="Gene Expression",
            guide_feature_type="CRISPR Guide Capture",
            guide_threshold=1,
            multi_guide="allow",
            source_format="10x-h5",
            source_path="fake.h5",
            covariate_file=None,
            covariate_keys=None,
        )
    assert not any("more than one guide" in str(w.message) for w in caught)
    assert screen.G[0].sum() == 3.0


def test_load_10x_covariate_file(tmp_path):
    adata = _make_10x_adata(n=10, n_exp=5, n_guide=2)
    rng = np.random.default_rng(0)

    cov_df = pd.DataFrame(
        {"batch": rng.choice(["A", "B"], 10), "depth": rng.normal(0, 1, 10)},
        index=list(adata.obs_names),
    )
    cov_path = tmp_path / "covariates.csv"
    cov_df.to_csv(cov_path)

    screen = _load_10x(
        adata,
        expression_feature_type="Gene Expression",
        guide_feature_type="CRISPR Guide Capture",
        guide_threshold=1,
        multi_guide="allow",
        source_format="10x-h5",
        source_path="fake.h5",
        covariate_file=str(cov_path),
        covariate_keys=["batch", "depth"],
    )
    assert screen.covariates is not None
    assert screen.covariates.shape == (10, 2)
    assert screen.covariate_names == ["batch", "depth"]


def test_load_10x_covariate_file_missing_barcode(tmp_path):
    adata = _make_10x_adata(n=10, n_exp=5, n_guide=2)
    rng = np.random.default_rng(0)

    # Only 5 of 10 barcodes present
    cov_df = pd.DataFrame(
        {"depth": rng.normal(0, 1, 5)},
        index=list(adata.obs_names)[:5],
    )
    cov_path = tmp_path / "covariates.csv"
    cov_df.to_csv(cov_path)

    with pytest.raises(ValueError, match="barcodes not found in covariate file"):
        _load_10x(
            adata,
            expression_feature_type="Gene Expression",
            guide_feature_type="CRISPR Guide Capture",
            guide_threshold=1,
            multi_guide="allow",
            source_format="10x-h5",
            source_path="fake.h5",
            covariate_file=str(cov_path),
            covariate_keys=["depth"],
        )


def test_load_10x_covariate_file_without_keys_raises(tmp_path):
    adata = _make_10x_adata(n=5, n_exp=3, n_guide=2)
    cov_path = tmp_path / "cov.csv"
    pd.DataFrame({"x": [1] * 5}, index=list(adata.obs_names)).to_csv(cov_path)

    with pytest.raises(ValueError, match="covariates= must be provided alongside covariate_file"):
        _load_10x(
            adata,
            expression_feature_type="Gene Expression",
            guide_feature_type="CRISPR Guide Capture",
            guide_threshold=1,
            multi_guide="allow",
            source_format="10x-h5",
            source_path="fake.h5",
            covariate_file=str(cov_path),
            covariate_keys=None,
        )


def test_load_screen_10x_covariates_without_file_raises():
    from perturbvi import load_screen

    with pytest.raises(ValueError, match="covariate_file="):
        load_screen("fake.h5", format="10x-h5", covariates=["batch"])
