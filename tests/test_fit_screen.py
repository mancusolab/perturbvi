import numpy as np
import pandas as pd
import pytest

from perturbvi import infer
from perturbvi.screen import fit_screen, PerturbData


def _make_data(n=30, p=25, g=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    G = np.zeros((n, g))
    for i in range(n):
        G[i, i % g] = 1.0
    return X, G


_KWARGS = dict(z_dim=2, l_dim=5, tau=10.0, p_prior=0.1, standardize=False, init="random", max_iter=3, seed=0)


def _screen(X, G, covariates=None):
    return PerturbData(
        X=X,
        G=G,
        gene_names=[f"gene_{index}" for index in range(X.shape[1])],
        perturbation_names=[f"pert_{index}" for index in range(G.shape[1])],
        covariates=covariates,
    )


def test_fit_screen_shapes_match_infer():
    X, G = _make_data()
    screen = _screen(X, G)

    result_screen = fit_screen(screen, **_KWARGS)
    result_infer = infer(
        X,
        G,
        z_dim=_KWARGS["z_dim"],
        l_dim=_KWARGS["l_dim"],
        tau=_KWARGS["tau"],
        p_prior=_KWARGS["p_prior"],
        standardize=_KWARGS["standardize"],
        init=_KWARGS["init"],
        max_iter=_KWARGS["max_iter"],
        seed=_KWARGS["seed"],
    )

    assert result_screen.pve.shape == result_infer.pve.shape
    assert result_screen.pip.shape == result_infer.pip.shape
    assert result_screen.W.shape == result_infer.W.shape


def test_fit_screen_accepts_aligned_dataframes_without_anndata():
    X, G = _make_data(n=30, p=25, g=3)
    cells = [f"cell_{index}" for index in range(X.shape[0])]
    data = PerturbData(
        X=pd.DataFrame(X, index=cells, columns=[f"gene_{index}" for index in range(X.shape[1])]),
        G=pd.DataFrame(G, index=cells, columns=[f"target_{index}" for index in range(G.shape[1])]),
    )

    result = fit_screen(data, **(_KWARGS | {"max_iter": 1}))

    assert result.gene_names == tuple(data.X.columns)
    assert result.perturbation_names == tuple(data.G.columns)


def test_fit_screen_rejects_all_zero_g():
    X, _ = _make_data()
    G_bad = np.zeros((30, 3))
    screen = _screen(X, G_bad)
    with pytest.raises(ValueError, match="all-zero"):
        fit_screen(screen, **_KWARGS)


def test_fit_screen_supports_annotation_prior():
    X, G = _make_data()
    screen = _screen(X, G)
    annotations = np.column_stack((np.ones(X.shape[1]), np.arange(X.shape[1]) % 2))

    result = fit_screen(
        screen,
        **(_KWARGS | {"max_iter": 1}),
        A=annotations,
        learning_rate=5e-3,
    )

    assert np.isfinite(np.asarray(result.pip)).all()


def test_fit_screen_automatically_residualizes_selected_covariates():
    X, G = _make_data()
    covariates = pd.DataFrame({"batch": pd.Categorical(["A", "B"] * 15)})

    result = fit_screen(_screen(X, G, covariates), **(_KWARGS | {"max_iter": 1}))

    assert result.gene_names[0] == "gene_0"
    assert not hasattr(result, "config")
