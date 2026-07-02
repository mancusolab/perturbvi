import numpy as np
import pytest

from perturbvi import infer
from perturbvi.screen import fit_screen, ScreenData


def _make_data(n=30, p=25, g=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    G = np.zeros((n, g))
    for i in range(n):
        G[i, i % g] = 1.0
    return X, G


_KWARGS = dict(z_dim=2, l_dim=5, tau=10.0, p_prior=0.1, standardize=False, init="random", max_iter=3, seed=0)


def test_fit_screen_shapes_match_infer():
    X, G = _make_data()
    screen = ScreenData(X=X, G=G, genes=None, perturbations=None, cell_names=None, source={})

    result_screen = fit_screen(screen, **_KWARGS)
    result_infer = infer(
        X, z_dim=_KWARGS["z_dim"], l_dim=_KWARGS["l_dim"], G=G,
        tau=_KWARGS["tau"], p_prior=_KWARGS["p_prior"],
        standardize=_KWARGS["standardize"], init=_KWARGS["init"],
        max_iter=_KWARGS["max_iter"], seed=_KWARGS["seed"],
    )

    assert result_screen.pve.shape == result_infer.pve.shape
    assert result_screen.pip.shape == result_infer.pip.shape
    assert result_screen.W.shape == result_infer.W.shape


def test_fit_screen_rejects_all_zero_g():
    X, _ = _make_data()
    G_bad = np.zeros((30, 3))
    screen = ScreenData(X=X, G=G_bad, genes=None, perturbations=None, cell_names=None, source={})
    with pytest.raises(ValueError, match="all-zero"):
        fit_screen(screen, **_KWARGS)


def test_fit_screen_rejects_row_mismatch():
    X, G = _make_data()
    screen = ScreenData(X=X, G=G[:20], genes=None, perturbations=None, cell_names=None, source={})
    with pytest.raises(ValueError, match="rows"):
        fit_screen(screen, **_KWARGS)
