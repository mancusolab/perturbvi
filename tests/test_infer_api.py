import inspect

import numpy as np
import pytest

from scipy import sparse

from jax.experimental import sparse as jax_sparse

from perturbvi import infer
from perturbvi.loaders import _to_internal_matrix


def _inputs():
    X = np.array(
        [
            [1.0, 0.0, 2.0, 3.0],
            [0.0, 2.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
            [4.0, 1.0, 3.0, 2.0],
            [1.0, 4.0, 2.0, 4.0],
        ]
    )
    G = np.array([[1.0, 0.0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]])
    return X, G


def test_infer_low_level_positional_contract_is_stable():
    parameters = list(inspect.signature(infer).parameters)
    assert parameters[:4] == ["X", "z_dim", "l_dim", "G"]


def test_infer_rejects_zero_variance_before_standardizing():
    X, G = _inputs()
    X[:, 0] = 1.0
    with pytest.raises(ValueError, match="zero or invalid variance"):
        infer(
            X,
            z_dim=1,
            l_dim=1,
            G=G,
            tau=2.0,
            standardize=True,
            init="random",
            max_iter=1,
            verbose=False,
        )


def test_infer_sparse_expression_and_guides_remain_supported():
    X, G = _inputs()
    results = infer(
        _to_internal_matrix(sparse.csr_matrix(X)),
        z_dim=1,
        l_dim=1,
        G=_to_internal_matrix(sparse.csr_matrix(G)),
        tau=2.0,
        standardize=True,
        init="random",
        max_iter=1,
        verbose=False,
    )
    assert results.pip.shape == (1, X.shape[1])
    assert np.isfinite(np.asarray(results.pip)).all()


def test_infer_accepts_bcsr_through_sparse_layout_boundary():
    X, G = _inputs()
    results = infer(
        jax_sparse.BCSR.fromdense(X),
        1,
        1,
        jax_sparse.BCSR.fromdense(G),
        tau=2.0,
        standardize=True,
        init="random",
        max_iter=1,
        verbose=False,
    )
    assert results.pip.shape == (1, X.shape[1])


def test_infer_sparse_default_pca_initialization_remains_supported():
    X, G = _inputs()
    results = infer(
        _to_internal_matrix(sparse.csr_matrix(X)),
        1,
        1,
        _to_internal_matrix(sparse.csr_matrix(G)),
        tau=2.0,
        max_iter=1,
        verbose=False,
    )
    assert np.isfinite(np.asarray(results.pip)).all()


def test_infer_annotation_prior_and_dense_guide_modes_remain_supported():
    X, G = _inputs()
    A = np.column_stack((np.ones(X.shape[1]), np.arange(X.shape[1]) % 2))
    annotated = infer(
        X,
        1,
        1,
        G,
        A=A,
        tau=2.0,
        init="random",
        max_iter=1,
        verbose=False,
    )
    dense_guides = infer(
        X,
        1,
        1,
        G,
        p_prior=None,
        tau=2.0,
        init="random",
        max_iter=1,
        verbose=False,
    )
    assert np.isfinite(np.asarray(annotated.pip)).all()
    assert np.isfinite(np.asarray(dense_guides.params.mean_beta)).all()


def test_infer_does_not_mutate_numpy_inputs_when_standardizing():
    X, G = _inputs()
    original_X = X.copy()
    original_G = G.copy()
    infer(
        X,
        1,
        1,
        G,
        tau=2.0,
        standardize=True,
        init="random",
        max_iter=1,
        verbose=False,
    )
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(G, original_G)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tau": 0.0}, "tau must be"),
        ({"p_prior": 2.0}, "p_prior"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0.0}, "tol must be"),
        ({"verbose": "yes"}, "verbose must be"),
    ],
)
def test_infer_rejects_invalid_control_arguments(overrides, message):
    X, G = _inputs()
    kwargs = {
        "z_dim": 1,
        "l_dim": 1,
        "G": G,
        "tau": 2.0,
        "init": "random",
        "max_iter": 1,
        "verbose": False,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        infer(X, **kwargs)


@pytest.mark.parametrize(
    ("dimension", "value"),
    [("z_dim", 1.5), ("z_dim", True), ("l_dim", "1"), ("l_dim", False)],
)
def test_infer_rejects_noninteger_dimensions_before_initialization(dimension, value):
    X, G = _inputs()
    kwargs = {
        "z_dim": 1,
        "l_dim": 1,
        "G": G,
        "tau": 2.0,
        "init": "random",
        "max_iter": 1,
        "verbose": False,
    }
    kwargs[dimension] = value
    with pytest.raises(ValueError, match=dimension):
        infer(X, **kwargs)


def test_infer_rejects_non_array_matrix_with_clear_error():
    _, G = _inputs()
    with pytest.raises(ValueError, match="X must be"):
        infer(object(), 1, 1, G, init="random", max_iter=1, verbose=False)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda X, G: (X, G[:-1], None), "Leading dimension"),
        (lambda X, G: (X, np.column_stack((G[:, 0], np.zeros(len(G)))), None), "all-zero"),
        (lambda X, G: (X, G, np.ones((X.shape[1] - 1, 2))), "feature dimension"),
        (lambda X, G: (X, G, np.empty((X.shape[1], 0))), "annotation column"),
    ],
)
def test_infer_rejects_invalid_guide_and_annotation_shapes(change, message):
    X, G = _inputs()
    changed_X, changed_G, A = change(X, G)
    with pytest.raises(ValueError, match=message):
        infer(
            changed_X,
            1,
            1,
            changed_G,
            A=A,
            init="random",
            max_iter=1,
            verbose=False,
        )
