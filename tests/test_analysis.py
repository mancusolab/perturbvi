import numpy as np
import pandas as pd
import pytest

from perturbvi import FitResults, infer, load_results, save_results
from perturbvi.utils import analyze


@pytest.fixture
def fitted(tmp_path):
    rng = np.random.default_rng(0)
    n_cells, n_genes, n_perturbations = 80, 30, 6
    X = rng.normal(size=(n_cells, n_genes))
    G = np.zeros((n_cells, n_perturbations))
    G[np.arange(n_cells), np.arange(n_cells) % n_perturbations] = 1
    inference = infer(X, G, z_dim=3, l_dim=5, tau=20, max_iter=2, seed=0, verbose=False)
    fit = FitResults(
        inference=inference,
        gene_names=tuple(f"gene_{index}" for index in range(n_genes)),
        perturbation_names=tuple(f"target_{index}" for index in range(n_perturbations)),
    )
    output = tmp_path / "fit"
    save_results(fit, output)
    return fit, output


def test_analysis_returns_only_distinct_core_tables(fitted):
    fit, _ = fitted
    tables = analyze(fit)

    assert set(tables) == {
        "pip",
        "pve",
        "perturbation_effect",
        "perturbation_pip",
        "gene_effect",
    }
    assert tables["pip"].shape == (30, 3)
    assert tables["perturbation_effect"].shape == (6, 3)
    assert tables["perturbation_pip"].shape == (6, 3)
    assert tables["gene_effect"].shape == (30, 6)


def test_saved_analysis_uses_embedded_labels(fitted):
    _, output = fitted
    tables = analyze(output)
    assert list(tables["pip"].index) == [f"gene_{index}" for index in range(30)]
    assert list(tables["perturbation_effect"].index) == [f"target_{index}" for index in range(6)]


def test_raw_inference_can_be_labeled_explicitly(fitted):
    fit, _ = fitted
    genes = [f"feature_{index}" for index in range(30)]
    perturbations = [f"perturbation_{index}" for index in range(6)]
    tables = analyze(fit.inference, gene_names=genes, perturbation_names=perturbations)
    assert list(tables["pip"].index) == genes
    assert list(tables["gene_effect"].columns) == perturbations


def test_lfsr_is_computed_only_when_explicitly_requested(fitted, monkeypatch):
    import jax.numpy as jnp

    fit, output = fitted
    pd.DataFrame(np.ones((30, 6))).to_csv(output / "lfsr.csv")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LFSR must be opt-in")

    monkeypatch.setattr("perturbvi.utils.compute_lfsr", fail_if_called)
    assert "lfsr" not in analyze(output)

    calls = {}

    def fake_lfsr(key, params, iters):
        calls["iters"] = iters
        calls["key"] = np.asarray(key)
        return jnp.zeros((params.mean_beta.shape[0], params.W.shape[1]))

    monkeypatch.setattr("perturbvi.utils.compute_lfsr", fake_lfsr)
    tables = analyze(fit, compute_lfsr=True, lfsr_iters=17, seed=9)
    assert calls["iters"] == 17
    assert tables["lfsr"].shape == (30, 6)


@pytest.mark.parametrize(("kwargs", "message"), [({"lfsr_iters": 0}, "lfsr_iters"), ({"seed": True}, "seed")])
def test_lfsr_controls_are_checked_when_lfsr_is_requested(fitted, kwargs, message):
    fit, _ = fitted
    with pytest.raises(ValueError, match=message):
        analyze(fit, compute_lfsr=True, **kwargs)


def test_text_exports_are_outputs_not_load_dependencies(fitted):
    _, output = fitted
    for name in ("W.txt", "pip.txt", "pve.txt"):
        (output / name).unlink()

    loaded = load_results(output)
    assert isinstance(loaded, FitResults)
    assert loaded.pip.shape == (3, 30)
    assert loaded.pve.shape == (3,)


def test_duplicate_labels_are_rejected_at_the_table_boundary(fitted):
    fit, _ = fitted
    with pytest.raises(ValueError, match="duplicate"):
        analyze(fit.inference, gene_names=["same"] * 30)


def test_public_analyze_lives_in_utils_and_legacy_name_is_gone():
    import perturbvi.utils as utils

    assert callable(utils.analyze)
    assert not hasattr(utils, "analyze_output")
