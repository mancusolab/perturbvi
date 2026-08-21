import json

import numpy as np
import pandas as pd
import pytest

from perturbvi import analyze, infer, load_results, save_results


@pytest.fixture
def small_results(tmp_path):
    rng = np.random.default_rng(0)
    cells, genes, perturbations = 50, 15, 4
    X = rng.normal(0, 1, (cells, genes))
    G = (rng.random((cells, perturbations)) > 0.5).astype(float)
    for column in range(perturbations):
        if G[:, column].sum() == 0:
            G[0, column] = 1.0
    results = infer(X, z_dim=2, l_dim=5, G=G, tau=50, max_iter=5, seed=0)
    save_results(results, path=str(tmp_path))
    return results, tmp_path


def test_analyze_lfsr_none_by_default(small_results):
    results, _ = small_results
    tables = analyze(results)
    assert tables["lfsr"] is None
    assert {"pip", "pve", "overall_effect"} <= set(tables)


def test_analyze_compute_lfsr(small_results):
    results, _ = small_results
    tables = analyze(results, compute_lfsr=True, lfsr_iters=50)
    assert tables["lfsr"].shape == (15, 4)


def test_analyze_with_names(small_results):
    results, _ = small_results
    gene_names = [f"gene_{index}" for index in range(15)]
    perturbation_names = [f"pert_{index}" for index in range(4)]
    tables = analyze(
        results,
        gene_names=gene_names,
        perturbation_names=perturbation_names,
    )
    assert list(tables["pip"].index) == gene_names
    assert list(tables["overall_effect"].columns) == perturbation_names


def test_analysis_thresholds_drive_summary_tables(small_results):
    results, _ = small_results
    pip_threshold = 0.5
    tables = analyze(results, pip_threshold=pip_threshold)
    assert len(tables["pip_significant"]) == int(
        (np.asarray(results.pip) >= pip_threshold).sum()
    )
    np.testing.assert_array_equal(
        tables["pip_summary"]["n_pip_significant"],
        (np.asarray(results.pip) >= pip_threshold).sum(axis=1),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pip_threshold": "high"}, "pip_threshold"),
        ({"lfsr_threshold": np.nan}, "lfsr_threshold"),
        ({"lfsr_iters": False}, "lfsr_iters"),
        ({"seed": True}, "seed"),
    ],
)
def test_analysis_rejects_invalid_scalar_controls(small_results, kwargs, message):
    results, _ = small_results
    with pytest.raises(ValueError, match=message):
        analyze(results, **kwargs)


def test_analyze_forwards_lfsr_seed_and_iterations(small_results, monkeypatch):
    import jax.numpy as jnp

    results, _ = small_results
    called = {}

    def fake_compute_lfsr(key, params, iters):
        called["key"] = np.asarray(key)
        called["iters"] = iters
        return jnp.zeros((params.mean_beta.shape[0], params.W.shape[1]))

    monkeypatch.setattr("perturbvi.utils.compute_lfsr", fake_compute_lfsr)
    tables = analyze(
        results,
        compute_lfsr=True,
        lfsr_iters=17,
        lfsr_threshold=0.1,
        seed=9,
    )
    assert called["iters"] == 17
    assert not np.array_equal(called["key"], np.array([0, 0], dtype=np.uint32))
    assert tables["lfsr"].shape == (15, 4)
    assert len(tables["lfsr_significant"]) == 15 * 4


def test_analysis_thresholds_are_inclusive(small_results, monkeypatch):
    import jax.numpy as jnp

    results, _ = small_results
    pip_threshold = float(np.asarray(results.pip)[0, 0])

    def fake_compute_lfsr(key, params, iters):
        return jnp.full((params.mean_beta.shape[0], params.W.shape[1]), 0.05)

    monkeypatch.setattr("perturbvi.utils.compute_lfsr", fake_compute_lfsr)
    tables = analyze(
        results,
        pip_threshold=pip_threshold,
        compute_lfsr=True,
        lfsr_threshold=0.05,
        lfsr_iters=1,
    )

    first_gene = tables["pip"].index[0]
    assert (
        (tables["pip_significant"]["factor"] == "w0")
        & (tables["pip_significant"]["gene"] == first_gene)
    ).any()
    assert len(tables["lfsr_significant"]) == 15 * 4


def test_saved_analysis_uses_saved_names_and_cached_lfsr(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 5))
    G = np.zeros((12, 2))
    G[:6, 0] = 1
    G[6:, 1] = 1
    results = infer(X, z_dim=1, l_dim=1, G=G, tau=5, max_iter=1, verbose=False)
    output = tmp_path / "results"
    gene_names = [f"g{index}" for index in range(5)]
    perturbation_names = ["p1", "p2"]
    save_results(results, str(output))
    (output / "input_summary.json").write_text(
        json.dumps(
            {
                "gene_names": gene_names,
                "perturbation_names": perturbation_names,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        np.zeros((5, 2)),
        index=gene_names,
        columns=perturbation_names,
    ).to_csv(output / "lfsr.csv")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("saved LFSR should be reused")

    monkeypatch.setattr("perturbvi.utils.compute_lfsr", fail_if_called)

    loaded = load_results(str(output))
    assert loaded.pip.shape == (1, 5)
    tables = analyze(str(output))
    assert list(tables["pip"].index) == gene_names
    assert list(tables["beta"].index) == perturbation_names
    assert tables["lfsr"].shape == (5, 2)
    assert not tables["lfsr"].to_numpy().any()


@pytest.mark.parametrize("mismatched_axis", ["gene", "perturbation"])
def test_saved_analysis_rejects_mislabeled_cached_lfsr(small_results, mismatched_axis):
    _, output = small_results
    gene_names = [f"gene_{index}" for index in range(15)]
    perturbation_names = [f"pert_{index}" for index in range(4)]
    (output / "input_summary.json").write_text(
        json.dumps(
            {
                "gene_names": gene_names,
                "perturbation_names": perturbation_names,
            }
        ),
        encoding="utf-8",
    )

    cached_genes = gene_names.copy()
    cached_perturbations = perturbation_names.copy()
    if mismatched_axis == "gene":
        cached_genes[0] = "wrong_gene"
    else:
        cached_perturbations[0] = "wrong_perturbation"
    pd.DataFrame(
        np.zeros((15, 4)),
        index=cached_genes,
        columns=cached_perturbations,
    ).to_csv(output / "lfsr.csv")

    with pytest.raises(ValueError, match=f"LFSR {mismatched_axis} labels"):
        analyze(str(output))


def test_analysis_rejects_duplicate_names(small_results):
    results, _ = small_results
    with pytest.raises(ValueError, match="duplicate"):
        analyze(results, gene_names=["same"] * results.W.shape[1])
