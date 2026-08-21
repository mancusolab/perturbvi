from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]


def _documented_python_blocks():
    markdown = (ROOT / "examples" / "cookbook.md").read_text(encoding="utf-8")
    return markdown, [part.split("```", maxsplit=1)[0] for part in markdown.split("```python")[1:]]


def _documented_runner():
    _, blocks = _documented_python_blocks()
    namespace = {}
    exec(blocks[0], namespace)
    return namespace["run_perturbvi"]


def test_run_perturbvi_executes_complete_saved_workflow(tmp_path):
    rng = np.random.default_rng(4)
    cells = [f"cell_{index}" for index in range(18)]
    adata = ad.AnnData(
        X=rng.normal(size=(18, 6)),
        obs=pd.DataFrame(
            {
                "perturbation": ["control", "target_a", "target_b"] * 6,
                "batch": ["batch_1"] * 9 + ["batch_2"] * 9,
                "depth": np.linspace(1.0, 3.0, 18),
            },
            index=cells,
        ),
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(6)]),
    )

    output = tmp_path / "results"
    screen, results, tables = _documented_runner()(
        adata,
        output,
        load_kwargs={
            "guide_key": "perturbation",
            "control_label": "control",
            "covariates": ["batch", "depth"],
        },
        categorical_covariates=["batch"],
        z_dim=1,
        l_dim=1,
        tau=1.0,
        max_iter=1,
        seed=3,
    )

    assert screen.X.shape == (18, 6)
    assert results.pip.shape == (1, 6)
    assert tables["beta"].shape == (2, 1)
    assert (output / "params_file.pkl").is_file()
    assert (output / "gene_names.txt").is_file()
    assert (output / "analysis" / "pip.csv").is_file()


def test_run_perturbvi_rejects_unloaded_categorical_covariate(tmp_path):
    with pytest.raises(ValueError, match="must also be loaded"):
        _documented_runner()(
            object(),
            tmp_path,
            load_kwargs={"covariates": ["depth"]},
            categorical_covariates=["batch"],
        )


def test_documented_python_blocks_compile_and_use_one_runner():
    markdown, blocks = _documented_python_blocks()
    for index, block in enumerate(blocks):
        compile(block, f"cookbook.md block {index}", "exec")
    assert markdown.count("screen, results, tables = run_perturbvi(") == 7


def test_documented_raw_count_preprocessing_executes():
    _, blocks = _documented_python_blocks()
    adata = ad.AnnData(
        X=sparse.csr_matrix([[2, 1, 0], [0, 3, 1], [4, 0, 2]], dtype=float),
        obs=pd.DataFrame(index=["cell_1", "cell_2", "cell_3"]),
        var=pd.DataFrame(index=["MT-ND1", "GENE_A", "GENE_B"]),
    )
    exec(blocks[1], {"adata": adata})
    assert np.isfinite(adata.X.data).all()
    np.testing.assert_allclose(adata.obs["log_total_counts"], np.log1p([3.0, 4.0, 6.0]))
    np.testing.assert_allclose(adata.obs["percent_mito"], [200 / 3, 0.0, 200 / 3])
