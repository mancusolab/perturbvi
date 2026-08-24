from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _documented_python_blocks():
    markdown = (ROOT / "examples" / "cookbook.md").read_text(encoding="utf-8")
    return markdown, [part.split("```", maxsplit=1)[0] for part in markdown.split("```python")[1:]]


def test_documented_workflow_writes_only_documented_files(tmp_path):
    from perturbvi import fit_screen, load_screen, residualize_screen, save_results
    from perturbvi.utils import analyze

    rng = np.random.default_rng(4)
    cells = [f"cell_{index}" for index in range(18)]
    adata = ad.AnnData(
        X=rng.normal(size=(18, 6)),
        obs=pd.DataFrame(
            {
                "perturbation": ["control", "target_a", "target_b"] * 6,
                "batch": pd.Categorical(["batch_1"] * 9 + ["batch_2"] * 9),
                "depth": np.linspace(1.0, 3.0, 18),
            },
            index=cells,
        ),
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(6)]),
    )
    adata.obsm["G"] = (
        adata.obs["perturbation"].astype("string").str.get_dummies().astype(int)
    )

    output = tmp_path / "results"
    screen = load_screen(
        adata,
        control="control",
        covariates=["depth", "batch"],
    )
    screen = residualize_screen(screen)
    results = fit_screen(
        screen,
        z_dim=1,
        l_dim=1,
        tau=1.0,
        max_iter=1,
        seed=3,
    )
    save_results(results, str(output))
    tables = analyze(results)

    assert screen.X.shape == (18, 6)
    assert results.pip.shape == (1, 6)
    assert tables["perturbation_effect"].shape == (2, 1)
    assert (output / "params_file.pkl").is_file()
    assert not list(output.glob("*.json"))


def test_documented_python_blocks_compile():
    markdown, blocks = _documented_python_blocks()
    for index, block in enumerate(blocks):
        compile(block, f"cookbook.md block {index}", "exec")
    assert "def transform_counts(" in markdown
    assert "fit = fit_screen(" in markdown


def test_documented_raw_count_preprocessing_executes():
    _, blocks = _documented_python_blocks()
    transform_block = next(block for block in blocks if "def transform_counts(" in block)
    namespace = {}
    exec(transform_block, namespace)

    rng = np.random.default_rng(12)
    adata = ad.AnnData(
        X=rng.poisson(3, size=(12, 8)),
        obs=pd.DataFrame(
            {"condition": ["control", "target"] * 6},
            index=[f"cell_{index}" for index in range(12)],
        ),
        var=pd.DataFrame(index=[f"GENE_{index}" for index in range(8)]),
    )
    transformed = namespace["transform_counts"](
        adata,
        n_top_genes=4,
        min_genes=1,
        min_cells=1,
        max_pct_mt=100.0,
    )
    assert transformed.shape == (12, 4)
    assert np.isfinite(np.asarray(transformed.X)).all()
