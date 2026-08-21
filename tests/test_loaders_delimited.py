import numpy as np
import pandas as pd
import pytest

from perturbvi import load_screen


def _write_expression(tmp_path):
    expression = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=["c1", "c2", "c3"],
        columns=["g1", "g2"],
    )
    path = tmp_path / "expression.csv"
    expression.to_csv(path)
    return path


def test_load_delimited_expression_and_guide_matrix_aligns_rows(tmp_path):
    expression_path = _write_expression(tmp_path)
    guides = pd.DataFrame(
        [[0, 1], [1, 0], [1, 0]],
        index=["c3", "c1", "c2"],
        columns=["p1", "p2"],
    )
    guide_path = tmp_path / "guides.tsv"
    guides.to_csv(guide_path, sep="\t")

    screen = load_screen(str(expression_path), guide_path=str(guide_path))

    assert screen.source["format"] == "csv"
    assert screen.gene_names == ["g1", "g2"]
    assert screen.perturbation_names == ["p1", "p2"]
    np.testing.assert_array_equal(screen.G, [[1, 0], [1, 0], [0, 1]])


def test_load_delimited_metadata_guides_and_covariates(tmp_path):
    expression_path = _write_expression(tmp_path)
    metadata = pd.DataFrame(
        {
            "guide": ["p1", "control", "p2"],
            "batch": ["A", "A", "B"],
            "depth": [10.0, 20.0, 30.0],
        },
        index=["c1", "c2", "c3"],
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path)

    screen = load_screen(
        str(expression_path),
        metadata_path=str(metadata_path),
        guide_key="guide",
        control_label="control",
        covariates=["batch", "depth"],
    )

    assert screen.perturbation_names == ["p1", "p2"]
    assert screen.covariate_names == ["batch", "depth"]
    assert screen.covariates.shape == (3, 2)


def test_load_delimited_rejects_misaligned_guide_rows(tmp_path):
    expression_path = _write_expression(tmp_path)
    guide_path = tmp_path / "guides.csv"
    pd.DataFrame([[1], [1]], index=["c1", "c2"], columns=["p1"]).to_csv(guide_path)

    with pytest.raises(ValueError, match="row names do not match"):
        load_screen(str(expression_path), guide_path=str(guide_path))
