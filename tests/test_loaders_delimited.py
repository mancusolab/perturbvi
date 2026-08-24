import anndata as ad
import numpy as np
import pandas as pd

from perturbvi import load_screen


def test_csv_tables_convert_to_canonical_h5ad(tmp_path):
    cells = ["c1", "c2", "c3"]
    expression = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=cells,
        columns=["g1", "g2"],
    )
    guides = pd.DataFrame(
        [[1, 0], [0, 0], [0, 1]],
        index=cells,
        columns=["p1", "p2"],
    )
    metadata = pd.DataFrame(
        {
            "batch": pd.Categorical(["A", "A", "B"]),
            "depth": [10.0, 20.0, 30.0],
        },
        index=cells,
    )

    metadata["perturbation"] = pd.Categorical(["p1", "control", "p2"])
    metadata[["p1", "p2"]] = guides.astype(bool)
    adata = ad.AnnData(
        X=expression.to_numpy(dtype=np.float64),
        obs=metadata,
        var=pd.DataFrame(index=expression.columns.astype(str)),
    )
    adata.obsm["G"] = guides
    prepared_path = tmp_path / "prepared.h5ad"
    adata.write_h5ad(prepared_path)

    screen = load_screen(
        prepared_path,
        covariates=["batch", "depth"],
    )
    assert screen.gene_names == ("g1", "g2")
    assert screen.perturbation_names == ("p1", "p2")
    assert list(screen.covariates) == ["batch", "depth"]
    np.testing.assert_array_equal(np.asarray(screen.G), guides.to_numpy())
