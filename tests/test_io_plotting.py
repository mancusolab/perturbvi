from types import SimpleNamespace

import numpy as np

from perturbvi.io import plot_beta


def test_plot_beta_uses_positional_indexing_for_default_integer_labels(tmp_path):
    params = SimpleNamespace(
        mean_beta=np.array([[0.2], [-0.5]]),
        p_hat=np.ones((1, 2)),
    )
    results = SimpleNamespace(params=params)

    output = tmp_path / "beta.png"
    plot_beta(results, factor_idx=0, save_path=str(output))

    assert output.is_file()
