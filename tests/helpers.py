import numpy as np
import pandas as pd

import jax.numpy as jnp

from perturbvi.common import ModelParams
from perturbvi.preprocess import _build_design_matrix_with_names


def build_design_matrix(covariates: pd.DataFrame) -> np.ndarray:
    """Test shim returning just the design matrix, without column names."""
    return _build_design_matrix_with_names(covariates)[0]


def make_model_params(
    x_ssq,
    mean_z,
    var_z,
    mean_w,
    var_w,
    alpha,
    tau,
    tau_0,
    theta,
    pi,
    ann_state,
    mean_beta,
    var_beta,
    tau_beta,
    p,
    p_hat=None,
):
    if p_hat is None:
        z_dim = mean_z.shape[1]
        g_dim = mean_beta.shape[0]
        p_hat = jnp.ones((z_dim, g_dim)) * 0.5
    return ModelParams(
        x_ssq=x_ssq,
        mean_z=mean_z,
        var_z=var_z,
        mean_w=mean_w,
        var_w=var_w,
        alpha=alpha,
        tau=tau,
        tau_0=tau_0,
        theta=theta,
        pi=pi,
        ann_state=ann_state,
        mean_beta=mean_beta,
        var_beta=var_beta,
        tau_beta=tau_beta,
        p=p,
        p_hat=p_hat,
    )
