import gzip
import io

import h5py
import numpy as np

from scipy import sparse
from scipy.io import mmwrite

import jax.numpy as jnp

from perturbvi.common import ModelParams


def tenx_matrix():
    values = np.array(
        [
            [1, 0, 2, 2, 0],
            [0, 2, 1, 0, 3],
            [3, 1, 0, 2, 0],
            [2, 3, 1, 0, 4],
            [4, 1, 3, 2, 0],
            [1, 4, 2, 0, 3],
        ],
        dtype=np.int32,
    )
    ids = ["gene_1", "gene_2", "gene_3", "guide_1", "guide_2"]
    feature_types = ["Gene Expression"] * 3 + ["CRISPR Guide Capture"] * 2
    barcodes = [f"cell_{index}" for index in range(values.shape[0])]
    return values, ids, feature_types, barcodes


def write_10x_h5(path):
    values, ids, feature_types, barcodes = tenx_matrix()
    matrix = sparse.csr_matrix(values)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        group.create_dataset("shape", data=np.array([values.shape[1], values.shape[0]], dtype=np.int64))
        group.create_dataset("barcodes", data=np.asarray(barcodes, dtype="S"))
        features = group.create_group("features")
        features.create_dataset("id", data=np.asarray(ids, dtype="S"))
        features.create_dataset("name", data=np.asarray(ids, dtype="S"))
        features.create_dataset("feature_type", data=np.asarray(feature_types, dtype="S"))
    return path


def write_10x_mex(path):
    values, ids, feature_types, barcodes = tenx_matrix()
    path.mkdir()
    buffer = io.BytesIO()
    mmwrite(buffer, sparse.coo_matrix(values.T))
    with gzip.open(path / "matrix.mtx.gz", "wb") as handle:
        handle.write(buffer.getvalue())
    with gzip.open(path / "features.tsv.gz", "wt", encoding="utf-8") as handle:
        for feature_id, feature_type in zip(ids, feature_types):
            handle.write(f"{feature_id}\t{feature_id}\t{feature_type}\n")
    with gzip.open(path / "barcodes.tsv.gz", "wt", encoding="utf-8") as handle:
        for barcode in barcodes:
            handle.write(f"{barcode}\n")
    return path


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
