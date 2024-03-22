from __future__ import annotations

from typing import Optional

from plum import dispatch

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.linalg as jnla
import lineax as lx

from jax._src.dtypes import JAXType  # ug...
from jax.experimental import sparse
from jaxtyping import Array, ArrayLike, Float, Num


_sparse_mean = sparse.sparsify(jnp.mean)


@jax.jit
@sparse.sparsify
def _get_mean_terms(geno: Array, covar: Array) -> Array:
    m, n = covar.shape
    dtype = covar.dtype
    rcond = jnp.finfo(dtype).eps * max(n, m)
    u, s, vt = jnla.svd(covar, full_matrices=False)
    mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]

    safe_s = jnp.where(mask, s, 1).astype(covar.dtype)
    s_inv = jnp.where(mask, 1 / safe_s, 0)[:, jnp.newaxis]
    uTb = jnp.matmul(u.conj().T, geno, precision=lax.Precision.HIGHEST)

    beta = jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)

    return beta


def _get_dense_var(geno: sparse.JAXSparse, dense_dtype: JAXType):
    # def _inner(_, variant):
    #     var_idx = jnp.mean(variant **2) - jnp.mean(variant) ** 2
    #     return _, var_idx
    #
    # _, var_geno = lax.scan(_inner, 0.0, geno.T)
    var_geno = _sparse_mean(geno**2, axis=0, dtype=dense_dtype) - _sparse_mean(geno, axis=0, dtype=dense_dtype) ** 2

    return var_geno.todense()


class SparseMatrix(lx.AbstractLinearOperator):
    matrix: sparse.JAXSparse

    def __init__(self, matrix: sparse.JAXSparse):
        self.matrix = matrix

    def mv(self, vector: ArrayLike):
        return sparse.sparsify(jnp.matmul)(self.matrix, vector, precision=lax.Precision.HIGHEST)  # type: ignore

    def mm(self, matrix: Num[ArrayLike, "p k"]) -> Float[Array, "n k"]:
        return jax.vmap(self.mv, (1,), 1)(matrix)

    @dispatch
    def __matmul__(self, other: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
        return self.matrix @ other

    @dispatch
    def __matmul__(self, vector: Num[ArrayLike, " p"]) -> Float[Array, " n"]:
        return self.mv(vector)

    @dispatch
    def __matmul__(self, matrix: Num[ArrayLike, "p k"]) -> Float[Array, "p k"]:
        return self.mm(matrix)

    @dispatch
    def __rmatmul__(self, other: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
        return other @ self.matrix

    @dispatch
    def __rmatmul__(self, vector: Num[ArrayLike, " n"]) -> Float[Array, " p"]:
        return self.T.mv(jnp.asarray(vector).T).T

    @dispatch
    def __rmatmul__(self, matrix: Num[ArrayLike, "k n"]) -> Float[Array, "k p"]:
        return self.T.mm(jnp.asarray(matrix).T).T

    def as_matrix(self) -> Float[Array, "n p"]:
        # raise ValueError("Refusing to materialise sparse matrix.")
        # Or you could do:
        return self.matrix.todense()

    def transpose(self) -> "SparseMatrix":
        return SparseMatrix(self.matrix.T)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        _, in_size = self.matrix.shape
        return jax.ShapeDtypeStruct((in_size,), self.matrix.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        out_size, _ = self.matrix.shape
        return jax.ShapeDtypeStruct((out_size,), self.matrix.dtype)

    @property
    def shape(self):
        n, *_ = self.out_structure().shape
        p, *_ = self.in_structure().shape

        return n, p


class CenteredSparseMatrix(lx.AbstractLinearOperator):
    data: lx.AbstractLinearOperator

    @dispatch
    def __init__(self, data: lx.AbstractLinearOperator):
        self.data = data

    @dispatch
    def __init__(
        self,
        matrix: sparse.JAXSparse,
        covar: Optional[ArrayLike] = None,
        scale: bool = False,
    ):
        n, p = matrix.shape
        geno_op = SparseMatrix(matrix)
        dtype = matrix.dtype

        if covar is None:
            covar = jnp.ones((n, 1), dtype=dtype)
            beta = _sparse_mean(matrix, axis=0, dtype=dtype).todense()
            beta = beta.reshape((1, p))
        else:
            beta = _get_mean_terms(matrix, covar)

        center_op = lx.MatrixLinearOperator(covar) @ lx.MatrixLinearOperator(beta)

        if scale:
            wgt = jnp.sqrt(_get_dense_var(matrix, dtype))
            scale_op = lx.DiagonalLinearOperator(1.0 / wgt)
            self.data = (geno_op - center_op) @ scale_op
        else:
            self.data = geno_op - center_op

    @property
    def dense_dtype(self) -> JAXType:
        return self.out_structure().dtype

    def mv(self, vector: Num[ArrayLike, " p"]) -> Float[Array, " n"]:
        return self.data.mv(vector)

    def mm(self, matrix: Num[ArrayLike, "p k"]) -> Float[Array, "n k"]:
        return jax.vmap(self.data.mv, (1,), 1)(jnp.asarray(matrix))

    @dispatch
    def __matmul__(self, other: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
        return self.data @ other

    @dispatch
    def __matmul__(self, vector: Num[ArrayLike, " p"]) -> Float[Array, " n"]:
        return self.mv(vector)

    @dispatch
    def __matmul__(self, matrix: Num[ArrayLike, "p k"]) -> Float[Array, "p k"]:
        return self.mm(matrix)

    @dispatch
    def __rmatmul__(self, other: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
        return other @ self.data

    @dispatch
    def __rmatmul__(self, vector: Num[ArrayLike, " n"]) -> Float[Array, " p"]:
        return self.T.mv(jnp.asarray(vector).T).T

    @dispatch
    def __rmatmul__(self, matrix: Num[ArrayLike, "k n"]) -> Float[Array, "k p"]:
        return self.T.mm(jnp.asarray(matrix).T).T

    def as_matrix(self) -> Float[Array, "n p"]:
        return self.data.as_matrix()

    def transpose(self) -> "CenteredSparseMatrix":
        return CenteredSparseMatrix(self.data.T)

    @property
    def shape(self):
        n, *_ = self.out_structure().shape
        p, *_ = self.in_structure().shape

        return n, p

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self.data.in_structure()

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.data.out_structure()


@lx.is_symmetric.register(SparseMatrix)
@lx.is_symmetric.register(CenteredSparseMatrix)
def _(op):
    return False


@lx.is_negative_semidefinite.register(SparseMatrix)
@lx.is_negative_semidefinite.register(CenteredSparseMatrix)
def _(op):
    return False


@lx.linearise.register(SparseMatrix)
@lx.linearise.register(CenteredSparseMatrix)
def _(op):
    return op


@lx.conj.register(SparseMatrix)
def _(op):
    return SparseMatrix(op.matrix)


@lx.conj.register(CenteredSparseMatrix)
def _(op):
    return CenteredSparseMatrix(op.data)
