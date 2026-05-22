# Mathematical Issues In The Code

This note lists mathematical issues found by comparing `math.pdf` against the current implementation. It excludes general software bugs unless they change the model mathematics.

Status reflects the current working tree.

## 1. Bernoulli KL For `eta` Is Incomplete

**ISSUE:** `SparseGuideModel.kl_divergence` computes the KL for `eta` with `kl_discrete(params.p_hat, params.p)`.

Code path: `src/perturbvi/guide.py`

Status: fixed. `SparseGuideModel.kl_divergence` now uses full Bernoulli KL via `kl_bernoulli`.

This treats each `p_hat[k, g]` as if it were one state in a categorical probability vector. But `eta_{gk}` is Bernoulli, so each entry has two states:

```text
q(eta_{gk} = 1) = p_hat_{kg}
q(eta_{gk} = 0) = 1 - p_hat_{kg}
```

The current expression includes only:

```text
sum_{k,g} p_hat_{kg} log(p_hat_{kg} / p_g)
```

It misses:

```text
sum_{k,g} (1 - p_hat_{kg}) log((1 - p_hat_{kg}) / (1 - p_g))
```

**FIX:** Add a Bernoulli KL helper and use it for `eta`.

```python
def kl_bernoulli(q, p, eps=1e-8):
    q = jnp.clip(q, eps, 1 - eps)
    p = jnp.clip(p, eps, 1 - eps)
    return jnp.sum(
        jspec.xlogy(q, q / p)
        + jspec.xlogy(1 - q, (1 - q) / (1 - p))
    )
```

Then replace:

```python
kl_eta = kl_discrete(params.p_hat, params.p)
```

with:

```python
kl_eta = kl_bernoulli(params.p_hat, params.p[None, :])
```

## 2. `E[B B^T]` Is Treated As Diagonal

**ISSUE:** `SparseGuideModel.weighted_sumsq` uses only diagonal second moments of `B`.

Code path: `src/perturbvi/guide.py`

Status: fixed. `SparseGuideModel.weighted_sumsq` now uses the decomposition `||G E[B]||^2 + diag(G^T G)^T Var(B)`.

Current logic computes:

```python
mean_bb = jnp.sum((params.mean_beta**2 + params.var_beta) * params.p_hat.T, axis=1)
return _wgt_sumsq(self.guide_data, jnp.sqrt(mean_bb))
```

This corresponds to:

```text
sum_g (G^T G)_{gg} E[B_gk^2]
```

But factorization under `Q` only gives zero covariance. It does not make cross-moments zero:

```text
E[B_gk B_hk] = E[B_gk] E[B_hk],  g != h
```

For each factor `k`, the full second moment is:

```text
E[b_k b_k^T] = m_k m_k^T + diag(v_k)
```

where:

```text
m_{gk} = p_hat_{kg} mean_beta_{gk}
v_{gk} = p_hat_{kg}(mean_beta_{gk}^2 + var_beta_{gk}) - m_{gk}^2
```

The diagonal approximation is valid only when `G^T G` is diagonal, such as mutually exclusive one-hot perturbation assignments.

**FIX:** Choose one model contract.

If guide columns are required to be mutually exclusive or orthogonal, validate this at ingress and document it:

```text
G^T G must be diagonal for the sparse guide ELBO approximation.
```

If overlapping perturbations are allowed, compute the full term:

```text
sum_k tr(G^T G E[b_k b_k^T])
```

One dense expression is:

```python
S = G.T @ G
m = params.mean_beta * params.p_hat.T
second = (params.mean_beta**2 + params.var_beta) * params.p_hat.T
var = second - m**2
weighted = sum(
    m[:, k] @ S @ m[:, k] + jnp.sum(jnp.diag(S) * var[:, k])
    for k in range(params.z_dim)
)
```

The implementation avoids materializing `E[B B^T]` and uses:

```text
sum_k [ ||G m_k||^2 + sum_g ||G_g||^2 v_gk ]
```

## 3. Dense Guide Mode Is A Different Mathematical Model

**ISSUE:** `DenseGuideModel` implements deterministic dense regression for `B`, not the spike-and-slab variational model.

Code path: `src/perturbvi/guide.py`

Status: addressed by documenting `DenseGuideModel` as a deterministic dense-regression guide.

Current dense guide behavior:

```python
predict(params) = G @ mean_beta
kl_divergence(params) = 0
weighted_sumsq(params) = sum((G @ mean_beta)**2)
```

This is mathematically consistent with a deterministic least-squares `B`, but not with the spike-and-slab prior over `beta` and `eta` in `math.pdf`.

The sparse guide tracks:

```text
mean_beta, var_beta, p_hat, tau_beta, p
```

The dense guide ignores the posterior variance and inclusion probabilities entirely.

**FIX:** Make the model switch explicit.

Option A: Document that `p_prior is None` or `p_prior ~= 0` switches to a deterministic dense-regression guide with no spike-and-slab prior and no `B, eta` KL contribution.

Option B: Implement a dense spike-and-slab guide that includes:

```text
Q(beta_{gk} | eta_{gk} = 1)
Q(eta_{gk})
KL(Q(beta, eta) || P(beta, eta))
E[B B^T]
```

Use Option B if dense `G` can still represent perturbation assignments under the spike-and-slab model.

## 4. `p_hat` Update And ELBO Accounting Are Inconsistent

**ISSUE:** The coordinate update for `p_hat` uses the correct Bernoulli log-odds form, but the ELBO/KL term uses the incomplete Bernoulli KL from Issue 1.

Code path: `src/perturbvi/guide.py`

Status: fixed by replacing the `eta` KL term with full Bernoulli KL.

The update is:

```python
log_bf = 0.5 * (
    jnp.log(var_beta_g)
    + jnp.log(params.tau_beta)
    + (mean_beta_g**2) / var_beta_g
)
p_hat_g = nn.sigmoid(logit(params.p[gdx]) + log_bf)
```

This corresponds to:

```text
logit(p_hat_{gk}) = logit(p_g) + log BF_{gk}
```

But the ELBO penalty omits the `eta = 0` state.

**FIX:** Keep the `p_hat` update. Fix only the KL term with the Bernoulli KL in Issue 1.

## 5. ELBO Is Wrong If Overlapping Perturbations Are Allowed

**ISSUE:** When guide columns overlap, the coordinate updates and ELBO diagnostics no longer represent the same objective.

Code path: `src/perturbvi/guide.py`

Status: fixed through the decomposed `weighted_sumsq` expression from Issue 2.

The beta update uses residual products involving `G`, so it can respond to correlated or overlapping guide columns. But `weighted_sumsq` drops off-diagonal terms in `G^T G`, so the `Z | B` KL contribution used in the ELBO is missing cross terms.

The missing terms are:

```text
sum_k sum_{g != h} (G^T G)_{gh} E[B_{gk}] E[B_{hk}]
```

These terms vanish only if:

```text
(G^T G)_{gh} = 0 for g != h
```

or if the relevant posterior means are zero.

**FIX:** Either enforce an orthogonal/mutually exclusive guide design, or update `weighted_sumsq` to use the full `G^T G E[B B^T]` expression from Issue 2.
