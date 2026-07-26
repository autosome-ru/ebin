"""Starting values for the compound NB-Poisson model.

1. Fit a zero-inflated NB (ZINB) independently to each of the S*B count columns.
2. Take a W2 barycenter of those NBs (location-scale surrogate: mean of means,
   mean of sds) as a reference NB.
3. Map every column onto the reference by the mid-distribution quantile
   transform, Z = F_ref^{-1}(F_sb(x) - 0.5 f_sb(x)); normalize per column, sum
   over replicates, and row-normalize to get Pi0.
4. R0 = r_hat / (Lambda0_s * mean_n Pi0) matches the model means to the
   marginal fits, and P0 = (1 + R0) / (1/p_hat + R0) matches their dispersion.
"""

from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from scipy.optimize import minimize
from scipy.stats import nbinom as sp_nbinom

jax.config.update("jax_enable_x64", True)


@jax.jit
def _zinb_nll_grad(theta, xu, w):
    def nll(theta):
        a, c, d = theta
        r = jnp.exp(a)
        logp = -jnp.logaddexp(0.0, -c)          # log sigmoid(c)
        log1mp = -jnp.logaddexp(0.0, c)
        log_pi0 = -jnp.logaddexp(0.0, -d)
        log_1mpi0 = -jnp.logaddexp(0.0, d)
        log_nb = (gammaln(xu + r) - gammaln(r) - gammaln(xu + 1.0)
                  + r * logp + xu * log1mp)
        log_zinb = jnp.where(
            xu == 0,
            jnp.logaddexp(log_pi0, log_1mpi0 + log_nb),
            log_1mpi0 + log_nb)
        return -jnp.sum(w * log_zinb) / jnp.sum(w)
    return jax.value_and_grad(nll)(theta)


def _pad_pow2(a, fill):
    n = 1
    while n < a.size:
        n *= 2
    out = np.full(n, fill, dtype=np.float64)
    out[:a.size] = a
    return out


def fit_zinb(x, weights=None):
    """Fit ZINB(r, p, pi0) to counts ``x``.  Returns (r, p, pi0, nll_per_obs)."""
    x = np.asarray(x, dtype=np.float64)
    if weights is None:
        xu, w = np.unique(x, return_counts=True)
        w = w.astype(np.float64)
    else:
        xu, w = x, np.asarray(weights, dtype=np.float64)
    # pad to a power of two so jit recompiles only O(log U) shapes
    xu_p = _pad_pow2(xu, 1.0)
    w_p = _pad_pow2(w, 0.0)

    wt = w.sum()
    m = float((xu * w).sum() / wt)
    v = float((w * (xu - m) ** 2).sum() / wt)
    frac0 = float(w[xu == 0].sum() / wt) if (xu == 0).any() else 0.0
    m_pos = m / max(1.0 - frac0, 1e-6)          # crude nonzero-component mean
    v_pos = max(v, m_pos * 1.5)
    r0 = np.clip(m_pos ** 2 / max(v_pos - m_pos, 1e-6), 1e-3, 1e6)
    p0 = np.clip(r0 / (r0 + m_pos), 1e-6, 1 - 1e-6)
    pi0_0 = np.clip(frac0 * 0.5 + 1e-4, 1e-4, 0.9)
    theta0 = np.array([np.log(r0), np.log(p0 / (1 - p0)),
                       np.log(pi0_0 / (1 - pi0_0))])

    def fg(theta):
        v, g = _zinb_nll_grad(jnp.asarray(theta), jnp.asarray(xu_p),
                              jnp.asarray(w_p))
        return float(v), np.asarray(g)

    res = minimize(fg, theta0, jac=True, method="L-BFGS-B",
                   bounds=[(-12, 18), (-25, 25), (-12, 12)],
                   options=dict(maxiter=300))
    a, c, d = res.x
    return (float(np.exp(a)), float(1 / (1 + np.exp(-c))),
            float(1 / (1 + np.exp(-d))), float(res.fun))


@dataclass
class InitResult:
    R: np.ndarray            # (S, B)
    P: np.ndarray            # (S, B)
    Pi: np.ndarray           # (N, B)
    Lambda: np.ndarray       # (S,)
    phi: float
    r_hat: np.ndarray = field(repr=False, default=None)   # marginal ZINB fits
    p_hat: np.ndarray = field(repr=False, default=None)
    pi0_hat: np.ndarray = field(repr=False, default=None)
    r_ref: float = None
    p_ref: float = None


def initialize(X, mask=None, lambda_init=50.0, verbose=False):
    """Starting values (R0, P0, Pi0, Lambda0, phi0) for ``fit_normal``.

    ``lambda_init`` mostly sets the resolution of the latent Poisson; R0
    absorbs the scale.
    """
    X = np.asarray(X, dtype=np.float64)
    N, S, B = X.shape
    nan_mask = ~np.isnan(X)
    mask = nan_mask if mask is None else (np.asarray(mask, bool) & nan_mask)
    Xz = np.where(mask, np.nan_to_num(X), 0.0)

    # ---- 1. per-column ZINB fits -------------------------------------------
    r_hat = np.zeros((S, B))
    p_hat = np.zeros((S, B))
    pi0_hat = np.zeros((S, B))
    for s in range(S):
        for b in range(B):
            xs = Xz[:, s, b][mask[:, s, b]]
            r_hat[s, b], p_hat[s, b], pi0_hat[s, b], nll = fit_zinb(xs)
            if verbose:
                print(f"  ZINB[{s},{b}]: r={r_hat[s,b]:.3f} "
                      f"p={p_hat[s,b]:.4f} pi0={pi0_hat[s,b]:.4f} "
                      f"nll={nll:.4f}")

    # ---- 2. W2 barycenter reference NB -------------------------------------
    means = r_hat * (1 - p_hat) / p_hat
    sds = np.sqrt(means / p_hat)
    mu_ref = float(means.mean())
    sd_ref = float(sds.mean())
    var_ref = max(sd_ref ** 2, mu_ref * (1 + 1e-6))
    p_ref = np.clip(mu_ref / var_ref, 1e-8, 1 - 1e-8)
    r_ref = mu_ref * p_ref / (1 - p_ref)

    # ---- 3. quantile transform each column onto the reference --------------
    Z = np.zeros((N, S, B))
    for s in range(S):
        for b in range(B):
            obs = mask[:, s, b]
            xu, inv = np.unique(Xz[:, s, b], return_inverse=True)
            u = (sp_nbinom.cdf(xu, r_hat[s, b], p_hat[s, b])
                 - 0.5 * sp_nbinom.pmf(xu, r_hat[s, b], p_hat[s, b]))
            u = np.clip(u, 1e-12, 1 - 1e-12)
            z = sp_nbinom.ppf(u, r_ref, p_ref).astype(np.float64)[inv]
            fill = z[obs].mean() if obs.any() else 0.0
            Z[:, s, b] = np.where(obs, z, fill)

    # ---- 4. normalize -> Pi0 ------------------------------------------------
    Z = Z / np.maximum(Z.sum(axis=0, keepdims=True), 1e-300)
    Znb = Z.sum(axis=1)                                     # (N, B)
    row = Znb.sum(axis=-1, keepdims=True)
    Pi0 = np.where(row > 0, Znb / np.maximum(row, 1e-300), 1.0 / B)
    Pi0 = np.clip(Pi0, 1e-8, 1.0)
    Pi0 = Pi0 / Pi0.sum(axis=-1, keepdims=True)

    # ---- 5. Lambda0, R0, P0, phi0 ------------------------------------------
    Lambda0 = np.broadcast_to(np.asarray(lambda_init, dtype=np.float64),
                              (S,)).copy()
    pi_bar = Pi0.mean(axis=0)                               # (B,)
    R0 = r_hat / (Lambda0[:, None] * pi_bar[None, :])
    P0 = np.clip((1.0 + R0) / (1.0 / p_hat + R0), 1e-6, 1 - 1e-6)
    all_zero = (Xz * mask).sum(axis=(1, 2)) == 0
    phi0 = float(np.clip(all_zero.mean(), 1e-4, 0.5))

    return InitResult(R=R0, P=P0, Pi=Pi0, Lambda=Lambda0, phi=phi0,
                      r_hat=r_hat, p_hat=p_hat, pi0_hat=pi0_hat,
                      r_ref=float(r_ref), p_ref=float(p_ref))


def init_from_fit(fit):
    """Build an ``InitResult`` from a previous fit, for warm starts.

    The fitted profile Pi is enough to recover (mu, sigma); R/P/phi carry the
    global block.  Useful when refitting the same data under a different
    abundance model.
    """
    get = (lambda k: fit[k]) if isinstance(fit, dict) \
        else (lambda k: getattr(fit, k))
    Pi = np.asarray(get("Pi"), float).copy()
    Pi[~np.isfinite(Pi).all(1)] = 1.0 / Pi.shape[1]     # fully-missing rows
    return InitResult(R=np.asarray(get("R")), P=np.asarray(get("P")), Pi=Pi,
                      Lambda=np.asarray(get("rates")).reshape(-1),
                      phi=float(get("phi")))
