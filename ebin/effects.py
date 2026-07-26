"""Normal-effects parameterization of the bin profile Pi.

Each object carries a latent Gaussian effect law E_n ~ Normal(mu_n, sigma_n^2)
-- how the cells of that sequence spread over the measured phenotype.  The
library's marginal effect law is then the N-component mixture of those per-
sequence normals, and the B sorting bins are its quantile cells:

    G(x) = sum_n w_n * Phi((x - mu_n) / sigma_n),      w_n = 1 / N_observed,

with cut points q_1 < ... < q_{B-1} solving G(q_j) = j / B, so that

    Pi[n, b] = Phi((q_b - mu_n)/sigma_n) - Phi((q_{b-1} - mu_n)/sigma_n).

That spends 2 parameters per object instead of B-1, and ties every object to
the global marginal through the shared cuts (which move when any object moves).
The cuts are an implicit function of all (mu, sigma); gradients flow through the
root solve by the implicit function theorem.

Gauge freedom: the likelihood sees the parameters only through the z-scores
(q_j - mu_n)/sigma_n, so it is exactly invariant under
(mu, sigma, q) -> (a + k*mu, k*sigma, a + k*q), k > 0.  Location and scale of
the effect axis are conventions; ``gauge_normalize`` pins them (median cut = 0,
first cut = -1).  B >= 3 is required for per-object sigma to mean anything.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as _jnorm
from scipy.special import ndtri as _ndtri

jax.config.update("jax_enable_x64", True)

# Solver constants in standardized (mixture sd = 1) coordinates.  The step clamp
# tames Newton in flat CDF regions; the bracket contains any p-quantile with
# p in [1/2500, 1 - 1/2500] by Chebyshev, far wider than any 1/B.
_STEP_CLAMP = 3.0
_BRACKET = 50.0
_XATOL = 1e-13
_MAX_ITER = 200


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _newton_root(fun, dfun, args, x0):
    """Solve fun(x, args) = 0 by bracketed Newton from x0.

    ``fun`` must be increasing in x with derivative ``dfun``, and the root must
    lie in [x0 - _BRACKET, x0 + _BRACKET].  Every iterate tightens the sign
    bracket; a clamped Newton candidate that leaves the bracket is replaced by
    bisection, so isolated modes of a spiky mixture cannot be jumped over.
    """
    def cond(state):
        x_prev, i, x, lo, hi = state
        return (jnp.abs(x - x_prev) > _XATOL) & (i < _MAX_ITER)

    def body(state):
        _, i, x, lo, hi = state
        f = fun(x, args)
        lo = jnp.where(f < 0, jnp.maximum(lo, x), lo)
        hi = jnp.where(f >= 0, jnp.minimum(hi, x), hi)
        step = jnp.clip(f / dfun(x, args), -_STEP_CLAMP, _STEP_CLAMP)
        x_newton = x - step
        inside = (x_newton > lo) & (x_newton < hi)
        x_next = jnp.where(inside, x_newton, 0.5 * (lo + hi))
        return (x, i + 1, x_next, lo, hi)

    state = (x0 + 1.0, 0, x0, x0 - _BRACKET, x0 + _BRACKET)
    return jax.lax.while_loop(cond, body, state)[2]


@_newton_root.defjvp
def _newton_root_jvp(fun, dfun, primals, tangents):
    args, x0 = primals
    t_args, _ = tangents                       # the root is independent of x0
    x_star = _newton_root(fun, dfun, args, x0)
    # implicit function theorem: dx*/dargs = -(dfun/dargs) / (dfun/dx)
    _, j_args = jax.jvp(lambda a: fun(x_star, a), (args,), (t_args,))
    return x_star, -j_args / dfun(x_star, args)


def _mix_resid(t, args, p):
    """G(m0 + s0*t) - p for the weighted normal mixture."""
    mu, sigma, w, m0, s0 = args
    x = m0 + s0 * t
    return jnp.sum(w * _jnorm.cdf((x - mu) / sigma)) - p


def _mix_slope(t, args):
    """d/dt of _mix_resid (floored: a quantile in a density gap is genuinely
    ill-conditioned; the floor only prevents NaNs)."""
    mu, sigma, w, m0, s0 = args
    x = m0 + s0 * t
    dens = jnp.sum(w * _jnorm.pdf((x - mu) / sigma) / sigma)
    return jnp.maximum(dens * s0, 1e-300)


def mixture_quantiles(mu, sigma, weights, target_probs):
    """Cut points q_j with sum_n w_n Phi((q_j - mu_n)/sigma_n) = p_j.

    Differentiable in (mu, sigma) and in ``weights`` through the implicit
    function theorem; ``target_probs`` are constants.  The solve runs in
    standardized coordinates, so Newton's start and step clamp are scale-free;
    freezing the standardization costs nothing since the root is exact either
    way and all derivatives flow through the root.
    """
    target_probs = np.asarray(target_probs, dtype=float)
    m = jnp.sum(weights * mu)
    v = jnp.sum(weights * (sigma ** 2 + mu ** 2)) - m ** 2
    m0 = jax.lax.stop_gradient(m)
    s0 = jax.lax.stop_gradient(jnp.sqrt(jnp.clip(v, 1e-30, None)))
    args = (mu, sigma, weights, m0, s0)
    cuts = []
    for p in target_probs:
        fun = partial(_mix_resid, p=float(p))
        t0 = jnp.asarray(float(_ndtri(p)), dtype=jnp.float64)
        t_star = _newton_root(fun, _mix_slope, args, t0)
        cuts.append(m0 + s0 * t_star)
    return jnp.stack(cuts)


def normal_bin_probs(mu, sigma, cuts, floor=1e-12):
    """(N, B) bin probabilities Pi[n,b] = Phi(z_{n,b}) - Phi(z_{n,b-1}).

    Rows sum to 1 exactly; ``floor`` keeps every cell strictly positive (an
    object far outside the cuts would otherwise underflow whole bins to 0 and
    give -inf loglik for nonzero counts there).
    """
    z = (cuts[None, :] - mu[:, None]) / sigma[:, None]       # (N, B-1)
    cdf = _jnorm.cdf(z)
    zero = jnp.zeros((cdf.shape[0], 1), dtype=cdf.dtype)
    one = jnp.ones((cdf.shape[0], 1), dtype=cdf.dtype)
    raw = jnp.maximum(jnp.concatenate([cdf, one], axis=1)
                      - jnp.concatenate([zero, cdf], axis=1), 0.0)
    B = raw.shape[1]
    return (raw + floor) / (1.0 + B * floor)


def gauge_normalize(mu, sigma, cuts, pin_median=0.0, pin_left=-1.0):
    """Exact affine normalization of the (unidentified) effect-axis gauge.

    Maps (mu, sigma, cuts) -> (a + k*mu, k*sigma, a + k*cuts) so the middle cut
    equals ``pin_median`` and (if ``pin_left`` is given and B >= 3) the first cut
    equals ``pin_left``.  The likelihood is exactly invariant under this map.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cuts = np.asarray(cuts, dtype=float)
    mid = len(cuts) // 2
    if pin_left is not None and mid > 0:
        if pin_median <= pin_left:
            raise ValueError("pin_median must exceed pin_left")
        k = (cuts[mid] - cuts[0]) / (pin_median - pin_left)
        if k <= 0:
            raise ValueError("cuts must be increasing")
    else:
        k = 1.0
    mu2 = pin_median + (mu - cuts[mid]) / k
    cuts2 = pin_median + (cuts - cuts[mid]) / k
    return mu2, sigma / k, cuts2
