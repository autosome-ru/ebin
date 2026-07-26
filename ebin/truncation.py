"""Zero handling: the probability that an object's observed cells are all zero.

Two ways to treat the all-zero rows:

* zero-TRUNCATION (``conditional=True``): drop them and condition every kept
  object on Sum_{s,b} X > 0, i.e. subtract log(1 - P_n(0));
* zero-INFLATION (``conditional=False``): fit them, with a probability phi that
  an object is a structural zero.

P_n(0) has a closed form -- for one cell it is the Poisson PGF evaluated at the
NB zero mass,

    sum_tau Pois(tau | Pi_b r) NB(0 | R tau, P) = exp(Pi_b r (P^R - 1)),

so no tau summation is needed.
"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def log1mexp(x):
    """log(1 - e^x) for x < 0, stable on both ends."""
    return jnp.where(x > -0.6931471805599453,          # -log 2
                     jnp.log(-jnp.expm1(x)),
                     jnp.log1p(-jnp.exp(x)))


def log_zero_prob(R, P, Pi, rates, log_wmix, mask):
    """(N,) log P(all observed cells of the object are zero), phi = 0.

    Mirrors ``LoglikBuilder.object_loglik``: the rate mixture sits outside the
    product over b, per (object, replicate).
    """
    rates = jnp.atleast_2d(rates)
    log_wmix = jnp.atleast_2d(log_wmix)
    zeta = P ** R - 1.0                                  # (S, B) <= 0
    mzeta = jnp.where(mask, zeta[None, :, :], 0.0)       # (N, S, B)
    a = jnp.einsum("nb,nsb->ns", Pi, mzeta)              # (N, S)
    t = a[:, :, None] * rates[None, :, :] + log_wmix[None, :, :]
    return logsumexp(t, axis=2).sum(axis=1)


def log_zero_prob_abund(R, P, Pi, base_lambda, a_nodes, log_wa, mask):
    """(N,) log P(all observed cells zero) with the abundance integrated out.

    Given a_n = a the cells are independent, so log P(all zero | a) = a * c_n
    with c_n = sum_{observed (s,b)} Pi_nb * base_lambda[s] * (P^R - 1)_{s,b} <= 0,
    and marginally log P_n(0) = logsumexp_k [log_wa_k + a_k * c_n] -- the
    discretized-Gamma MGF at c_n.  Exact: no tau truncation enters.
    """
    base_lambda = jnp.atleast_1d(jnp.asarray(base_lambda))
    a_nodes = jnp.asarray(a_nodes)
    zeta = P ** R - 1.0                                  # (S, B) <= 0
    c = jnp.zeros(mask.shape[0], dtype=jnp.float64)
    for s in range(mask.shape[1]):
        contrib = Pi * (zeta[s] * base_lambda[s])[None, :]            # (N, B)
        c = c + jnp.sum(jnp.where(mask[:, s, :], contrib, 0.0), axis=1)
    return logsumexp(log_wa[None, :] + a_nodes[None, :] * c[:, None], axis=1)
