"""Marginal likelihood of the compound NB-Poisson count model.

    T[n,s,b]            ~ Poisson(a_n * Pi[n,b] * lambda_s)     latent cells
    X[n,s,b] | T[n,s,b] ~ NB(R[s,b] * T[n,s,b], P[s,b])         reads

Pi[n, :] is the object's bin profile (a simplex row), a_n its abundance and
lambda_s the per-replicate latent level.  The reads of a cell are a sum over the
cells that landed in bin b, so the emission compounds with the Poisson: the
marginal pmf is a sum over the latent count tau, truncated at the point where
the Poisson tail is below ``tail_eps``.  NB tables are evaluated on the unique
count values of each channel and gathered back.

lambda is not identified upward -- it rides the R*lambda ridge, and only
products like c*Pi*lambda with c = R(1-P)/P are identified -- so it is held
fixed (``lambda_fix``) and R absorbs the scale.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from scipy.stats import poisson as _sp_poisson

from .gauss_rules import compute_nodes_and_logweights, Rules

jax.config.update("jax_enable_x64", True)

_NEG_INF = -jnp.inf


def nb_logpmf(x, r, p):
    """log NB(x | r, p), scipy convention (mean r(1-p)/p).  r may be
    non-integer; r == 0 is the point mass at 0."""
    r_safe = jnp.maximum(r, 1e-300)
    lp = (gammaln(x + r_safe) - gammaln(r_safe) - gammaln(x + 1.0)
          + r_safe * jnp.log(p) + x * jnp.log1p(-p))
    degenerate = jnp.where(x == 0, 0.0, _NEG_INF)
    return jnp.where(r > 0, lp, degenerate)


def poisson_trunc_bound(lam_max, tail_eps=1e-10, slack=5):
    """Smallest T with P(Poisson(lam_max) > T) < tail_eps, plus slack."""
    return int(_sp_poisson.isf(tail_eps, max(lam_max, 1e-6))) + 1 + slack


def gamma_mixture_rule(alpha, mean, num_nodes):
    """Gamma(alpha, theta=mean/alpha) discretized on ``num_nodes`` generalized
    Gauss-Laguerre nodes.  Differentiable in alpha through the custom-JVP
    tridiagonal eigensolver in gauss_rules.  Returns (nodes, log weights)."""
    nodes, logw = compute_nodes_and_logweights(
        num_nodes, alpha - 1.0, rule=Rules.GenLaguerre, norm=True)
    return nodes * (mean / alpha), logw


@jax.checkpoint
def _logmarg_one_rate(log_em_n, mu_n, tau, lg):
    """(N,) log sum_tau NB(x | R*tau, P) * Poisson(tau | mu_n) for one rate.

    Checkpointed: the (N, T) intermediates are recomputed in the backward pass
    instead of being stored."""
    log_pois = (tau[None, :]
                * jnp.log(jnp.maximum(mu_n, 1e-300))[:, None]
                - mu_n[:, None] - lg[None, :])
    return logsumexp(log_em_n + log_pois, axis=1)


class LoglikBuilder:
    """Precomputes the X-dependent tables and exposes traceable likelihoods.

    X : (N, S, B) counts (NaN = missing), mask : optional (N, S, B) bool.
    rate_max : upper bound on any latent rate; sizes the tau grid and must
        dominate the largest rate the optimizer can reach (a_max * lambda when
        an abundance is modelled).
    """

    def __init__(self, X, mask=None, rate_max=200.0, tail_eps=1e-10):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("X must have shape (N, S, B)")
        nan_mask = ~np.isnan(X) if np.issubdtype(X.dtype, np.floating) else \
            np.ones(X.shape, bool)
        mask = nan_mask if mask is None else (np.asarray(mask, bool) & nan_mask)
        Xi = np.where(mask, np.nan_to_num(X, nan=0.0), 0.0)
        if not np.allclose(Xi, np.round(Xi)):
            raise ValueError("X must contain (masked) non-negative integers")
        Xi = np.round(Xi).astype(np.int64)
        if (Xi < 0).any():
            raise ValueError("X must be non-negative")

        self.N, self.S, self.B = Xi.shape
        self.rate_max = float(rate_max)

        # per-channel unique values: each (s,b) NB table only covers the counts
        # that actually occur in that channel
        self.xu = [[None] * self.B for _ in range(self.S)]
        self.inv = [[None] * self.B for _ in range(self.S)]
        for s in range(self.S):
            for b in range(self.B):
                xu, inv = np.unique(Xi[:, s, b], return_inverse=True)
                self.xu[s][b] = jnp.asarray(xu, dtype=jnp.float64)
                self.inv[s][b] = jnp.asarray(inv)
        self.mask = jnp.asarray(mask)
        self.X = Xi
        self.all_zero = jnp.asarray((Xi * mask).sum(axis=(1, 2)) == 0)
        self.n_observed = int(mask.sum())

        T_full = poisson_trunc_bound(rate_max, tail_eps)
        self.tau_full = jnp.arange(T_full, dtype=jnp.float64)
        self.lg_tau_full = gammaln(self.tau_full + 1.0)

    def _log_nb_gathered(self, s, b, r, p, tau):
        """(N, T) log NB(x_nsb | r * tau, p) via the unique-value table."""
        log_nb = nb_logpmf(self.xu[s][b][:, None], r * tau[None, :], p)
        return log_nb[self.inv[s][b]]

    def _acc_over_rates(self, s, tbls, Pi, rates_s):
        """(N, K) sum over b of the masked log-marginals, one column per rate.

        Scanned with a rematerialized body so only one rate's (N, T)
        intermediates are live at a time; a plain loop over k lets XLA allocate
        all K concurrently (tens of GB at N=30000).
        """
        tau, lg = self.tau_full, self.lg_tau_full
        mask_s = [self.mask[:, s, b] for b in range(self.B)]

        def body(carry, rate):
            acc_k = jnp.zeros(self.N, dtype=jnp.float64)
            for b in range(self.B):
                lm = _logmarg_one_rate(tbls[b], Pi[:, b] * rate, tau, lg)
                acc_k = acc_k + jnp.where(mask_s[b], lm, 0.0)
            return carry, acc_k

        _, accT = jax.lax.scan(jax.checkpoint(body), None, rates_s)   # (K, N)
        return accT.T

    def _nb_tables(self, s, R, P):
        return [self._log_nb_gathered(s, b, R[s, b], P[s, b], self.tau_full)
                for b in range(self.B)]

    @staticmethod
    def _zero_inflate(inner, phi, all_zero):
        phi_c = jnp.clip(phi, 0.0, 1.0 - 1e-12)
        log_phi = jnp.log(jnp.maximum(phi_c, 1e-300))
        log_1mphi = jnp.log1p(-phi_c)
        with_zero = jnp.logaddexp(log_phi, log_1mphi + inner)
        return jnp.where(all_zero, with_zero, log_1mphi + inner)

    def object_loglik(self, R, P, Pi, rates, log_wmix, phi=0.0):
        """(N,) per-object log-likelihood.

        Pi : (N, B) rows fed to the latent Poisson (a_n * profile when an
             abundance is fitted).  rates / log_wmix : (S, K) latent rates and
             their log-weights (K = 1 for the fixed-rate model).  phi : object
             level zero-inflation probability.
        """
        rates = jnp.atleast_2d(rates)
        log_wmix = jnp.atleast_2d(log_wmix)
        inner = jnp.zeros(self.N, dtype=jnp.float64)
        for s in range(self.S):
            acc = self._acc_over_rates(s, self._nb_tables(s, R, P), Pi,
                                       rates[s])                     # (N, K)
            inner = inner + logsumexp(acc + log_wmix[s][None, :], axis=1)
        return self._zero_inflate(inner, phi, self.all_zero)

    def object_loglik_abund(self, R, P, Pi, base_lambda, a_nodes, log_wa,
                            phi=0.0):
        """(N,) per-object log-likelihood with the abundance integrated/marginalized out:

            a_n ~ sum_k exp(log_wa[k]) delta(a_nodes[k]),
            T[n,s,b] | a_n ~ Poisson(a_n * Pi[n,b] * base_lambda[s]).

        a_n couples every cell of the object, so the mixture is resolved
        (logsumexp over nodes) only AFTER summing the per-cell log-marginals
        over both s and b -- unlike ``object_loglik``, whose rate mixture is
        resolved within each replicate.  Nodes past the tau grid self-truncate;
        they carry negligible weight, and truncating only shrinks their
        contribution, so the bound is conservative.
        """
        a_nodes = jnp.asarray(a_nodes)
        base_lambda = jnp.atleast_1d(jnp.asarray(base_lambda))
        acc_total = jnp.zeros((self.N, a_nodes.shape[0]), dtype=jnp.float64)
        for s in range(self.S):
            acc_total = acc_total + self._acc_over_rates(
                s, self._nb_tables(s, R, P), Pi, a_nodes * base_lambda[s])
        inner = logsumexp(acc_total + log_wa[None, :], axis=1)
        return self._zero_inflate(inner, phi, self.all_zero)
