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

Emission mixture (K > 1).  The reads-per-cell scale c = R(1-P)/P is an
amplification efficiency, and in some libraries it is visibly bimodal across
sequences: the emission then becomes a K-component NB mixture over the SAME
latent T, with the component chosen once per sequence,

    X[n,s,b] | T[n,s,b], k_n = k ~ NB(R_k[s,b] * T[n,s,b], P_k[s,b]).

The ``*_components`` methods return the (N, K) log-likelihood of every object
under every component; the mixture over k is resolved outside, because k_n is
shared across cell lines (see mixture.py).  Note that a component is NOT
redundant with the abundance a_n: scaling a scales the marginal mean AND
variance in step (Fano stays 1/P + c), whereas scaling c at a matched mean moves
the Fano factor, so the two are separately identified.

Tau buckets.  How long the tau grid has to be is set by the largest latent rate
an object can reach, and that is a property of the OBJECT: a 400-read sequence
has no Poisson mass above tau ~ 50, while the deepest 1% of a library can need
thousands of nodes.  Sizing one grid by the deepest object and charging it to
all N is what makes a large ``a_max`` expensive.  So ``rate_max`` may be an (N,)
array of per-object rate ceilings; objects are then bucketed onto a few shared
grid lengths (chosen to minimize the total padded work) and each bucket is
summed on its own grid.  On the 3'UTR libraries that is ~390 nodes per object
against the 5462 an a_max=100 grid needs, so raising a_max costs a few percent
instead of 4.5x -- and stays inside a 16 GB card, which the shared grid at that
ceiling does not.

The ceiling is a BOUND, not an estimate: whatever supplies it must guarantee
that a_n * Pi[n,b] * lambda_s stays under it for every (b, s) the likelihood is
ever evaluated at, or the truncation silently drops real mass.  In the fit that
guarantee is the optimizer's own box constraint on a_n (fit.py); in the readout
and the mixture it is exact algebra on quantities that are already fixed.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from scipy.stats import poisson as _sp_poisson

from .gauss_rules import compute_nodes_and_logweights, Rules

jax.config.update("jax_enable_x64", True)

_NEG_INF = -jnp.inf

# no bucket is shorter than this: below it the grid costs nothing anyway and the
# extra kernel is not worth compiling
_MIN_TAU = 16


def nb_logpmf(x, r, p):
    """log NB(x | r, p), scipy convention (mean r(1-p)/p).  r may be
    non-integer; r == 0 is the point mass at 0."""
    r_safe = jnp.maximum(r, 1e-300)
    lp = (gammaln(x + r_safe) - gammaln(r_safe) - gammaln(x + 1.0)
          + r_safe * jnp.log(p) + x * jnp.log1p(-p))
    degenerate = jnp.where(x == 0, 0.0, _NEG_INF)
    return jnp.where(r > 0, lp, degenerate)


def as_components(A):
    """(S, B) -> (1, S, B); a (K, S, B) emission table passes through.

    ``jnp.atleast_3d`` appends the axis (giving (S, B, 1)), which is the wrong
    end for a leading component axis.
    """
    A = jnp.asarray(A)
    return A[None] if A.ndim == 2 else A


def poisson_trunc_bound(lam_max, tail_eps=1e-10, slack=5):
    """Smallest T with P(Poisson(lam_max) > T) < tail_eps, plus slack.

    Elementwise on an array ``lam_max``; a scalar comes back as a Python int.
    """
    lam = np.maximum(np.asarray(lam_max, dtype=np.float64), 1e-6)
    T = _sp_poisson.isf(tail_eps, lam).astype(np.int64) + 1 + slack
    return int(T) if np.ndim(lam_max) == 0 else T


def bucket_lengths(T_req, max_buckets=6, min_len=_MIN_TAU,
                   max_candidates=192):
    """Choose <= ``max_buckets`` grid lengths for the per-object needs ``T_req``.

    Every object is summed on the shortest chosen length that still covers it,
    so the padded work is sum_n L(n) and the lengths are picked to minimize
    exactly that -- an O(G * U^2) DP over the sorted distinct requirements
    (subsampled to ``max_candidates`` of them when there are more, which only
    gives up a little padding).  The longest length is always the largest
    requirement, so no object is ever truncated below what it asked for.

    Returns the ascending (G,) lengths; ``searchsorted`` maps an object to its
    bucket.
    """
    T = np.maximum(np.asarray(T_req, dtype=np.int64).ravel(), int(min_len))
    vals, counts = np.unique(T, return_counts=True)
    if max_buckets <= 1 or vals.size == 1:
        return vals[-1:].copy()
    if vals.size > max_candidates:
        keep = np.unique(np.linspace(0, vals.size - 1, max_candidates)
                         .round().astype(np.int64))
        vals_k = vals[keep]
        # every value rounds UP to the next kept candidate (the largest value is
        # kept, so searchsorted never runs off the end)
        counts = np.bincount(np.searchsorted(vals_k, vals), weights=counts,
                             minlength=vals_k.size)
        vals = vals_k
    U = vals.size
    G = min(int(max_buckets), U)
    cum = np.concatenate([[0], np.cumsum(counts)])          # (U + 1,)

    # dp[j] = least padded work covering vals[:j+1] with the current number of
    # buckets, the last of which ends at j; back[g, j] is where that bucket began
    dp = cum[1:] * vals
    back = np.zeros((G, U), dtype=np.int64)
    for g in range(1, G):
        # start[i] = dp_prev[i - 1] is the cost of everything below bucket start i
        prev = np.concatenate([[0.0], dp[:-1]])             # (U,) indexed by start
        cand = prev[None, :] + (cum[1:, None] - cum[None, :U]) * vals[:, None]
        cand = np.where(np.arange(U)[None, :] <= np.arange(U)[:, None],
                        cand, np.inf)                       # start <= end
        back[g] = cand.argmin(axis=1)
        dp = cand.min(axis=1)

    ends, j = [], U - 1
    for g in range(G - 1, -1, -1):
        ends.append(j)
        j = int(back[g, j]) - 1
        if j < 0:
            break
    return vals[np.array(sorted(ends), dtype=np.int64)]


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


class _TauBucket:
    """One group of objects summed on a shared tau grid.

    Holds the group's own unique-value NB tables: restricting them to the
    group's rows is what keeps the table build proportional to the group's work
    rather than to the whole line's.
    """

    __slots__ = ("idx", "n", "tau", "lg", "xu", "inv", "mask")

    def __init__(self, idx, T, Xi, mask):
        S, B = Xi.shape[1], Xi.shape[2]
        self.idx = jnp.asarray(idx)
        self.n = int(idx.size)
        self.tau = jnp.arange(int(T), dtype=jnp.float64)
        self.lg = gammaln(self.tau + 1.0)
        Xg, mg = Xi[idx], mask[idx]
        self.xu = [[None] * B for _ in range(S)]
        self.inv = [[None] * B for _ in range(S)]
        for s in range(S):
            for b in range(B):
                xu, inv = np.unique(Xg[:, s, b], return_inverse=True)
                self.xu[s][b] = jnp.asarray(xu, dtype=jnp.float64)
                self.inv[s][b] = jnp.asarray(inv)
        self.mask = [[jnp.asarray(mg[:, s, b]) for b in range(B)]
                     for s in range(S)]


class LoglikBuilder:
    """Precomputes the X-dependent tables and exposes traceable likelihoods.

    X : (N, S, B) counts (NaN = missing), mask : optional (N, S, B) bool.
    rate_max : upper bound on the latent rate a_n * Pi[n,b] * lambda_s.  A
        scalar bounds every object (one shared grid, the original behaviour);
        an (N,) array bounds each object separately, and the objects are then
        bucketed onto at most ``max_buckets`` grid lengths so a shallow object
        does not pay for a deep one.  It must be a genuine BOUND over the whole
        region the likelihood is evaluated in -- see the module docstring.
    max_buckets : how many distinct grid lengths to allow.  Each one is a
        separate XLA kernel, so this trades compile time against padding; 1
        restores a single shared grid.
    """

    def __init__(self, X, mask=None, rate_max=200.0, tail_eps=1e-10,
                 max_buckets=6):
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

        self.mask = jnp.asarray(mask)
        self.X = Xi
        self.all_zero = jnp.asarray((Xi * mask).sum(axis=(1, 2)) == 0)
        self.n_observed = int(mask.sum())

        rm = np.asarray(rate_max, dtype=np.float64)
        if rm.size not in (1, self.N) or not np.all(np.isfinite(rm)) \
                or np.any(rm <= 0):
            # a NaN or an inf here would size a grid silently wrong rather than
            # fail, and the result is a truncated likelihood that still looks
            # like a number
            raise ValueError("rate_max must be a positive finite scalar or an "
                             f"({self.N},) array of them")
        self.rate_max = float(rm.max())
        T_req = poisson_trunc_bound(np.broadcast_to(rm, (self.N,)), tail_eps)
        self.lengths = bucket_lengths(T_req, max_buckets=max_buckets)
        # the ladder's last entry is max(T_req), so this never truncates below
        # what an object asked for
        which = np.searchsorted(self.lengths, np.maximum(T_req, _MIN_TAU))
        self.buckets = [_TauBucket(np.flatnonzero(which == g), int(L), Xi, mask)
                        for g, L in enumerate(self.lengths)
                        if np.any(which == g)]
        order = np.concatenate([np.asarray(bk.idx) for bk in self.buckets])
        # one bucket holding every object in order: the gathers are identities
        # and are skipped, so a scalar rate_max reproduces the shared-grid
        # numbers exactly rather than merely to within a permutation
        self._flat = len(self.buckets) == 1 and np.array_equal(
            order, np.arange(self.N))
        self._unsort = None if self._flat else jnp.asarray(np.argsort(order))
        self.bucket_sizes = [bk.n for bk in self.buckets]
        self.tau_lengths = [int(bk.tau.shape[0]) for bk in self.buckets]
        # the longest grid, kept under the old name for callers that report it
        self.tau_full = self.buckets[-1].tau
        self.lg_tau_full = self.buckets[-1].lg

    @property
    def tau_work(self):
        """Mean tau nodes summed per object -- the cost the buckets actually
        pay, against ``max(tau_lengths)`` for one shared grid."""
        return sum(n * T for n, T in zip(self.bucket_sizes, self.tau_lengths)) \
            / max(self.N, 1)

    def _rows(self, bk, A):
        """The bucket's rows of a full (N, ...) array."""
        return A if self._flat else A[bk.idx]

    def _scatter(self, parts):
        """Bucket results, concatenated and put back in object order."""
        out = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)
        return out if self._flat else out[self._unsort]

    def _acc_over_rates(self, bk, s, tbls, Pi, rates_s):
        """(n_g, K) sum over b of the masked log-marginals, one column per rate.

        Scanned with a rematerialized body so only ONE rate's (n_g, T)
        intermediates are live at a time; a plain loop over k lets XLA allocate
        all K concurrently (tens of GB at N=30000).
        """
        tau, lg = bk.tau, bk.lg
        mask_s = bk.mask[s]

        def body(carry, rate):
            acc_k = jnp.zeros(bk.n, dtype=jnp.float64)
            for b in range(self.B):
                lm = _logmarg_one_rate(tbls[b], Pi[:, b] * rate, tau, lg)
                acc_k = acc_k + jnp.where(mask_s[b], lm, 0.0)
            return carry, acc_k

        _, accT = jax.lax.scan(jax.checkpoint(body), None, rates_s)  # (K, n_g)
        return accT.T

    def _nb_tables(self, bk, s, R, P):
        """(n_g, T) log NB(x_nsb | R*tau, P) per bin, via the unique-value
        table of this bucket."""
        return [nb_logpmf(bk.xu[s][b][:, None],
                          R[s, b] * bk.tau[None, :],
                          P[s, b])[bk.inv[s][b]]
                for b in range(self.B)]

    def _over_buckets(self, fn):
        """(N,) from a per-bucket inner log-likelihood, back in object order."""
        return self._scatter([fn(bk) for bk in self.buckets])

    def _inner_rates(self, bk, R, P, Pi, rates, log_wmix):
        """(n_g,) with the latent-rate mixture resolved inside each replicate."""
        Pi_g = self._rows(bk, Pi)
        inner = jnp.zeros(bk.n, dtype=jnp.float64)
        for s in range(self.S):
            acc = self._acc_over_rates(bk, s, self._nb_tables(bk, s, R, P),
                                       Pi_g, rates[s])              # (n_g, M)
            inner = inner + logsumexp(acc + log_wmix[s][None, :], axis=1)
        return inner

    def _inner_abund(self, bk, R, P, Pi, base_lambda, a_nodes, log_wa):
        """(n_g,) with the abundance resolved only AFTER summing over s and b --
        a_n couples every cell of the object (see ``object_loglik_abund``)."""
        Pi_g = self._rows(bk, Pi)
        acc = jnp.zeros((bk.n, a_nodes.shape[0]), dtype=jnp.float64)
        for s in range(self.S):
            acc = acc + self._acc_over_rates(
                bk, s, self._nb_tables(bk, s, R, P), Pi_g,
                a_nodes * base_lambda[s])
        return logsumexp(acc + log_wa[None, :], axis=1)

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
             abundance is fitted).  rates / log_wmix : (S, M) latent rates and
             their log-weights (M = 1 for the fixed-rate model; unrelated to
             the K emission components).  phi : object level zero-inflation
             probability.
        """
        rates = jnp.atleast_2d(rates)
        log_wmix = jnp.atleast_2d(log_wmix)
        inner = self._over_buckets(
            lambda bk: self._inner_rates(bk, R, P, Pi, rates, log_wmix))
        return self._zero_inflate(inner, phi, self.all_zero)

    def object_loglik_abund(self, R, P, Pi, base_lambda, a_nodes, log_wa,
                            phi=0.0):
        """(N,) per-object log-likelihood with the abundance INTEGRATED OUT:

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
        inner = self._over_buckets(lambda bk: self._inner_abund(
            bk, R, P, Pi, base_lambda, a_nodes, log_wa))
        return self._zero_inflate(inner, phi, self.all_zero)

    def _scan_components(self, R, P, inner_fn):
        """(N, K) from a per-component inner log-likelihood.

        Scanned over the component axis for the same reason ``_acc_over_rates``
        scans over the rates: a Python loop lets XLA keep every component's
        (N, T) tables live at once.  The structural-zero wrap is applied to the
        whole (N, K) block afterwards -- phi is a property of the cell line, not
        of the emission component.
        """
        R, P = as_components(R), as_components(P)

        def body(carry, RP):
            return carry, inner_fn(RP[0], RP[1])

        _, innerT = jax.lax.scan(body, None, (R, P))            # (K, N)
        return innerT.T

    def object_loglik_components(self, R, P, Pi, rates, log_wmix, phi=0.0):
        """(N, K) per-object log-likelihood under each emission component.

        R, P : (K, S, B).  Otherwise exactly ``object_loglik``, once per
        component; the mixture over components is NOT resolved here.
        """
        rates = jnp.atleast_2d(rates)
        log_wmix = jnp.atleast_2d(log_wmix)

        def inner_fn(Rk, Pk):
            return self._over_buckets(
                lambda bk: self._inner_rates(bk, Rk, Pk, Pi, rates, log_wmix))

        return self._zero_inflate(self._scan_components(R, P, inner_fn), phi,
                                  self.all_zero[:, None])

    def object_loglik_abund_components(self, R, P, Pi, base_lambda, a_nodes,
                                       log_wa, phi=0.0):
        """(N, K) version of ``object_loglik_abund``: the abundance is
        integrated out separately under each emission component."""
        a_nodes = jnp.asarray(a_nodes)
        base_lambda = jnp.atleast_1d(jnp.asarray(base_lambda))

        def inner_fn(Rk, Pk):
            return self._over_buckets(lambda bk: self._inner_abund(
                bk, Rk, Pk, Pi, base_lambda, a_nodes, log_wa))

        return self._zero_inflate(self._scan_components(R, P, inner_fn), phi,
                                  self.all_zero[:, None])
