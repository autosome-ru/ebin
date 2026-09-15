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
summed on its own grid.  On a typical library that is ~390 nodes per object
against the 5462 an a_max=100 shared grid needs, so raising a_max costs a few
percent instead of 4.5x -- and stays inside a 16 GB card, which the shared grid
at that ceiling does not.

The ceiling is a BOUND, not an estimate: whatever supplies it must guarantee
that a_n * Pi[n,b] * lambda_s stays under it for every (b, s) the likelihood is
ever evaluated at, or the truncation silently drops real mass.  In the fit that
guarantee is the optimizer's own box constraint on a_n (fit.py); in the readout
and the mixture it is exact algebra on quantities that are already fixed.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln, logsumexp
from scipy.stats import poisson as _sp_poisson

from .gauss_rules import compute_nodes_and_logweights, Rules

jax.config.update("jax_enable_x64", True)

_NEG_INF = -jnp.inf

# no bucket is shorter than this: below it the grid costs nothing anyway and the
# extra kernel is not worth compiling
_MIN_TAU = 16

# Mode-centred rule: nodes per cell and the level set they span.
#
# MEASURED, against a brute-force lattice sum over 174 (R, P, m, x) cells
# spanning m in [0.5, 5e4] and x/(c m) in [0.02, 20]:
#
#     W   D     max |err|   median    cells > 1e-6
#    21  30      1.10e-03   3.3e-12       8 / 174
#    25  30      4.59e-04   4.4e-12       4 / 174
#    31  45      5.99e-04   9.5e-13       5 / 174
#    41  60      1.62e-06   4.0e-13       1 / 174
#    51  75      4.53e-08   3.2e-13       0 / 174
#    61  90      2.26e-08   3.2e-13       0 / 174
#
# The "0.68 decimal digits per node" balance assumes a GAUSSIAN bump and does
# not hold at the crossover.  The bad cells are all small m with small x, where
# sigma ~ 1.3-1.7 sits just past the Delta = 1 clamp: the lattice has been
# abandoned but the integral approximation is not yet valid.  Decomposed on the
# worst cell (R=1, P=0.5, m=5, x=2, err 1.1e-3):
#
#     truncation (integers <= 16 vs exact)   -4.3e-08   <- negligible
#     sum -> integral                        +4.9e-04   <- dominant
#     trapezoid -> integral                  +6.2e-04   <- dominant
#
# i.e. it is NOT truncation, which is why the edge-drop certificate cannot see
# it (see ``logmarg_window``).  W = 51 pushes the crossover to sigma ~ 3.4 where
# the Gaussian estimate does hold, and costs 51 constant nodes against the dense
# grid's 373 (shallow) to 207691 (a_max=4096).
WINDOW_NODES = 51
WINDOW_LEVEL = 75.0


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


def _trigamma(y):
    """psi'(y) by an upward-shifted asymptotic series.

    Only ever sets the NODE SPACING, never the value, so a relative error of
    1e-6 is irrelevant -- and it is a great deal cheaper than a true trigamma.
    """
    z = y + 3.0
    zi = 1.0 / z
    p = zi * (1.0 + 0.5 * zi + zi * zi / 6.0 - zi ** 4 / 30.0)
    return p + 1.0 / y ** 2 + 1.0 / (y + 1.0) ** 2 + 1.0 / (y + 2.0) ** 2


def mode_scale(x, m, R, P, iters=4):
    """(tau_hat, sigma): mode and curvature scale of the log-summand

        l(tau) = tau log m - m - lgamma(tau+1)
               + lgamma(x + R tau) - lgamma(R tau) - lgamma(x+1)
               + R tau log P + x log(1-P),

    which is strictly concave in tau for x > 0 (l'' = -psi'(tau+1)
    + R^2 [psi'(x+R tau) - psi'(R tau)], and psi' is decreasing), so the mode is
    unique.

    Two details are load-bearing, because a mode estimate that is silently wrong
    gives a window that is silently too narrow:

    * START FROM THE CLOSED FORM.  psi(y) ~ log y and then x >> R tau collapse
      the stationarity condition to tau_0 = [m (P x / R)^R]^(1/(R+1)), which is
      within a few percent of the mode across the whole range.  A naive start
      such as min(m, x/c) can be orders of magnitude out.
    * ITERATE IN u = log tau.  With F(u) = l'(e^u), F'(u) = tau l''(tau) lies in
      [-(1+R), -1]: monotone, bounded away from 0 and from infinity, so Newton
      needs no step clipping and converges in ~4 steps.  Newton in tau needs
      clipping to stay positive, and a clipped step cannot cross several orders
      of magnitude -- it stops short and reports a plausible wrong answer.

    Placement only: the caller stop_gradients this.
    """
    lo = jnp.log(jnp.maximum(m, 1e-300)) + R * jnp.log(P)
    u = (lo + R * jnp.log(jnp.maximum(x / R, 1e-300))) / (R + 1.0)
    u = jnp.clip(u, -30.0, 30.0)

    def step(u, _):
        t = jnp.exp(u)
        F = lo - digamma(t + 1.0) + R * (digamma(x + R * t) - digamma(R * t))
        Fp = t * (-_trigamma(t + 1.0)
                  + R ** 2 * (_trigamma(x + R * t) - _trigamma(R * t)))
        return jnp.clip(u - F / jnp.minimum(Fp, -1e-12), -30.0, 30.0), None

    u, _ = jax.lax.scan(step, u, None, length=int(iters))
    t = jnp.exp(u)
    neg_h = _trigamma(t + 1.0) - R ** 2 * (_trigamma(x + R * t)
                                           - _trigamma(R * t))
    return t, 1.0 / jnp.sqrt(jnp.maximum(neg_h, 1e-300))


def logmarg_window(x, m, R, P, W=WINDOW_NODES, D=WINDOW_LEVEL, certify=False):
    """(n,) log sum_tau Pois(tau|m) NB(x|R tau, P) on W mode-centred nodes.

    The dense grid spans the support of the PRIOR (length ~ m) at unit spacing,
    while the summand is a bump of width sigma = O(sqrt(m/(1+R))) sitting at its
    own mode.  Centring on that mode and spacing by the curvature removes both
    factors, so the work is O(W) with no dependence on m, on depth, or on a_max.

    Nodes span the l_max - D level set, half-width h = sqrt(2D) sigma, spacing
    Delta = max(1, 2h/(W-1)).  The clamp splits two regimes, and it is not a
    special case: Delta = 1 is exactly where a lattice stops resolving the bump.

    * Delta == 1 -- the nodes are consecutive integers and log Delta = 0, so
      this is an EXACT windowed lattice sum, arithmetically the dense grid
      restricted to the nodes carrying mass.  ~85% of cells in practice.
    * Delta > 1 -- a uniform rule on the reals, i.e. trapezoid for the integral.
      By Poisson summation the integer sum and the strided sum approximate the
      same integral and differ by O(exp(-2 pi^2 sigma^2 / Delta^2)); since Delta
      is proportional to sigma that is a CONSTANT, so accuracy does not decay
      with depth.

    Balancing window truncation exp(-D) against aliasing
    exp(-pi^2 (W-1)^2 / 4D) gives D_opt = pi (W-1)/2 and an error floor
    exp(-pi (W-1)/2) -- about 0.68 decimal digits per node, independent of
    everything else.

    ``certify=True`` also returns the edge drop l_max - l(edge).  Log-concavity
    makes the discarded tail a geometric series, so a drop >= D certifies a
    relative TRUNCATION error O(exp(-D)) -- a runtime property of the number
    actually computed, needing no bound on a_n.  Contrast the dense grid, whose
    criterion bounds only the ABSOLUTE prior mass dropped and can therefore be
    wrong in the log by an unbounded amount (see ``sum_mode``).

    **The certificate bounds truncation ONLY, and truncation is not the
    dominant error.**  Measured on the worst cell of a 174-cell sweep
    (R=1, P=0.5, m=5, x=2): the realised error is 1.1e-3 while the certificate
    reports exp(-14.6) = 4.7e-7, off by 2300x -- because truncation there is
    4.3e-8 and the error is really sum-vs-integral (4.9e-4) plus
    trapezoid-vs-integral (6.2e-4), neither of which moves the edge drop.  Do
    not use the drop as an error bound in the Delta > 1 regime; use it as what
    it is, a truncation check.  The defence against the other two is W (see
    WINDOW_NODES), which pushes the crossover to a sigma where the Gaussian
    aliasing estimate is actually valid.
    """
    x = jnp.asarray(x)
    # x == 0 has a closed form; the window branch still has to return something
    # FINITE there or jnp.where would propagate NaN into its gradient
    xs = jnp.maximum(x, 1.0)
    log_m = jnp.log(jnp.maximum(m, 1e-300))
    t_hat, sig = jax.lax.stop_gradient(mode_scale(xs, m, R, P))
    half = jnp.sqrt(2.0 * D) * sig
    delta = jnp.maximum(1.0, 2.0 * half / (W - 1.0))
    centre = jnp.where(delta <= 1.0, jnp.round(t_hat), t_hat)
    j = jnp.arange(W, dtype=jnp.float64) - (W - 1.0) / 2.0
    nodes = centre[:, None] + delta[:, None] * j[None, :]
    live = nodes > 0.0                       # tau <= 0 contributes nothing
    t = jnp.where(live, nodes, 1.0)          # keep the dead nodes finite
    lp = (t * log_m[:, None] - m[:, None] - gammaln(t + 1.0)
          + gammaln(xs[:, None] + R * t) - gammaln(R * t)
          - gammaln(xs[:, None] + 1.0)
          + R * t * jnp.log(P) + xs[:, None] * jnp.log1p(-P))
    lp = jnp.where(live, lp, _NEG_INF)
    # sum_tau Pois(tau|m) P^(R tau) = exp(m (P^R - 1)): the Poisson PGF at the
    # NB zero mass, exact and free (the same identity truncation.py uses)
    val = jnp.where(x == 0, m * (P ** R - 1.0),
                    logsumexp(lp, axis=1) + jnp.log(delta))
    if not certify:
        return val
    peak = jnp.max(lp, axis=1)
    inf = jnp.asarray(jnp.inf, dtype=lp.dtype)
    left = jnp.where(live[:, 0], peak - lp[:, 0], inf)      # a dead edge
    right = jnp.where(live[:, -1], peak - lp[:, -1], inf)   # truncates nothing
    return val, jnp.where(x == 0, inf, jnp.minimum(left, right))


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

    __slots__ = ("idx", "n", "tau", "lg", "xu", "inv", "mask", "xf")

    def __init__(self, idx, T, Xi, mask, tables=True):
        S, B = Xi.shape[1], Xi.shape[2]
        self.idx = jnp.asarray(idx)
        self.n = int(idx.size)
        Xg, mg = Xi[idx], mask[idx]
        self.mask = [[jnp.asarray(mg[:, s, b]) for b in range(B)]
                     for s in range(S)]
        if not tables:
            # the window rule evaluates one row per cell, so the unique-value
            # table has nothing to share and the tau grid does not exist
            self.tau = self.lg = self.xu = self.inv = None
            self.xf = [[jnp.asarray(Xg[:, s, b], dtype=jnp.float64)
                        for b in range(B)] for s in range(S)]
            return
        self.xf = None
        self.tau = jnp.arange(int(T), dtype=jnp.float64)
        self.lg = gammaln(self.tau + 1.0)
        self.xu = [[None] * B for _ in range(S)]
        self.inv = [[None] * B for _ in range(S)]
        for s in range(S):
            for b in range(B):
                xu, inv = np.unique(Xg[:, s, b], return_inverse=True)
                self.xu[s][b] = jnp.asarray(xu, dtype=jnp.float64)
                self.inv[s][b] = jnp.asarray(inv)


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
        restores a single shared grid.  Ignored by ``sum_mode="window"``.
    sum_mode : how the latent count is summed out.

        "grid" (default) -- the dense truncated lattice described above.
        "window" -- ``logmarg_window``: W nodes centred on the mode of the
            SUMMAND, spaced by its curvature.  Work is O(N S B W) with W a
            compile-time constant, so there is ONE kernel and no dependence on
            m, on depth or on ``a_max`` -- the bucketing machinery is not
            needed and ``rate_max`` is not read.

        The two also differ in CORRECTNESS, not only in cost.  The grid is
        sized from a quantile of the PRIOR, which implicitly assumes the
        summand peaks below it; when x > c m the reads argue for more latent
        cells than m supplies and the mode moves ABOVE m, and above the grid if
        a_max is tight.  Its criterion bounds the ABSOLUTE prior mass dropped,
        which is not a bound on the error in the log -- so it can be wrong by an
        unbounded amount while looking ordinary.  Measured on a low-depth
        line at a_max=15, two cells of 120000 were off by 180 and 59 nats (239
        between them), and the gradient by 33% at a_max=5, against a
        brute-force sum to tau=400000; the window rule matched it to 5e-10.  The
        affected cells are the deepest sequences -- the ones pinned at the
        ceiling.  Both agree to 1e-13 once a_max >= 40.
    window, window_level : W and D for ``sum_mode="window"``.
    """

    def __init__(self, X, mask=None, rate_max=200.0, tail_eps=1e-10,
                 max_buckets=6, sum_mode="grid", window=WINDOW_NODES,
                 window_level=WINDOW_LEVEL):
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

        if sum_mode not in ("grid", "window"):
            raise ValueError(f"sum_mode must be 'grid' or 'window', "
                             f"got {sum_mode!r}")
        self.sum_mode = sum_mode
        self.window = int(window)
        self.window_level = float(window_level)
        if sum_mode == "window":
            # no bound is needed and no grid is sized, so rate_max is not read;
            # one bucket in object order keeps every gather an identity
            self.rate_max = float(np.max(np.asarray(rate_max, np.float64)))
            self.lengths = np.array([self.window], dtype=np.int64)
            self.buckets = [_TauBucket(np.arange(self.N), 0, Xi, mask,
                                       tables=False)]
            self._flat = True
            self._unsort = None
            self.bucket_sizes = [self.N]
            self.tau_lengths = [self.window]
            self.tau_full = self.lg_tau_full = None
            return

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
        pay, against ``max(tau_lengths)`` for one shared grid.  Under
        ``sum_mode="window"`` it is W, by construction."""
        return sum(n * T for n, T in zip(self.bucket_sizes, self.tau_lengths)) \
            / max(self.N, 1)

    def _rows(self, bk, A):
        """The bucket's rows of a full (N, ...) array."""
        return A if self._flat else A[bk.idx]

    def _scatter(self, parts):
        """Bucket results, concatenated and put back in object order."""
        out = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)
        return out if self._flat else out[self._unsort]

    def _acc_over_rates(self, bk, s, R, P, Pi, rates_s):
        """(n_g, K) sum over b of the masked log-marginals, one column per rate.

        Scanned with a rematerialized body so only ONE rate's (n_g, T)
        intermediates are live at a time; a plain loop over k lets XLA allocate
        all K concurrently (tens of GB at N=30000).
        """
        mask_s = bk.mask[s]
        window = self.sum_mode == "window"
        tbls = None if window else self._nb_tables(bk, s, R, P)
        tau, lg = bk.tau, bk.lg
        W, D = self.window, self.window_level

        def body(carry, rate):
            acc_k = jnp.zeros(bk.n, dtype=jnp.float64)
            for b in range(self.B):
                m_b = Pi[:, b] * rate
                lm = (logmarg_window(bk.xf[s][b], m_b, R[s, b], P[s, b], W, D)
                      if window
                      else _logmarg_one_rate(tbls[b], m_b, tau, lg))
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

    @staticmethod
    def _rep(Pi_g, s):
        """This replicate's (n_g, B) rows of a profile block.

        A 2-D block is shared across replicates (the usual a_n * Pi[n,b]); a
        3-D (n, S, B) one carries a row per replicate, which is what a
        replicate-specific abundance a_{n,s} * Pi[n,b] produces.  Nothing else
        in the likelihood changes -- cells of different replicates are already
        independent given their rows.
        """
        return Pi_g if Pi_g.ndim == 2 else Pi_g[:, s]

    def _inner_rates(self, bk, R, P, Pi, rates, log_wmix):
        """(n_g,) with the latent-rate mixture resolved inside each replicate."""
        Pi_g = self._rows(bk, Pi)
        inner = jnp.zeros(bk.n, dtype=jnp.float64)
        for s in range(self.S):
            acc = self._acc_over_rates(bk, s, R, P, self._rep(Pi_g, s),
                                       rates[s])                    # (n_g, M)
            inner = inner + logsumexp(acc + log_wmix[s][None, :], axis=1)
        return inner

    def _inner_abund(self, bk, R, P, Pi, base_lambda, a_nodes, log_wa):
        """(n_g,) with the abundance resolved only AFTER summing over s and b --
        a_n couples every cell of the object (see ``object_loglik_abund``)."""
        Pi_g = self._rows(bk, Pi)
        acc = jnp.zeros((bk.n, a_nodes.shape[0]), dtype=jnp.float64)
        for s in range(self.S):
            acc = acc + self._acc_over_rates(
                bk, s, R, P, self._rep(Pi_g, s), a_nodes * base_lambda[s])
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
             abundance is fitted), or (N, S, B) when the abundance is
             replicate-specific, a_{n,s} * profile.  rates / log_wmix : (S, M)
             latent rates and their log-weights (M = 1 for the fixed-rate
             model; unrelated to the K emission components).  phi : object
             level zero-inflation probability.
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
