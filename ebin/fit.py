"""Maximum-marginal-likelihood fit of the normal-effects compound model.

Parameters split into a global block theta = [log R, logit P, logit phi] (S*B
channels) and a per-object block eta = [mu, log sigma, (log a | log kappa),
(log delta, eps)].
The optimizer runs a short block-coordinate warmup (global block over a
SLSQP/TNC/L-BFGS-B cycle, then the eta block) followed by joint finishing rounds
(TNC, falling back to L-BFGS-B).  Steps are accepted on the PENALIZED objective,
which is what ``minimize`` actually descends.

Regularization (all in the pinned gauge, where the cut spacing is 1):

* ``sigma_prior``  Normal prior on log sigma_n.  The per-object MLE is
  unbounded: an object whose reads land only in the outer bins wants the
  profile (t, 0, 0, 1-t), reachable only as sigma -> inf along mu = c*sigma,
  where the Gaussian degenerates into two point masses.  Such objects otherwise
  run to the log-sigma bound.
* ``sigma_shared``  the hard version of the same prior: ONE effect scale for
  the whole cell line instead of one per object (an ordered probit with
  homoskedastic latent).  Note the effect axis has no units -- the likelihood
  sees only the z-scores (q_j - mu_n)/sigma_n -- so "all sigmas equal to 1" and
  "all sigmas equal to any other common constant" are the SAME model, reached
  from each other by the gauge map.  What is fitted is therefore one free
  scalar, reported in the pinned gauge where the cut spacing is 1; only its
  RATIO to the cut spacing is identified.
* ``mu_prior``  Normal prior on the effect means.  sigma_prior bounds the
  effect SPREAD but not its LOCATION, so a single-bin low-count object still
  runs mu_n large and its Pi walks to a simplex vertex (activity exactly 1 or
  B).  The gauge pins Q50 = 0 and Q25 = -1, i.e. population sd ~ 1.5 and
  between-object sd ~ 1.1, so the gauge-consistent tau is ~1.3, not 3.
* ``eps_prior``  Normal prior on the sinh-arcsinh skew, and only interesting
  when that skew is per object: with B = 4 bins an object has 3 degrees of
  freedom, so (mu, sigma, eps) per object is a SATURATED profile -- free Pi in
  disguise, with free Pi's no-shrinkage behaviour.  The prior is what makes it a
  regularized free Pi rather than a reparameterized one.
* the soft gauge pin, which costs nothing at the optimum (the likelihood
  gradient along the gauge is exactly zero) and is applied exactly on exit.

Effect law.  ``shash`` replaces the Normal link by the sinh-arcsinh family
(effects.py): ``"global"`` fits one (delta, eps) for the cell line, ``"eps"``
fits a shared delta and a per-object eps.  Both nest the Normal exactly at
(delta, eps) = (1, 0), which is where the fit starts, so the likelihood can only
improve and the difference is a clean nested test.  The gauge is unaffected --
the shape acts on the z-score (q_b - mu_n)/sigma_n, which the affine map leaves
alone -- so ``gauge_normalize`` and the pins apply verbatim.

Abundance.  Without one, every object has the same expected latent total
(lambda_s), so any real depth variation has nowhere to go but Pi, which tilts
toward the bins with the largest channel scale c = R(1-P)/P -- Pi stops being a
profile and becomes an abundance proxy.  Either fit a free a_n
(``abundance=True``) or integrate it out under a Gamma prior
(``abundance_prior='gamma'``), which costs one global parameter instead of N and
propagates the abundance uncertainty into the effect.

The abundance bound.  a_n is box-constrained, and the bound does double duty: it
is a modelling choice AND it sizes the truncated latent-count grid, so a
generous one used to be paid for by every object at every step.  Both jobs are
now per object.  ``a_cap`` is an (N,) ceiling, started at a small multiple of
the object's own depth and doubled on whatever objects reach it, until fewer
than ``bound_frac`` of them sit at their bound (``fit_effects_adaptive``, which
is what ``a_max="auto"`` runs).  A constraint that is inactive does not move the
optimum, so the escalation converges to the unconstrained fit; what it buys is
that the tau grid is sized per object, and a deep sequence no longer makes a
shallow one pay.  ``a_max`` remains the hard global ceiling -- a number pins it
where it is, "auto" lets it escalate too.

Emission mixture (K > 1).  The channel block then holds K sets of (R, P) and the
data term is the RESPONSIBILITY-WEIGHTED log-likelihood

    sum_n sum_k gamma[n,k] * loglik(n | component k),

with ``gamma`` fixed.  That is the M-step of an EM whose latent component index
k_n is shared by every cell line, which is why the weights come in from outside
and are not estimated here (see mixture.py).  The per-object block eta is shared
across components: a sequence has one effect law and one abundance, and only its
emission differs.
"""

import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.special import ndtri

from .model import LoglikBuilder, gamma_mixture_rule
from .batch import broadcast_tilt
from .initialize import initialize
from .effects import mixture_quantiles, normal_bin_probs, gauge_normalize
from .truncation import (log_zero_prob, log_zero_prob_abund, log1mexp,
                         log_zero_prob_components,
                         log_zero_prob_abund_components)

jax.config.update("jax_enable_x64", True)

_GLOBAL_CYCLE = ["SLSQP", "TNC", "L-BFGS-B"]

# bound on log R in the channel block.  Exported because anything that writes R
# back into a fit has to respect the SAME bound: a tighter one silently rewrites
# the fit after its objective was evaluated (see mixture.ridge_step).
LOG_R_BOUND = 20.0


def initial_abundance(X, mask, kept):
    """(N,) starting size factor: the object's per-cell read total relative to
    the geometric mean over the objects that count.  This is the quantity a_n
    exists to explain, and the fit both starts a_n here and sizes its bound
    from it."""
    Xz = np.where(np.asarray(mask, bool), np.nan_to_num(np.asarray(X), nan=0.0),
                  0.0)
    cells = np.maximum(np.asarray(mask, bool).sum(axis=(1, 2)), 1)
    rate_n = Xz.sum(axis=(1, 2)) / cells
    ref = np.exp(np.mean(np.log(rate_n[kept] + 1e-6))) if np.any(kept) else 1.0
    return rate_n / max(ref, 1e-12)


def abundance_caps(X, mask, kept, a_max, headroom=3.0, floor=1.0):
    """(N,) per-object ceiling on a_n: ``headroom`` times the depth-derived
    start, floored so a low-count object can still move and clipped to the hard
    ``a_max``.

    This is what sizes the object's tau grid, so a tight headroom is cheap and a
    loose one is not; anything that turns out to bind is doubled by
    ``fit_effects_adaptive`` rather than guessed right in advance.
    """
    a0 = initial_abundance(X, mask, kept)
    return np.clip(headroom * a0, floor, a_max)


@dataclass
class FitResult:
    R: np.ndarray                 # (S, B) emission size       -- (K, S, B) if K > 1
    P: np.ndarray                 # (S, B) emission probability -- likewise
    mu: np.ndarray                # (N,) effect means      (pinned gauge)
    sigma: np.ndarray             # (N,) effect sds        (pinned gauge)
    cuts: np.ndarray              # (B-1,) shared bin cut points
    Pi: np.ndarray                # (N, B) implied bin profile
    rates: np.ndarray             # (S, 1) latent Poisson rates (lambda_fix)
    log_w: np.ndarray             # (S, 1) their log weights
    phi: float                    # structural-zero probability (0 if truncated)
    loglik: float                 # K > 1: the M-step's gamma-weighted data
                                  # term, not the observed-data log-likelihood
                                  # (which only exists across all cell lines)
    n_observed: int
    converged: bool
    observed: np.ndarray          # (N,) objects entering the fit
    a: np.ndarray = None          # (N,) fitted abundance (ones if not fitted)
    a_cap: np.ndarray = None      # (N,) per-object bound a_n was fitted under
    delta: float = None           # sinh-arcsinh tail weight (None = Normal law)
    eps: np.ndarray = None        # its skew: scalar, or (N,) when per object
    n_zero_dropped: int = 0       # all-zero rows removed by the truncation
    extras: dict = None           # kappa / abundance CV under a Gamma prior
    history: list = field(repr=False, default=None)
    config: dict = field(repr=False, default=None)


class StoredFit:
    """The part of a fit the readout needs, as loaded from disk."""

    def __init__(self, d):
        self.__dict__.update(d)


def light(res):
    """A ``FitResult`` without the optimizer history: everything the readout
    reads, plus the fitted per-object effect law and abundance."""
    return dict(R=res.R, P=res.P, rates=res.rates, log_w=res.log_w,
                cuts=res.cuts, Pi=res.Pi, mu=res.mu, sigma=res.sigma, a=res.a,
                a_cap=res.a_cap, delta=res.delta, eps=res.eps,
                observed=res.observed, phi=res.phi,
                loglik=res.loglik, converged=res.converged, extras=res.extras,
                config=res.config)


def save_fits(path, fits):
    with open(path, "wb") as f:
        pickle.dump(fits, f)


def load_fits(path):
    with open(path, "rb") as f:
        return {g: StoredFit(d) for g, d in pickle.load(f).items()}


def _solver_options(method, budget):
    """Per-solver budgets and explicit tolerances.

    The objective is scaled by 1/n_observed, so scipy's default stopping tests
    fire while thousands of nats remain.  These are deliberately tight; the
    iteration budget and the outer test (tol * n_obs) are the real stopping
    rules.
    """
    if method == "TNC":
        return dict(maxfun=int(budget), ftol=1e-12, xtol=1e-14, gtol=1e-11)
    if method == "L-BFGS-B":
        return dict(maxiter=int(budget), maxcor=20, ftol=1e-14, gtol=1e-11)
    if method == "SLSQP":
        return dict(maxiter=int(budget), ftol=1e-12)
    return dict(maxiter=int(budget))


def _res_line(res, scale):
    msg = str(getattr(res, "message", ""))
    return (f"status={getattr(res, 'status', '?')} nit={getattr(res, 'nit', -1)} "
            f"nfev={getattr(res, 'nfev', -1)} ll={-res.fun / scale:.4f} "
            f"| {msg[:60]}")


class _Globals:
    """The channel block: theta = [log R (K*S*B), logit P (K*S*B), logit phi].

    With K = 1 the layout is exactly the single-emission one, so a theta packed
    here is interchangeable with the pre-mixture code.  phi is NOT per component:
    a structural zero is a property of the cell line (the sequence never made it
    into this library), not of how well it amplifies.
    """

    def __init__(self, S, B, lambda_fix, K=1):
        self.S, self.B, self.K = S, B, int(K)
        self.lambda_fix = float(lambda_fix)
        n = self.K * S * B
        self.idx_R = slice(0, n)
        self.idx_P = slice(n, 2 * n)
        self.idx_phi = 2 * n
        self.n_params = 2 * n + 1

    def _components(self, A, name):
        """Broadcast an (S, B) emission table over the K components."""
        A = np.asarray(A, dtype=float)
        if A.shape == (self.S, self.B):
            return np.broadcast_to(A, (self.K, self.S, self.B))
        if A.shape == (self.K, self.S, self.B):
            return A
        raise ValueError(f"init.{name} has shape {A.shape}; expected "
                         f"{(self.S, self.B)} or {(self.K, self.S, self.B)}")

    def pack(self, init):
        th = np.zeros(self.n_params)
        th[self.idx_R] = np.log(self._components(init.R, "R")).ravel()
        Pc = np.clip(self._components(init.P, "P"), 1e-12, 1 - 1e-12)
        th[self.idx_P] = np.log(Pc / (1 - Pc)).ravel()
        phi = float(np.clip(init.phi, 1e-6, 0.95))
        th[self.idx_phi] = np.log(phi / (1 - phi))
        return th

    def bounds(self):
        n = self.K * self.S * self.B
        return ([(-LOG_R_BOUND, LOG_R_BOUND)] * n + [(-25.0, 25.0)] * n
                + [(-14.0, 3.0)])

    def unpack(self, theta):
        """(R, P) are (K, S, B); with K = 1 callers index [0] to recover the
        original (S, B) tables."""
        shape = (self.K, self.S, self.B)
        R = jnp.exp(theta[self.idx_R]).reshape(shape)
        P = jax.nn.sigmoid(theta[self.idx_P]).reshape(shape)
        phi = jax.nn.sigmoid(theta[self.idx_phi])
        rates = jnp.full((self.S,), self.lambda_fix)[:, None]
        return R, P, rates, jnp.zeros((self.S, 1)), phi


def _eta_init(Pi0, target_probs):
    """Probit least squares per object: with q_j the standard-normal baseline
    cuts, q_j ~ mu_n + sigma_n * ndtri(cumsum(Pi0)_nj)."""
    Pi0 = np.asarray(Pi0, dtype=float)
    B = Pi0.shape[1]
    c = np.clip(np.cumsum(Pi0, axis=1)[:, :B - 1], 1e-4, 1 - 1e-4)
    z = ndtri(c)                                        # (N, B-1)
    q0 = ndtri(np.asarray(target_probs, dtype=float))   # (B-1,)
    zbar = z.mean(axis=1)
    var_z = ((z - zbar[:, None]) ** 2).sum(axis=1)
    cov = ((z - zbar[:, None]) * (q0 - q0.mean())[None, :]).sum(axis=1)
    s = np.clip(np.where(var_z > 1e-10, cov / np.maximum(var_z, 1e-10), 1.0),
                1e-3, 1e3)
    return q0.mean() - s * zbar, s


def fit_effects(X, mask=None, *, lambda_fix=50.0, lambda_init=50.0,
                conditional=False, abundance=True, a_max="auto",
                a_cap=None, cap_headroom=3.0, tau_buckets=6,
                abundance_prior=None, n_abund_nodes=24,
                kappa_init=None, kappa_bounds=(0.5, 4096.0),
                sigma_prior=(0.0, 0.5), sigma_shared=False,
                mu_prior=1.3, mu_center=None, sum_mode="grid",
                shash=None, eps_prior=(0.0, 0.5), shape_init=None,
                delta_bounds=(0.2, 5.0), eps_bounds=(-4.0, 4.0),
                K=1, gamma=None, a_init=None, eta_init=None,
                pin_median=0.0, pin_left=-1.0, pin_weight=100.0,
                pi_floor=1e-12, init=None, tilt=None,
                tilt_design=None, tilt_prior=None, tilt_init=None,
                tilt_bounds=(-6.0, 6.0), tilt_shared=False, tilt_fixed=None,
                max_outer=3, tol=1e-6, global_maxiter=80, eta_maxiter=40,
                finish_maxiter=5000, finish_rounds=6, verbose=True,
                **adapt_kw):
    """Fit the normal-effects model to one cell line's (N, S, B) counts.

    conditional : zero-TRUNCATED likelihood -- all-zero rows are dropped from
        both the likelihood and the mixture that sets the cuts, and each
        kept object is conditioned on Sum X > 0.  The default (False) is
        zero-INFLATION instead: all-zero rows are fitted with a structural-zero
        probability phi and so enter the mixture that sets the cuts, which is
        what aligns the cuts with the read-count structure.
    abundance / abundance_prior : fit a free per-object a_n, or integrate a_n
        out under a Gamma(kappa, mean=1) prior discretized on ``n_abund_nodes``
        Gauss-Laguerre nodes (mutually exclusive).  Under the Gamma prior a_n is
        not a parameter, so ``a_max`` is only the grid ceiling (the far nodes
        carry negligible weight) and none of the per-object machinery below
        applies.
    a_max : hard global ceiling on a_n, or ``"auto"`` to let it escalate --
        which runs ``fit_effects_adaptive`` and returns its result.  Extra
        keyword arguments (``bound_frac``, ``max_rounds``, ``a_max_ceiling``)
        are passed on to it and are an error otherwise.
    a_cap, cap_headroom : the per-object ceiling on a_n, as an (N,) array or as
        the multiple of the object's depth-derived start to derive one from.
        It bounds a_n in the optimizer AND sizes that object's latent-count
        grid, so it must be a real bound; ``fit_effects_adaptive`` doubles
        whatever binds until almost nothing does.  A flat array reproduces the
        single-grid behaviour exactly.
    sum_mode : how the latent count is summed out -- "grid" (the dense
        truncated lattice, the default and what every earlier fit used) or
        "window" (``model.logmarg_window``: a constant number of nodes centred
        on the mode of the summand).  The window rule makes the cost
        independent of depth and of ``a_max``, ignores ``tau_buckets``, and does
        not need the abundance bound to be valid at all -- see
        ``model.LoglikBuilder`` for why the two can also DISAGREE.
    tau_buckets : how many distinct tau grid lengths the objects are bucketed
        onto.  Each is one more XLA kernel to compile; 1 puts every object on
        the longest grid, which is what the model did before.
    sigma_prior : (m, s) Normal prior on log sigma_n, or None.
    sigma_shared : fit ONE effect scale for the whole cell line instead of one
        per object, so every sequence differs only in its effect location mu_n
        (a homoskedastic ordered probit; N-1 parameters instead of 2N).  The
        common value is free, not pinned to 1: the effect axis is defined only
        up to scale, so a pinned value would be a constraint on the cut
        spacing, not a choice of units.  ``sigma_prior`` then applies to that
        single scalar and is effectively inert.
    mu_prior, mu_center : sd and target of the Normal prior on mu_n (None
        disables).  ``mu_center`` may be an (N,) array.
    tilt : fixed (N, S, B) -- or (S, B), or (N, B) -- multiplicative factor on
        the latent rate, mirroring ``fit_free_pi``'s ``tilt``.  UNLIKE free-Pi,
        where Pi is a saturated (N, B-1) free profile and a fixed rate factor
        is provably absorbed exactly when S = 1 (see
        ``scripts/test_batch_saturation.py``), here Pi is constrained to the
        ``normal_bin_probs(mu_n, sigma_n, cuts)`` family -- only 2 (or 3, under
        ``shash``) free dof per object against the tilt's up to B-1 -- so a
        fixed tilt is NOT generally absorbable by refitting (mu_n, sigma_n)
        even at S = 1.  Whether it moves a WELL-DETERMINED (high-depth) object
        at all is a separate, empirical question: for such an object the
        likelihood dominates the mu_prior/sigma_prior regularization either
        way, so the tilt still competes against real information in the
        counts, it merely competes through a narrower family than free-Pi's.
        mu_n / sigma_n themselves are read out as the DE-TILTED location and
        scale -- no readout change is needed, same convention as free-Pi.
    tilt_design, tilt_prior, tilt_init, tilt_bounds, tilt_shared, tilt_fixed :
        the tilt as a FITTED parameter rather than a supplied offset.
        ``tilt_design`` is an (N, B, P) design tensor, centred over bins, and
        the fit carries free amplitudes ``alpha`` of shape (S, P) -- or (1, P)
        with ``tilt_shared`` -- entering the latent rate as

            rate[n,s,b]  *=  exp( sum_p tilt_design[n,b,p] * alpha[s,p] ).

        Fitting it rather than fixing it is what makes the correction a
        statement the data can contradict: a supplied offset can only cost
        likelihood if it is wrong (that is why HepG2 lost 347 nats in the
        reptilt run), while a fitted one comes with a curvature, a standard
        error and a likelihood-ratio test.  ``tilt_prior`` is the sd of a
        Normal(0, .) shrinkage prior on alpha -- a scalar, or a (P,) array, or
        an (S, P) array to shrink each sample toward 0 by a different amount;
        ``None`` leaves the amplitudes unpenalized (pure profile ML).
        ``tilt_fixed`` is an (S, P) offset ADDED to the fitted alpha and held
        fixed, so a hierarchical mean can be carried while the deviation is
        fitted.  Both this and ``tilt`` may be given: the two factors multiply.

        Note what identifies alpha here.  A GC-dependent shift of mu_n alone is
        absorbed EXACTLY by refitting mu (it is a relabelling of the per-object
        effect location), so the part of the design lying in the effect law's
        tangent space is identified only through the mu / sigma priors and
        through the variation of that tangent space across objects.  The
        returned ``extras`` therefore carries the curvature of BOTH the data
        term and the penalized objective; they answer different questions and
        on this panel they differ.
    shash : effect law.  ``None`` is the Normal, and reproduces every earlier
        fit bit for bit.  ``"global"`` fits one sinh-arcsinh (delta, eps) for
        the whole cell line, ``"eps"`` a shared delta with a per-object eps.
        Both start at (1, 0), which IS the Normal, so the fit is nested and the
        likelihood gain is a likelihood-ratio statistic with 2 (or N+1) dof.
    eps_prior : (m, s) Normal prior on the skew, on the same 1/n_obs scale as
        the data term.  It matters only for ``shash="eps"``, where eps_n is the
        third free parameter of a 3-dof object and so is otherwise unshrunk (a
        reparameterized free Pi).  ``None`` disables it.
    shape_init : (delta, eps) to start the shape from; ``None`` starts at the
        Normal.  Pass the previous fit's when warm-starting.
    delta_bounds, eps_bounds : boxes on the shape.  delta is bounded away from 0
        (it multiplies asinh, so delta -> 0 flattens the law to a point mass).
    K, gamma : number of emission components and the (N, K) responsibilities
        that weight them.  K = 1 (the default) is the plain single-emission
        model and ignores ``gamma``.  For K > 1 this is an EM M-step: ``gamma``
        is FIXED here because the component index is shared across cell lines,
        so it can only be formed from all of them at once -- see
        ``mixture.fit_mixture``, which is what should normally be called.
        Passing no ``gamma`` with K > 1 fits K components with equal weights,
        which is a mixture with no assignment information and will not
        separate; it is allowed only as a starting point.
    init : an ``InitResult`` to start from (see ``init_from_fit`` for warm
        starts); ``None`` runs the per-column ZINB initialization.  With K > 1
        its R / P should be (K, S, B) and the components must DIFFER: identical
        components are a saddle point that the M-step cannot leave.
    a_init : (N,) starting abundance.  ``init`` does not carry one -- a_n is
        normally re-derived from the counts -- so this is the way to warm-start
        it.  Rescaled to geometric mean 1 like the derived start.
    eta_init : (mu, sigma) to start the effect law from.  ``init`` carries only
        Pi, from which (mu, sigma) are recovered by a probit least squares whose
        cumulative probabilities are clipped to [1e-4, 1-1e-4] -- lossy, and it
        caps a recoverable mu at about 3.7 sigma.  Pass the previous fit's mu
        and sigma directly when iterating (the EM does).

    Defaults reproduce the shipped configuration.
    """
    if isinstance(a_max, str):
        if a_max != "auto":
            raise ValueError(f"a_max must be a number or 'auto', got {a_max!r}")
        return fit_effects_adaptive(
            X, mask, lambda_fix=lambda_fix, lambda_init=lambda_init,
            conditional=conditional, abundance=abundance, a_cap=a_cap,
            cap_headroom=cap_headroom, tau_buckets=tau_buckets, tilt=tilt,
            abundance_prior=abundance_prior, n_abund_nodes=n_abund_nodes,
            kappa_init=kappa_init, kappa_bounds=kappa_bounds,
            sigma_prior=sigma_prior, sigma_shared=sigma_shared,
            mu_prior=mu_prior, mu_center=mu_center,
            sum_mode=sum_mode, shash=shash, eps_prior=eps_prior, shape_init=shape_init,
            delta_bounds=delta_bounds, eps_bounds=eps_bounds,
            K=K, gamma=gamma,
            a_init=a_init, eta_init=eta_init, pin_median=pin_median,
            pin_left=pin_left, pin_weight=pin_weight, pi_floor=pi_floor,
            tilt_design=tilt_design, tilt_prior=tilt_prior,
            tilt_init=tilt_init, tilt_bounds=tilt_bounds,
            tilt_shared=tilt_shared, tilt_fixed=tilt_fixed,
            init=init, max_outer=max_outer, tol=tol,
            global_maxiter=global_maxiter, eta_maxiter=eta_maxiter,
            finish_maxiter=finish_maxiter, finish_rounds=finish_rounds,
            verbose=verbose, **adapt_kw)
    if adapt_kw:
        raise TypeError(f"unexpected keyword arguments {sorted(adapt_kw)}; "
                        "these only apply when a_max='auto'")
    t0 = time.time()
    X = np.asarray(X)
    N, S, B = X.shape
    if B < 3:
        raise ValueError("B >= 3 is required for per-object sigma to be "
                         "identified")
    target_probs = np.arange(1, B) / B
    mid = (B - 1) // 2

    K = int(K)
    if K < 1:
        raise ValueError("K must be >= 1")
    if gamma is None:
        gamma = np.full((N, K), 1.0 / K)
    gamma = np.asarray(gamma, dtype=float)
    if gamma.shape != (N, K):
        raise ValueError(f"gamma must have shape {(N, K)}, got {gamma.shape}")
    if K > 1 and not np.allclose(gamma.sum(1), 1.0, atol=1e-8):
        # rows that do not sum to 1 silently rescale the data term against the
        # mu / sigma priors, which are not weighted
        raise ValueError("gamma rows must sum to 1 (they are responsibilities)")
    gamma_j = jnp.asarray(gamma)

    if shash not in (None, False, "global", "eps"):
        raise ValueError(f"shash must be None, 'global' or 'eps', got {shash!r}")
    shash = shash or None
    if shash and K > 1:
        raise NotImplementedError(
            "the sinh-arcsinh effect law is not wired through the emission "
            "mixture (the component axis and the shape block would both have "
            "to enter mixture.py's E-step)")
    n_eps = 0 if shash is None else (N if shash == "eps" else 1)

    abund_marg = abundance_prior is not None       # abundance integrated out
    if abund_marg:
        if abundance_prior != "gamma":
            raise ValueError(f"unknown abundance_prior {abundance_prior!r} "
                             "(only 'gamma')")
        if abundance:
            raise ValueError("abundance_prior integrates a_n OUT; it is "
                             "mutually exclusive with abundance=True")
        if n_abund_nodes < 2:
            raise ValueError("n_abund_nodes must be >= 2")

    if init is None:
        if verbose:
            print("[init] fitting per-column ZINBs / quantile transform ...")
        init = initialize(X, mask=mask, lambda_init=lambda_init, verbose=False)

    spec = _Globals(S, B, lambda_fix, K)
    a_max = float(a_max)

    # ``kept`` needs the mask and the all-zero rows, which is all the builder
    # does to X, so derive them here: the caps have to exist BEFORE the builder,
    # since they are what sizes its grids
    obs_mask = ~np.isnan(X) if np.issubdtype(X.dtype, np.floating) \
        else np.ones(X.shape, bool)
    if mask is not None:
        obs_mask = obs_mask & np.asarray(mask, bool)
    observed = obs_mask.any(axis=(1, 2))
    all_zero = np.where(obs_mask, np.nan_to_num(X, nan=0.0), 0.0
                        ).sum(axis=(1, 2)) == 0
    if conditional:
        kept = observed & ~all_zero
        n_zero_dropped = int((observed & all_zero).sum())
        n_obs = int(obs_mask[kept].sum())
    else:
        kept = observed
        n_zero_dropped = 0
        n_obs = int(obs_mask.sum())
    if not kept.any():
        raise ValueError("no objects left to fit")

    # a_n is bounded per object, and that same bound sizes the object's tau
    # grid: the latent rate is a_n * Pi[n,b] * lambda_s with Pi a simplex row,
    # so a_cap * lambda_fix dominates it for every (b, s).
    if abundance:
        if a_cap is None:
            a_cap = abundance_caps(X, obs_mask, kept, a_max,
                                   headroom=cap_headroom)
        a_cap = np.clip(np.asarray(a_cap, dtype=float).reshape(-1), 1e-3, a_max)
        if a_cap.shape != (N,):
            raise ValueError(f"a_cap must be ({N},), got {a_cap.shape}")
        rate_max = lambda_fix * a_cap
    else:
        # under the Gamma prior the abundance nodes are shared by every object,
        # so there is nothing per-object to bucket on
        a_cap = None
        rate_max = lambda_fix * (a_max if abund_marg else 1.0)
    if tilt is not None:
        if abund_marg:
            raise NotImplementedError(
                "tilt is not wired through the Gamma-marginalized abundance "
                "path")
        tilt_np = broadcast_tilt(tilt, N, S, B)
        rate_max = np.asarray(rate_max, float) * tilt_np.max(axis=(1, 2))
        tilt_j = jnp.asarray(tilt_np)

    n_tilt_rows = 0
    if tilt_design is not None:
        if abund_marg:
            raise NotImplementedError(
                "tilt_design is not wired through the Gamma-marginalized "
                "abundance path")
        V = np.asarray(tilt_design, float)
        if V.ndim == 2:                         # (N, B) -> one amplitude
            V = V[:, None, :, None]
        elif V.ndim == 3:                       # (N, B, P), shared over samples
            V = V[:, None, :, :]
        if V.ndim != 4 or V.shape[0] != N or V.shape[2] != B \
                or V.shape[1] not in (1, S):
            raise ValueError(
                f"tilt_design must be (N, B), (N, B, P) or (N, S, B, P) with "
                f"N={N}, S={S}, B={B}; got {np.shape(tilt_design)}")
        V = np.broadcast_to(V, (N, S, B, V.shape[3]))
        # a component common to all bins is an abundance and a_n eats it, so
        # the design is centred over bins -- exactly the gauge batch.py uses.
        # A sample-varying design is what expresses the one tilt contrast the
        # likelihood identifies WITHOUT help from the effect law: give
        # replicate 1 and replicate 2 of a line opposite signs and the shared
        # amplitude is their deviation, in which the line's biology cancels
        # exactly because the profile Pi is shared and the tilt is not.
        V = V - V.mean(axis=2, keepdims=True)
        P_tilt = V.shape[3]
        n_tilt_rows = 1 if tilt_shared else S
        n_tilt = n_tilt_rows * P_tilt
        V_j = jnp.asarray(V)
        lo_t, hi_t = (float(tilt_bounds[0]), float(tilt_bounds[1]))
        if tilt_fixed is None:
            tilt_off_j = jnp.zeros((n_tilt_rows, P_tilt))
        else:
            tilt_off_j = jnp.asarray(np.broadcast_to(
                np.asarray(tilt_fixed, float), (n_tilt_rows, P_tilt)))
        # the tau grid is sized once, so it must hold for every amplitude the
        # optimizer can reach: max_b sum_p V[n,b,p] alpha[s,p] <= sum_p
        # max_b|V[n,b,p]| * max|alpha_p|.  A loose bound costs grid nodes, and
        # nothing at all under sum_mode="window", which never reads rate_max.
        amax_box = max(abs(lo_t), abs(hi_t)) + float(
            np.abs(np.asarray(tilt_off_j)).max(initial=0.0))
        head = np.exp((np.abs(V).max(axis=(1, 2)) * amax_box).sum(axis=1))
        rate_max = np.asarray(rate_max, float) * head
        if sum_mode != "window":
            print(f"[fit] tilt_design: rate_max inflated by up to "
                  f"{head.max():.3g}x for the amplitude box; sum_mode='window'"
                  " avoids this entirely")
    builder = LoglikBuilder(X, mask=mask, rate_max=rate_max,
                            max_buckets=tau_buckets, sum_mode=sum_mode)

    scale = 1.0 / max(n_obs, 1)
    kept_j = jnp.asarray(kept)
    w_np = kept.astype(np.float64)
    w_np /= w_np.sum()
    w_mix = jnp.asarray(w_np)          # uniform over the objects that count
    # eta = [mu (N), log sigma (N or 1), log a (N) | log kappa (1),
    #        log delta (1), eps (1 or N)]
    # the shape block goes LAST so every index below it keeps its meaning
    n_sig = 1 if sigma_shared else N
    i_sig = N
    i_tail = N + n_sig                 # log a block, or the single log kappa
    i_kappa = i_tail
    i_shape = i_tail + (N if abundance else (1 if abund_marg else 0))
    # the tilt block goes after the shape block, for the same reason the shape
    # block goes after the abundance one: every index above keeps its meaning
    i_tilt = i_shape + (0 if shash is None else 1 + n_eps)
    if verbose:
        grid = (f"tau_window={builder.window} (mode-centred)"
                if sum_mode == "window" else
                f"tau_grid={builder.tau_lengths} sizes={builder.bucket_sizes} "
                f"({builder.tau_work:.0f} nodes/object)"
                if len(builder.buckets) > 1
                else f"tau_grid={builder.tau_lengths[0]}")
        print(f"[fit] N={N} S={S} B={B} K={K} kept={int(kept.sum())} "
              f"(missing={int((~observed).sum())}, "
              f"zero-dropped={n_zero_dropped}) conditional={conditional} "
              f"abundance={'gamma' if abund_marg else abundance} "
              f"a_max={a_max:g} {grid} "
              f"sigma_prior={sigma_prior} sigma_shared={sigma_shared} "
              f"mu_prior={mu_prior}"
              + (f" shash={shash} eps_prior={eps_prior}" if shash else ""))

    # --- objective ----------------------------------------------------------
    def _log_sigma(eta):
        """(N,) log sigma, broadcast from the single scalar when shared."""
        ls = eta[i_sig:i_sig + n_sig]
        return jnp.broadcast_to(ls, (N,)) if sigma_shared else ls

    def _shape(eta):
        """(delta, eps) of the sinh-arcsinh law, or None for the Normal.

        A GLOBAL eps stays a scalar rather than being broadcast to (N,): the
        mixture solve and the bin probabilities both then see one number, which
        is cheaper and, more to the point, keeps the two configurations
        distinguishable downstream.
        """
        if shash is None:
            return None
        eps = eta[i_shape + 1:i_shape + 1 + n_eps]
        return jnp.exp(eta[i_shape]), (eps if shash == "eps" else eps[0])

    def _pi_and_cuts(eta):
        mu = eta[:N]
        sig = jnp.exp(_log_sigma(eta))
        shape = _shape(eta)
        cuts = mixture_quantiles(mu, sig, w_mix, target_probs, shape=shape)
        return normal_bin_probs(mu, sig, cuts, floor=pi_floor,
                                shape=shape), cuts

    def _alpha(eta):
        """(n_tilt_rows, P) fitted tilt amplitudes, offset included."""
        return eta[i_tilt:i_tilt + n_tilt].reshape(n_tilt_rows, -1) \
            + tilt_off_j

    def _tilt_free(eta):
        """(N, S, B) multiplicative factor from the fitted amplitudes.

        ``V`` is centred over bins, so the factor has log-mean 0 over b for
        every object and carries no abundance -- the same normalisation
        ``batch.tilt_factor`` applies to a supplied one.
        """
        E = jnp.einsum("nsbp,sp->nsb", V_j, _alpha(eta)) if not tilt_shared \
            else jnp.einsum("nsbp,p->nsb", V_j, _alpha(eta)[0])
        return jnp.exp(E)

    def _rows_and_cuts(eta):
        """Rows fed to the latent Poisson: the profile, or a_n * profile.

        With ``tilt`` given this becomes (N, S, B): the object's profile is
        shared across replicates as always, the tilt is not.
        """
        Pi, cuts = _pi_and_cuts(eta)
        M = (jnp.exp(eta[i_tail:i_tail + N])[:, None] * Pi if abundance
             else Pi)
        if tilt is not None:
            M = M[:, None, :] * tilt_j
        if tilt_design is not None:
            Tf = _tilt_free(eta)
            M = (M[:, None, :] * Tf) if M.ndim == 2 else (M * Tf)
        return M, cuts

    def _abund_rule(eta):
        """(nodes, log weights) of the Gamma(kappa, mean=1) abundance prior."""
        return gamma_mixture_rule(jnp.exp(eta[i_kappa]), 1.0, n_abund_nodes)

    def _data_neg_ll(theta, eta):
        R, P, rates, log_w, phi = spec.unpack(theta)
        M, _ = _rows_and_cuts(eta)
        if K > 1:
            return _data_neg_ll_mix(R, P, rates, log_w, phi, M, eta)
        R, P = R[0], P[0]                        # the single-emission tables
        if abund_marg:
            a_nodes, log_wa = _abund_rule(eta)
            base_lambda = rates[:, 0]                       # (S,)
            obj = builder.object_loglik_abund(
                R, P, M, base_lambda, a_nodes, log_wa,
                0.0 if conditional else phi)
            if conditional:
                lz = jnp.minimum(log_zero_prob_abund(
                    R, P, M, base_lambda, a_nodes, log_wa, builder.mask),
                    -1e-12)
                total = jnp.sum(jnp.where(kept_j, obj - log1mexp(lz), 0.0))
            else:
                total = jnp.sum(obj)
            return -total * scale
        if conditional:
            obj = builder.object_loglik(R, P, M, rates, log_w, 0.0)
            lz = jnp.minimum(log_zero_prob(R, P, M, rates, log_w,
                                           builder.mask), -1e-12)
            total = jnp.sum(jnp.where(kept_j, obj - log1mexp(lz), 0.0))
        else:
            total = jnp.sum(builder.object_loglik(R, P, M, rates, log_w, phi))
        return -total * scale

    def _data_neg_ll_mix(R, P, rates, log_w, phi, M, eta):
        """The EM M-step term sum_n sum_k gamma[n,k] * loglik(n | k).

        A weighted SUM over components, not a logsumexp: the mixture over k is
        resolved in the E-step, across every cell line at once.
        """
        if abund_marg:
            a_nodes, log_wa = _abund_rule(eta)
            base_lambda = rates[:, 0]                       # (S,)
            LL = builder.object_loglik_abund_components(
                R, P, M, base_lambda, a_nodes, log_wa,
                0.0 if conditional else phi)                # (N, K)
            if conditional:
                lz = jnp.minimum(log_zero_prob_abund_components(
                    R, P, M, base_lambda, a_nodes, log_wa, builder.mask),
                    -1e-12)
                LL = LL - log1mexp(lz)
        else:
            LL = builder.object_loglik_components(
                R, P, M, rates, log_w, 0.0 if conditional else phi)
            if conditional:
                lz = jnp.minimum(log_zero_prob_components(
                    R, P, M, rates, log_w, builder.mask), -1e-12)
                LL = LL - log1mexp(lz)
        wll = jnp.sum(gamma_j * LL, axis=1)                 # (N,)
        total = jnp.sum(jnp.where(kept_j, wll, 0.0))
        return -total * scale

    def _gauge_pen(eta):
        """Soft pin of the effect-axis gauge (and of the a-scale ridge that a_n
        shares with R: only c*a*lambda is identified, so pin the geometric mean
        of a to 1)."""
        _, cuts = _pi_and_cuts(eta)
        pen = pin_weight * jnp.square(cuts[mid] - pin_median)
        if pin_left is not None and mid > 0:
            pen = pen + pin_weight * jnp.square(cuts[0] - pin_left)
        if abundance:
            la = eta[i_tail:i_tail + N]
            pen = pen + pin_weight * jnp.square(
                jnp.sum(jnp.where(kept_j, la, 0.0)) / jnp.sum(kept_j))
        return pen

    def _sigma_pen(eta):
        """-log prior on log sigma, on the same 1/n_obs scale as the data term
        (unlike the gauge pins, its strength relative to the likelihood
        matters).  Shared across objects it is one term against n_obs cells,
        i.e. inert -- the scale is then set by the data, as it should be."""
        if sigma_prior is None:
            return 0.0
        m_s, s_s = sigma_prior
        ls = eta[i_sig:i_sig + n_sig]
        quad = (jnp.sum((ls - m_s) ** 2) if sigma_shared else
                jnp.sum(jnp.where(kept_j, (ls - m_s) ** 2, 0.0)))
        return scale * quad / (2.0 * s_s ** 2)

    mu_center_j = (jnp.asarray(mu_center, dtype=jnp.float64)
                   if mu_center is not None else jnp.float64(0.0))

    def _mu_pen(eta):
        """-log prior on the effect means, on the 1/n_obs scale."""
        if mu_prior is None:
            return 0.0
        quad = jnp.sum(jnp.where(kept_j, (eta[:N] - mu_center_j) ** 2, 0.0))
        return scale * quad / (2.0 * mu_prior ** 2)

    def _eps_pen(eta):
        """-log prior on the sinh-arcsinh skew, on the 1/n_obs scale.

        Per object this is the shrinkage that separates the model from a
        saturated free-Pi profile; global it is one term against n_obs cells,
        i.e. inert, and left in only so the two configurations are graded under
        the same objective.
        """
        if shash is None or eps_prior is None:
            return 0.0
        m_e, s_e = eps_prior
        e = eta[i_shape + 1:i_shape + 1 + n_eps]
        quad = (jnp.sum(jnp.where(kept_j, (e - m_e) ** 2, 0.0))
                if shash == "eps" else jnp.sum((e - m_e) ** 2))
        return scale * quad / (2.0 * s_e ** 2)

    tilt_sd_j = None
    if tilt_design is not None and tilt_prior is not None:
        tilt_sd_j = jnp.asarray(np.broadcast_to(
            np.asarray(tilt_prior, float), (n_tilt_rows, P_tilt)))

    def _tilt_pen(eta):
        """-log prior on the tilt amplitudes, on the same 1/n_obs scale as the
        data term.

        A per-SAMPLE parameter against n_obs cells, so any prior wide enough to
        be honest is nearly inert -- which is the point: the shrinkage that
        matters is applied between samples, by the hierarchical model that
        chooses this sd, not inside one line's fit.
        """
        if tilt_sd_j is None:
            return 0.0
        a = eta[i_tilt:i_tilt + n_tilt].reshape(n_tilt_rows, -1)
        return scale * jnp.sum((a / tilt_sd_j) ** 2) / 2.0

    def _neg_obj(theta, eta):
        return (_data_neg_ll(theta, eta) + _gauge_pen(eta)
                + _sigma_pen(eta) + _mu_pen(eta) + _eps_pen(eta)
                + _tilt_pen(eta))

    vg_global = jax.jit(jax.value_and_grad(_neg_obj, argnums=0))
    vg_eta = jax.jit(jax.value_and_grad(
        lambda eta, theta: _neg_obj(theta, eta), argnums=0))
    n_glob = spec.n_params
    vg_joint = jax.jit(jax.value_and_grad(
        lambda x: _neg_obj(x[:n_glob], x[n_glob:])))
    ll_fn = jax.jit(lambda theta, eta: -_data_neg_ll(theta, eta) / scale)
    pll_fn = jax.jit(lambda theta, eta: -_neg_obj(theta, eta) / scale)
    cuts_fn = jax.jit(lambda eta: _pi_and_cuts(eta)[1])

    # --- starting point -----------------------------------------------------
    theta = spec.pack(init)
    g_bounds = spec.bounds()

    if eta_init is None:
        mu0, s0 = _eta_init(init.Pi, target_probs)
    else:
        mu0, s0 = (np.asarray(v, dtype=float).copy() for v in eta_init)
        if mu0.shape != (N,) or s0.shape != (N,):
            raise ValueError(f"eta_init must be two (N,) arrays, got "
                             f"{mu0.shape} and {s0.shape}")
        s0 = np.clip(s0, np.exp(-7.0), np.exp(7.0))
    if sigma_shared:
        # the per-object probit start is noisy on low-count objects; the median
        # is the robust summary of the scale they agree on
        s0 = np.full(1, np.median(s0[kept]) if kept.any() else np.median(s0))
    eta = np.concatenate([mu0, np.log(s0)])
    e_bounds = [(-40.0, 40.0)] * N + [(-7.0, 7.0)] * n_sig
    if abundance:
        if a_init is not None:
            # a supplied by the caller: a warm start, or (under an emission
            # mixture) the K = 1 abundance with the component's amplification
            # divided out, so a and the component do not start double-counting
            # the same depth
            la0 = np.log(np.clip(np.asarray(a_init, dtype=float),
                                 1e-3, a_cap * 0.9))
        else:
            # start a at the object's per-cell total relative to the group mean
            # (the quantity a exists to explain)
            la0 = np.log(np.clip(initial_abundance(X, obs_mask, kept),
                                 1e-3, a_cap * 0.9))
        la0 = np.asarray(la0, dtype=float).copy()
        la0[~kept] = 0.0
        la0 -= la0[kept].mean()                     # geometric mean pinned to 1
        # re-centering can push an object past its own bound, and scipy needs a
        # feasible start
        la0 = np.clip(la0, np.log(1e-4), np.log(a_cap))
        eta = np.concatenate([eta, la0])
        e_bounds += [(np.log(1e-4), float(np.log(c))) for c in a_cap]
    if abund_marg:
        # start the Gamma shape from the object-total CV: a Gamma(kappa,
        # mean=1) has CV = 1/sqrt(kappa)
        if kappa_init is None:
            Xz = np.where(obs_mask, np.nan_to_num(X, nan=0.0), 0.0)
            tot = Xz.sum(axis=(1, 2))[kept]
            mu_t = tot.mean()
            cv = tot.std() / mu_t if mu_t > 0 else 1.0
            k0 = 1.0 / max(cv, 1e-3) ** 2
        else:
            k0 = float(kappa_init)
        k0 = float(np.clip(k0, kappa_bounds[0] * 1.05, kappa_bounds[1] * 0.95))
        eta = np.concatenate([eta, [np.log(k0)]])
        e_bounds += [(np.log(kappa_bounds[0]), np.log(kappa_bounds[1]))]
        if verbose:
            print(f"[fit] gamma abundance: nodes={n_abund_nodes} "
                  f"kappa_init={k0:.3g} (CV~{1/np.sqrt(k0):.2f})")
    if shash is not None:
        # (1, 0) IS the Normal, so an unspecified start puts the fit exactly at
        # the model it generalizes and every nat gained from here is real
        d0, e0 = (1.0, np.zeros(n_eps)) if shape_init is None else \
            (float(shape_init[0]),
             np.broadcast_to(np.asarray(shape_init[1], float), (n_eps,)))
        d0 = float(np.clip(d0, delta_bounds[0] * 1.001, delta_bounds[1] * 0.999))
        e0 = np.clip(np.asarray(e0, float), eps_bounds[0] * 0.999,
                     eps_bounds[1] * 0.999)
        eta = np.concatenate([eta, [np.log(d0)], e0])
        e_bounds += [(np.log(delta_bounds[0]), np.log(delta_bounds[1]))]
        e_bounds += [tuple(float(v) for v in eps_bounds)] * n_eps
    if tilt_design is not None:
        # alpha = 0 IS the no-tilt model, so an unspecified start puts the fit
        # exactly at the model it nests and every nat from here is a
        # likelihood-ratio statistic with n_tilt dof
        a0 = (np.zeros((n_tilt_rows, P_tilt)) if tilt_init is None else
              np.broadcast_to(np.asarray(tilt_init, float),
                              (n_tilt_rows, P_tilt)).astype(float))
        # clipped to the box exactly, not to a shrunken one: tilt_bounds=(c, c)
        # is how an amplitude is PINNED, and a start nudged off c would fit a
        # different model
        a0 = np.clip(a0, lo_t, hi_t).reshape(-1)
        eta = np.concatenate([eta, a0])
        e_bounds += [(lo_t, hi_t)] * n_tilt
    if len(eta) != i_tilt + (0 if tilt_design is None else n_tilt):
        raise AssertionError("eta layout and its index map disagree")

    # start in the pinned gauge (the affine map rescales every sigma by the
    # same k, so a shared scale stays shared)
    mu0, s0, _ = gauge_normalize(eta[:N], np.exp(eta[i_sig:i_sig + n_sig]),
                                 np.asarray(cuts_fn(jnp.asarray(eta))),
                                 pin_median=pin_median, pin_left=pin_left)
    eta[:N] = mu0
    eta[i_sig:i_sig + n_sig] = np.log(s0)

    history = []

    def current_ll(theta, eta):
        return float(ll_fn(jnp.asarray(theta), jnp.asarray(eta)))

    def current_pll(theta, eta):
        return float(pll_fn(jnp.asarray(theta), jnp.asarray(eta)))

    def record(stage, ll, res=None):
        history.append(dict(stage=stage, loglik=ll, time=time.time() - t0))
        if verbose:
            line = f"[{time.time()-t0:7.1f}s] {stage:<24} loglik = {ll:.4f}"
            if res is not None:
                line += f"   [{_res_line(res, scale)}]"
            print(line, flush=True)

    ll, pll = current_ll(theta, eta), current_pll(theta, eta)
    record("init", ll)

    # --- block-coordinate warmup --------------------------------------------
    cyc = 0
    cycle_tol = tol * 10
    converged = False
    for outer in range(1, max_outer + 1):
        pll_prev_outer = pll

        opt = _GLOBAL_CYCLE[cyc % len(_GLOBAL_CYCLE)]
        res = minimize(
            lambda th: tuple(map(np.asarray, vg_global(
                jnp.asarray(th), jnp.asarray(eta)))),
            theta, jac=True, method=opt, bounds=g_bounds,
            options=_solver_options(opt, global_maxiter))
        if -res.fun / scale >= pll - 1e-12:
            theta = res.x
        pll_new = current_pll(theta, eta)
        if pll_new - pll < cycle_tol * n_obs:
            cyc += 1
        pll = pll_new
        ll = current_ll(theta, eta)
        record(f"outer{outer}:global({opt})", ll, res)

        res = minimize(
            lambda e: tuple(map(np.asarray, vg_eta(
                jnp.asarray(e), jnp.asarray(theta)))),
            eta, jac=True, method="L-BFGS-B", bounds=e_bounds,
            options=_solver_options("L-BFGS-B", eta_maxiter))
        if -res.fun / scale >= pll - 1e-12:
            eta = res.x
        ll, pll = current_ll(theta, eta), current_pll(theta, eta)
        record(f"outer{outer}:eta(L-BFGS-B)", ll, res)

        if pll - pll_prev_outer < tol * n_obs:
            converged = True
            break

    # --- joint finishing rounds: TNC first, L-BFGS-B on failure --------------
    converged = False
    finish_opt = "TNC"
    for rnd in range(1, finish_rounds + 1):
        pll_prev = pll
        x0 = np.concatenate([theta, eta])
        res = minimize(
            lambda x: tuple(map(np.asarray, vg_joint(jnp.asarray(x)))),
            x0, jac=True, method=finish_opt, bounds=g_bounds + e_bounds,
            options=_solver_options(finish_opt, finish_maxiter))
        if -res.fun / scale >= pll - 1e-12:
            theta, eta = res.x[:n_glob], res.x[n_glob:]
            ll, pll = current_ll(theta, eta), current_pll(theta, eta)
        record(f"finish{rnd}:joint({finish_opt})", ll, res)
        gain = pll - pll_prev
        tnc_healthy = res.success or getattr(res, "status", -1) == 3
        if finish_opt == "TNC" and (not tnc_healthy or gain < tol * n_obs):
            if verbose and rnd < finish_rounds:
                print(f"          TNC stalled (gain {gain:.4f}); "
                      f"switching finish optimizer to L-BFGS-B")
            finish_opt = "L-BFGS-B"
            if gain >= tol * n_obs:
                continue
            pll_prev2 = pll
            x0 = np.concatenate([theta, eta])
            res = minimize(
                lambda x: tuple(map(np.asarray, vg_joint(jnp.asarray(x)))),
                x0, jac=True, method="L-BFGS-B", bounds=g_bounds + e_bounds,
                options=_solver_options("L-BFGS-B", finish_maxiter))
            if -res.fun / scale >= pll - 1e-12:
                theta, eta = res.x[:n_glob], res.x[n_glob:]
                ll, pll = current_ll(theta, eta), current_pll(theta, eta)
            record(f"finish{rnd}b:joint(L-BFGS-B)", ll, res)
            if pll - pll_prev2 < tol * n_obs:
                converged = True
                break
        elif gain < tol * n_obs:
            converged = True
            break

    # --- exact gauge normalization + invariance check -----------------------
    # the affine gauge acts on (mu, sigma, cuts) only; a_n and kappa are
    # invariant under it
    tail = np.asarray(eta[i_tail:i_shape], dtype=float)  # log a | log kappa
    shape_blk = np.asarray(eta[i_shape:], dtype=float)   # log delta, eps
    mu_hat = np.asarray(eta[:N], dtype=float)
    sig_hat = np.exp(np.asarray(eta[i_sig:i_sig + n_sig], dtype=float))
    mu_hat, sig_hat, cuts_hat = gauge_normalize(
        mu_hat, sig_hat, np.asarray(cuts_fn(jnp.asarray(eta))),
        pin_median=pin_median, pin_left=pin_left)
    eta_norm = np.concatenate([mu_hat, np.log(sig_hat), tail, shape_blk])
    ll_norm = current_ll(theta, eta_norm)
    if abs(ll_norm - ll) > max(1e-6 * abs(ll), 1e-3):
        # should never happen (exact invariance) -- keep the better point
        if verbose:
            print(f"[warn] gauge normalization moved loglik {ll:.4f} -> "
                  f"{ll_norm:.4f}; keeping the raw optimum")
        mu_hat = np.asarray(eta[:N], dtype=float)
        sig_hat = np.exp(np.asarray(eta[i_sig:i_sig + n_sig], dtype=float))
        cuts_hat = np.asarray(cuts_fn(jnp.asarray(eta)))
        eta_final = np.asarray(eta, dtype=float)
    else:
        ll = ll_norm
        eta_final = eta_norm
    record("gauge-normalized", ll)

    # the reported sigma is always (N,), so a shared-scale fit stays a drop-in
    # everywhere downstream; its rows are simply all equal
    sig_hat = np.broadcast_to(sig_hat, (N,)).copy()
    a_hat = np.exp(tail[:N]) if abundance else np.ones(N)
    kappa_hat = float(np.exp(eta_final[i_kappa])) if abund_marg else None
    delta_hat, eps_hat, shape_hat = None, None, None
    if shash is not None:
        delta_hat = float(np.exp(eta_final[i_shape]))
        eps_blk = np.asarray(eta_final[i_shape + 1:i_shape + 1 + n_eps], float)
        eps_hat = eps_blk if shash == "eps" else float(eps_blk[0])
        shape_hat = (jnp.asarray(delta_hat), jnp.asarray(eps_hat))
    tilt_extras = {}
    if tilt_design is not None:
        a_hat_t = np.asarray(eta_final[i_tilt:i_tilt + n_tilt], float
                             ).reshape(n_tilt_rows, -1)
        # curvature of the data term and of the penalized objective in alpha,
        # every OTHER parameter held fixed.  These bound the profile curvature
        # from above -- letting (mu, sigma, a) re-absorb the tilt can only
        # flatten it -- and their ratio to the profile version measures how much
        # of the tilt the effect law simply relabels.
        def _ll_alpha(a, which):
            e = jnp.asarray(eta_final).at[i_tilt:i_tilt + n_tilt].set(a)
            f = _data_neg_ll if which == "data" else _neg_obj
            return f(jnp.asarray(theta), e) / scale
        a_flat = jnp.asarray(a_hat_t.reshape(-1))
        tilt_extras = dict(
            alpha=a_hat_t,
            alpha_grad_data=np.asarray(
                jax.grad(_ll_alpha)(a_flat, "data"), float),
            alpha_hess_data=np.asarray(
                jax.hessian(_ll_alpha)(a_flat, "data"), float),
            alpha_hess_pen=np.asarray(
                jax.hessian(_ll_alpha)(a_flat, "pen"), float),
            tilt_offset=np.asarray(tilt_off_j, float),
            tilt_shared=bool(tilt_shared))
    R, P, rates, log_w, phi = spec.unpack(jnp.asarray(theta))
    # K == 1 reports the plain (S, B) tables, so a single-emission fit stays
    # interchangeable with everything written before the mixture existed
    R = np.asarray(R)[0] if K == 1 else np.asarray(R)
    P = np.asarray(P)[0] if K == 1 else np.asarray(P)
    Pi = np.asarray(normal_bin_probs(jnp.asarray(mu_hat), jnp.asarray(sig_hat),
                                     jnp.asarray(cuts_hat), floor=pi_floor,
                                     shape=shape_hat))

    return FitResult(
        R=R, P=P, mu=mu_hat, sigma=sig_hat,
        cuts=cuts_hat, Pi=Pi, rates=np.asarray(rates), log_w=np.asarray(log_w),
        phi=0.0 if conditional else float(phi), loglik=ll, a=a_hat,
        a_cap=a_cap, delta=delta_hat, eps=eps_hat,
        n_observed=n_obs, converged=converged, observed=kept,
        n_zero_dropped=n_zero_dropped, history=history,
        extras=(dict(tilt_extras) if not abund_marg else
                dict(kappa=kappa_hat, abund_cv=kappa_hat ** -0.5,
                     **tilt_extras)),
        config=dict(lambda_fix=lambda_fix, conditional=conditional,
                    abundance=abundance, a_max=a_max,
                    abundance_prior=abundance_prior,
                    n_abund_nodes=n_abund_nodes, kappa=kappa_hat,
                    sigma_prior=sigma_prior, sigma_shared=sigma_shared,
                    mu_prior=mu_prior, shash=shash, eps_prior=eps_prior,
                    K=K, tau_buckets=tau_buckets, sum_mode=sum_mode,
                    pin_median=pin_median, pin_left=pin_left,
                    pin_weight=pin_weight, pi_floor=pi_floor,
                    tilt_prior=tilt_prior, tilt_shared=tilt_shared,
                    n_tilt=(0 if tilt_design is None else n_tilt)))


def fit_effects_adaptive(X, mask=None, *, a_max=20.0, a_max_ceiling=4096.0,
                         bound_frac=0.001, max_rounds=6, a_cap=None,
                         cap_headroom=3.0, abundance=True, verbose=True,
                         **kw):
    """``fit_effects`` with the abundance bound raised until it stops binding.

    A fixed bound on a_n is easy to underestimate: on some cell lines a
    percent or more of the sequences sit exactly at it, which is the model
    refusing to say how deep they are.  Raising
    the bound for everybody is correct and slow, because it is also what sizes
    the latent-count grid.  So the bound is per object and only the objects that
    reach it are raised:

      * start every object at ``cap_headroom`` times its depth-derived
        abundance, clipped to ``a_max``;
      * fit;
      * double the ceiling of every object sitting at its own, up to
        ``a_max_ceiling``, and refit warm from the previous fit;
      * stop when fewer than ``bound_frac`` of the fitted objects are at their
        bound, or when nothing can be raised any further.

    ``bound_frac`` is worth being strict about, because a bound that still
    binds anywhere moves the whole cell line -- the cuts are shared.  On a
    line where the bound bites, stopping at 0.5% leaves the ceiling at 20 and
    the activity 0.026 (median) from the a_max=100 fit, barely better than a
    fixed a_max=15's 0.032; 0.1% escalates to 80 and 0.05% to 160, where the
    largest fitted a_n is 98 -- the bound has stopped binding by itself -- and
    the activity is 0.004 from the reference, i.e. at the run-to-run noise
    floor of two fits of the same configuration.  Four warm rounds cost about
    what one fit on the full a_max=100 grid does, and never allocate it.

    ``a_max`` here is where the GLOBAL ceiling starts rather than where it
    stays: it rises with the caps, up to ``a_max_ceiling``.  Pass a number to
    ``fit_effects`` instead to pin it.

    An inactive box constraint does not move an optimum, so what this converges
    to is the fit with no abundance ceiling at all -- reached without ever
    sizing the tau grid for the deepest object in the line.

    ``max_rounds`` is the cost bound, and it matters: an UNDER-CONVERGED fit
    leaves objects at their bound for reasons that have nothing to do with the
    bound, and the loop will then keep doubling.  Six rounds take the ceiling
    from 20 to 640, which is past anything these libraries want; if a run keeps
    escalating to the end of that, suspect the optimizer schedule rather than
    the data.

    Returns the last ``FitResult``; ``res.a_cap`` is the converged per-object
    bound, which is the right ``a_cap`` to warm-start a later fit of the same
    data with.
    """
    from .initialize import init_from_fit

    if not abundance:
        # no free a_n to bound (either it is fixed at 1 or integrated out)
        return fit_effects(X, mask=mask, a_max=float(a_max),
                           abundance=abundance, verbose=verbose, **kw)
    X = np.asarray(X)
    a_ceiling = float(a_max_ceiling)
    ceiling = float(min(max(float(a_max), 1e-3), a_ceiling))

    res = None
    for rnd in range(1, int(max_rounds) + 1):
        if res is not None:
            kw = dict(kw, init=init_from_fit(res), a_init=res.a,
                      eta_init=(res.mu, res.sigma))
            if res.delta is not None:
                # the shape is part of the effect law, so a round that restarted
                # it at the Normal would throw away the previous round's fit
                kw["shape_init"] = (res.delta, res.eps)
        res = fit_effects(X, mask=mask, a_max=ceiling, a_cap=a_cap,
                          cap_headroom=cap_headroom, abundance=True,
                          verbose=verbose, **kw)
        cap, a, kept = res.a_cap, res.a, np.asarray(res.observed, bool)
        n_kept = max(int(kept.sum()), 1)
        at_bound = kept & (a >= (1.0 - 1e-3) * cap)
        frac = at_bound.sum() / n_kept
        if verbose:
            print(f"[a_max] round {rnd}: {at_bound.sum()}/{n_kept} "
                  f"({frac:.3%}) objects at their abundance bound; "
                  f"ceiling={ceiling:g} max a={a[kept].max():.3g}", flush=True)
        if frac <= bound_frac:
            break
        new_cap = np.where(at_bound, np.minimum(2.0 * cap, a_ceiling), cap)
        # the global ceiling only rises far enough to hold the raised caps: it
        # is still the number the readout clips its profiled abundance at, so
        # doubling it for everyone because a handful of objects moved would
        # change the readout for no reason
        new_ceiling = float(min(max(ceiling, new_cap.max()), a_ceiling))
        if np.array_equal(new_cap, cap) and new_ceiling == ceiling:
            if verbose:
                print(f"[a_max] ceiling {a_ceiling:g} reached with {frac:.3%} "
                      "still at the bound; stopping", flush=True)
            break
        a_cap, ceiling = new_cap, new_ceiling
    else:
        if verbose:
            print(f"[a_max] {max_rounds} rounds used without dropping below "
                  f"{bound_frac:.3%}"
                  + ("; the last fit did not converge either, so suspect the "
                     "optimizer schedule rather than the ceiling"
                     if not res.converged else ""), flush=True)
    return res
