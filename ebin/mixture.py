"""Emission mixture with a component index shared by every cell line.

The reads-per-cell scale c = R(1-P)/P is an amplification efficiency, and in
some libraries (5'UTR, say) it is visibly bimodal across sequences: the pooled
count histogram has a well-amplified mode and a much weaker one.  A sequence's
amplification is a property of the SEQUENCE, so it is the same in every cell
line and every replicate, while how many cells carry it and where they sort is
not.  So the emission becomes a K-component NB mixture over the same latent T,

    X[n,s,b] | T[n,s,b], k_n = k ~ NB(R_k[s,b] * T[n,s,b], P_k[s,b]),

with ONE component index k_n per sequence for the whole dataset.

Why the component is not just the abundance
-------------------------------------------
Scaling a_n by rho scales the marginal mean and variance together, leaving the
Fano factor at 1/P + c.  Scaling the emission scale c by rho gives the same mean
but Fano 1/P + rho*c.  A sequence with few cells and one with poor amplification
therefore look different -- same mean, different spread -- and that is the ONLY
thing that separates them, because a free per-object a_n can match any mean.

That is enough.  Measured on simulated data at the real depth (c ~ 20 reads per
cell, 1/P ~ 3): moving a true 12.5x amplification split out of the emission and
into the abundance, at a matched marginal mean, costs 16 nats PER SEQUENCE.  The
split is not a flat ridge -- but it is a stiff one, and a coordinate optimizer
walks it very slowly, so ``ridge_step`` searches that direction directly after
each M-step.  Read ``component_summary`` anyway: a ratio near 1 means the
mixture found nothing.

Why the cell lines stop being independent, and why EM fixes it
--------------------------------------------------------------
With k_n shared, the observed-data log-likelihood is

    log L = sum_n log sum_k pi_k prod_g p_g(X_{g,n} | k, Theta_g, eta_{g,n}),

and the sum over k sits OUTSIDE the product over cell lines g: no line can be
fitted on its own any more.  EM restores the separation exactly.  With
responsibilities gamma[n,k] from the current parameters,

    Q = sum_n sum_k gamma[n,k] log pi_k
      + sum_g [ sum_n sum_k gamma[n,k] log p_g(X_{g,n} | k, Theta_g, eta_{g,n}) ]

-- the second term is a plain sum over g, so the M-step IS one independent fit
per cell line, exactly the old problem with a responsibility-weighted data term
(``fit_effects(..., K=K, gamma=gamma)``).  Only the E-step touches all lines at
once, and it is a single forward pass per line:

    log gamma[n,k] = log pi_k + sum_g log p_g(X_{g,n} | k)   (then normalized).

So the answer to "does the Q function become independent with respect to cell
lines" is yes, apart from the K-1 free numbers in pi.  The coupling survives
only as the (N, K) responsibility matrix passed between the steps.

Each M-step raises Q, so the observed-data log-likelihood is non-decreasing
(generalized EM); ``state.history`` records it per round.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

from .data import load_groups, prep
from .model import LoglikBuilder, gamma_mixture_rule, as_components
from .truncation import (log_zero_prob_components,
                         log_zero_prob_abund_components, log1mexp)
from .fit import (fit_effects, light, save_fits, load_fits, StoredFit,
                  LOG_R_BOUND)
from .initialize import init_from_fit

_LOG_EPS = 1e-12


def _buckets(cfg):
    return int(cfg.get("tau_buckets", 6) or 1)


def _sum_mode(cfg):
    return cfg.get("sum_mode", "grid")


def _rate_ceiling(cfg, Pi, a, abund_marg, a_headroom=1.0):
    """Per-object bound on the latent rate a_n * Pi[n,b] * lambda_s.

    Everything here is evaluated at an ALREADY FITTED (Pi, a), so the bound is
    exact arithmetic rather than a guess: max_b Pi[n,b] times the abundance the
    caller may still move a_n to.  Under the Gamma prior a_n is integrated out
    over shared nodes, so there is nothing per-object to bound and the flat
    ``a_max`` grid comes back.
    """
    lam = float(cfg["lambda_fix"])
    if abund_marg or not cfg.get("abundance"):
        return lam * (float(cfg["a_max"]) if abund_marg else 1.0)
    a_bound = np.minimum(np.asarray(a, dtype=float) * a_headroom,
                         float(cfg["a_max"]))
    return lam * a_bound * np.asarray(Pi, dtype=float).max(axis=1)


@dataclass
class MixtureState:
    """Everything the EM produces beyond the per-line fits."""

    K: int
    pi: np.ndarray                # (K,) component prior
    gamma: np.ndarray             # (N, K) responsibilities, all lines pooled
    comp_loglik: dict             # cell line -> (N, K) log p_g(X_n | k)
    loglik: float                 # observed-data log-likelihood, all lines
    objective: float = None       # the above minus the M-step's priors -- the
                                  # quantity EM actually makes monotone
    order: list = None            # cell lines, in fit order
    index: object = None          # the sequence index
    history: list = field(repr=False, default=None)

    def log_gamma_loo(self, g):
        """(N, K) unnormalized log responsibilities EXCLUDING cell line ``g``.

        What the readout of line g must condition on: using the full gamma
        would let g's own counts inform the component and then be used again as
        the likelihood, counting the same reads twice.  The joint posterior of
        (k, theta_g) is proportional to pi_k * prod_{g' != g} p_{g'}(X|k) *
        p_g(X|theta_g, k) p(theta_g), and the first two factors are this.
        """
        acc = np.log(np.maximum(self.pi, _LOG_EPS))[None, :] \
            + sum(self.comp_loglik[o] for o in self.order)
        return acc - self.comp_loglik[g]

    def assignment(self):
        """(N,) hard component of each sequence (0-based) and its probability."""
        k = self.gamma.argmax(1)
        return k, self.gamma[np.arange(len(k)), k]


def component_loglik(fit, X, mask):
    """(N, K) log p(X_n | emission component k) for one cell line.

    Exactly the per-object term the fit's own objective uses -- the same zero
    handling, the same abundance treatment -- so summing it over cell lines
    gives the E-step accumulator.  Objects the fit did not keep contribute 0.

    ``fit`` is a ``FitResult``, a ``StoredFit`` or the plain dict ``light``
    returns.
    """
    fit = StoredFit(fit) if isinstance(fit, dict) else fit
    cfg = fit.config
    K = int(cfg.get("K", 1))
    conditional = bool(cfg.get("conditional", False))
    abund_marg = cfg.get("abundance_prior") is not None

    R, P = as_components(fit.R), as_components(fit.P)
    rates, log_w = jnp.asarray(fit.rates), jnp.asarray(fit.log_w)
    phi = 0.0 if conditional else float(fit.phi)
    Pi = jnp.asarray(fit.Pi)
    M = Pi if abund_marg else jnp.asarray(np.asarray(fit.a)[:, None]) * Pi
    builder = LoglikBuilder(X, mask=mask, max_buckets=_buckets(cfg),
                            sum_mode=_sum_mode(cfg),
                            rate_max=_rate_ceiling(cfg, fit.Pi, getattr(
                                fit, "a", None), abund_marg))

    LL = _component_loglik_terms(builder, cfg, R, P, M, rates, log_w, phi)
    LL = np.asarray(LL, dtype=float).reshape(-1, K)
    return np.where(np.asarray(fit.observed)[:, None], LL, 0.0)


def _component_loglik_terms(builder, cfg, R, P, M, rates, log_w, phi):
    """(N, K) per-component log-likelihood given already-assembled rows M."""
    if cfg.get("abundance_prior") is not None:
        a_nodes, log_wa = gamma_mixture_rule(
            jnp.float64(cfg["kappa"]), 1.0, cfg["n_abund_nodes"])
        base_lambda = rates[:, 0]
        LL = builder.object_loglik_abund_components(
            R, P, M, base_lambda, a_nodes, log_wa, phi)
        if cfg.get("conditional"):
            lz = jnp.minimum(log_zero_prob_abund_components(
                R, P, M, base_lambda, a_nodes, log_wa, builder.mask), -1e-12)
            LL = LL - log1mexp(lz)
        return LL
    LL = builder.object_loglik_components(R, P, M, rates, log_w, phi)
    if cfg.get("conditional"):
        lz = jnp.minimum(log_zero_prob_components(
            R, P, M, rates, log_w, builder.mask), -1e-12)
        LL = LL - log1mexp(lz)
    return LL


def ridge_step(fit, X, mask, gamma, max_shift=6.0):
    """Search the a_n <-> R_k ridge directly, which the M-step only crawls.

    The abundance and the emission scale enter the mean only through their
    product a_n * c_k, so trading depth between them is a very stiff direction
    -- and it is exactly the direction that decides how much of the
    amplification the component carries rather than the abundance.  A
    coordinate optimizer moves along it in tiny steps: with the true labels
    handed to it, the M-step alone recovered a 12.5x simulated split as 5.1x
    and was still not converged.  It is only K numbers, so search it as a
    block:

        R_k -> R_k * exp(d_k),   a_n -> a_n * exp(-sum_k gamma[n,k] d_k)

    which leaves each sequence's expected reads roughly where they were and
    changes only the attribution.  mu, sigma, the cuts, Pi and phi do not
    depend on R or a, so the result is a consistent fit.

    d is PROJECTED onto pi_bar . d = 0, pi_bar being the mean responsibility
    over the fitted objects, which leaves mean(log a) where it was.  Without
    that projection the search does not find the split at all: it finds the
    shared a <-> R scale, and since the M-steps here are warm-started and
    iteration-capped they leave that scale slightly off the pin, so an
    unprojected search happily burns hundreds of nats of data likelihood
    re-centering a.  Re-centering a is the M-step's job and is not free anyway
    -- rescaling a against R preserves the marginal mean but not the Fano
    factor, which is the whole signal here.  The split direction, which is what
    is left after projecting, is the one the M-step cannot walk.

    The objective still carries the fit's a-scale pin, so that when the a_max
    clip binds and the projection stops being exact the step is measured on the
    same objective ``fit_effects`` descends and cannot make it worse.

    Returns (updated fit dict, gain in the data term, in nats).  A gain of 0
    means the M-step was already on the ridge optimum.
    """
    from scipy.optimize import minimize

    fit = dict(fit if isinstance(fit, dict) else fit.__dict__)
    cfg = fit["config"]
    K = int(cfg.get("K", 1))
    if K < 2:
        return fit, 0.0
    free_a = bool(cfg.get("abundance"))
    abund_marg = cfg.get("abundance_prior") is not None
    a_max = cfg["a_max"] if (free_a or abund_marg) else 1.0
    # the search multiplies a by exp(-(gamma . d)) with |d| <= max_shift, so
    # the abundance can grow by at most e^max_shift before the a_max clip
    builder = LoglikBuilder(
        X, mask=mask, max_buckets=_buckets(cfg), sum_mode=_sum_mode(cfg),
        rate_max=_rate_ceiling(cfg, fit["Pi"], fit.get("a"), abund_marg,
                               a_headroom=float(np.exp(max_shift))))

    R0 = as_components(np.asarray(fit["R"], dtype=float))
    P = as_components(np.asarray(fit["P"], dtype=float))
    Pi = jnp.asarray(np.asarray(fit["Pi"], dtype=float))
    a0 = jnp.asarray(np.asarray(fit["a"], dtype=float))
    rates, log_w = jnp.asarray(fit["rates"]), jnp.asarray(fit["log_w"])
    phi = 0.0 if cfg.get("conditional") else float(fit["phi"])
    gamma = np.asarray(gamma, dtype=float)
    keptn = np.asarray(fit["observed"], dtype=bool)
    gam = jnp.asarray(gamma)
    kept = jnp.asarray(keptn)

    n_kept = max(int(keptn.sum()), 1)
    scale = 1.0 / max(int(np.asarray(mask).sum()), 1)   # as in fit_effects
    pin_weight = float(cfg.get("pin_weight", 100.0))
    # mean_n (gamma_n . d) = pi_bar . d, so pi_bar . d = 0 leaves mean(log a)
    # untouched and confines the search to the split
    pi_bar = jnp.asarray(gamma[keptn].mean(0) if keptn.any()
                         else np.full(K, 1.0 / K))

    def shifted(delta):
        d = delta - (jnp.dot(pi_bar, delta) if free_a else 0.0)
        R = R0 * jnp.exp(d)[:, None, None]
        if not free_a:
            return R, a0, Pi
        a = jnp.clip(a0 * jnp.exp(-(gam @ d)), 1e-4, a_max)
        return R, a, a[:, None] * Pi

    def data_nll(delta):
        R, _, M = shifted(delta)
        LL = _component_loglik_terms(builder, cfg, R, P, M, rates, log_w, phi)
        return -jnp.sum(jnp.where(kept, jnp.sum(gam * LL, axis=1), 0.0))

    def penalized(delta):
        """The M-step's objective restricted to (R, a): the a-scale pin is the
        only part of the gauge penalty that moves with them."""
        obj = scale * data_nll(delta)
        if not free_a:
            return obj
        _, a, _ = shifted(delta)
        la = jnp.log(a)
        return obj + pin_weight * jnp.square(
            jnp.sum(jnp.where(kept, la, 0.0)) / n_kept)

    nll_fn = jax.jit(data_nll)
    obj_and_grad = jax.jit(jax.value_and_grad(penalized))
    obj_fn = jax.jit(penalized)

    zero = jnp.zeros(K)
    f0 = float(obj_and_grad(zero)[0])
    nll0 = float(nll_fn(zero))
    res = minimize(lambda d: tuple(map(np.asarray,
                                       obj_and_grad(jnp.asarray(d)))),
                   np.zeros(K), jac=True, method="L-BFGS-B",
                   bounds=[(-max_shift, max_shift)] * K,
                   options=dict(maxiter=200, ftol=1e-14, gtol=1e-11))
    # Accept only a step that improves BOTH the penalized objective and the
    # data term Q -- raising Q at fixed gamma is the whole point, and it is
    # what the EM guarantee needs.  The two can disagree once the a_max clip
    # binds, because the clip stops the projection from holding mean(log a)
    # exactly fixed and the pin starts trading against the data again.  Back
    # off along the same direction rather than giving up on it.
    q_before = float((gamma * component_loglik(fit, X, mask)).sum())
    for t in (1.0, 0.5, 0.25, 0.125, 0.0625):
        d = jnp.asarray(t * res.x)
        if not (float(obj_fn(d)) < f0 and float(nll_fn(d)) < nll0):
            continue
        R, a, _ = shifted(d)
        # stay inside the bound the next M-step will pack theta into -- and
        # exactly that bound, not a "safe" margin inside it: the M-step really
        # does drive R to ~e^19.7, and clipping tighter here rewrote the fit
        # after its objective had been evaluated, which cost 1725 nats before
        # the check below caught it.
        cand = dict(fit)
        cand["R"] = np.clip(np.asarray(R), np.exp(-LOG_R_BOUND),
                            np.exp(LOG_R_BOUND))
        cand["a"] = np.asarray(a)
        # Re-measure with the function the E-STEP uses, on the fit as actually
        # stored.  The internal objective is not automatically the same number:
        # the stored fit goes through clips and an (R, a) round trip, and this
        # step has already shipped two versions whose internal accounting said
        # it had gained while Q had in fact dropped.  Whatever the cause, a
        # step that does not raise the E-step's own Q must not be taken.
        q_after = float((gamma * component_loglik(cand, X, mask)).sum())
        if q_after > q_before:
            cand["loglik"] = float(fit["loglik"]) + (q_after - q_before)
            return cand, q_after - q_before
    return fit, 0.0


def penalty_nats(fit):
    """The regularization the M-step carries for one cell line, in nats.

    What EM makes monotone is the PENALIZED objective -- the M-step descends
    data + priors, and is free to trade one for the other -- so the bare
    observed-data log-likelihood can dip in a round that is behaving perfectly.
    The driver tracks ``loglik - sum_g penalty_nats(fit_g)`` for its
    convergence test and reports both.

    The gauge pins are not included: ``fit_effects`` returns a gauge-normalized
    fit, so the cut terms are exactly zero and the a-scale term is ~1e-6.
    """
    cfg = fit["config"] if isinstance(fit, dict) else fit.config
    get = (lambda k: fit[k]) if isinstance(fit, dict) else (lambda k: getattr(fit, k))
    kept = np.asarray(get("observed"), dtype=bool)
    pen = 0.0
    sp = cfg.get("sigma_prior")
    if sp is not None:
        m_s, s_s = sp
        ls = np.log(np.asarray(get("sigma"), dtype=float)[kept])
        pen += float(np.sum((ls - m_s) ** 2) / (2.0 * s_s ** 2))
    tau = cfg.get("mu_prior")
    if tau is not None:
        mu = np.asarray(get("mu"), dtype=float)[kept]
        pen += float(np.sum(mu ** 2) / (2.0 * tau ** 2))
    return pen


def _softmax_rows(logits):
    m = logits.max(1, keepdims=True)
    w = np.exp(logits - m)
    return w / w.sum(1, keepdims=True)


def relative_depth(gdata, order):
    """(N, G) per-sequence read total in each cell line, divided by that line's
    mean.  The raw signal the amplification classes show up in: a sequence that
    amplifies badly is weak in EVERY line, which is exactly the cross-line
    structure the shared component is meant to capture."""
    D = np.empty((len(gdata[order[0]].index), len(order)))
    for j, g in enumerate(order):
        X, mask, _ = prep(gdata[g])
        tot = np.where(mask, np.nan_to_num(X, nan=0.0), 0.0).sum((1, 2))
        D[:, j] = tot / max(tot.mean(), 1e-12)
    return D


def _gmm1d(x, K, n_iter=200, tol=1e-8):
    """K-component 1-D Gaussian mixture by EM.  Means start at equally spaced
    quantiles, which for a two-mode histogram lands one component per mode.
    Components come back ordered by mean."""
    x = np.asarray(x, float)
    m = np.quantile(x, (np.arange(K) + 0.5) / K)
    s = np.full(K, max(x.std(), 1e-3))
    p = np.full(K, 1.0 / K)

    def responsibilities(m, s, p):
        logp = (np.log(np.maximum(p, _LOG_EPS))[None, :]
                - 0.5 * ((x[:, None] - m[None, :]) / s[None, :]) ** 2
                - np.log(s)[None, :])
        mx = logp.max(1)
        ll = float((mx + np.log(np.exp(logp - mx[:, None]).sum(1))).sum())
        return _softmax_rows(logp), ll

    ll_prev = -np.inf
    for _ in range(n_iter):
        r, ll = responsibilities(m, s, p)
        nk = np.maximum(r.sum(0), 1e-8)
        p = nk / nk.sum()
        m = (r * x[:, None]).sum(0) / nk
        s = np.maximum(np.sqrt((r * (x[:, None] - m[None, :]) ** 2).sum(0)
                               / nk), 1e-3)
        if ll - ll_prev < tol * max(abs(ll), 1.0):
            break
        ll_prev = ll
    r, _ = responsibilities(m, s, p)
    o = np.argsort(m)
    return r[:, o], p[o], m[o], s[o]


def init_components(gdata, order, K, min_depth=1e-6):
    """Starting responsibilities and per-component amplification factors.

    A K-component 1-D mixture on the cross-line mean of log relative depth.
    Averaging over cell lines is what isolates the sequence-intrinsic part:
    per line the depth is a mix of amplification and how many cells the
    sequence got, but only the amplification repeats across lines.

    Returns (gamma0, pi0, rho) with rho the per-component reads-per-cell factor,
    geometric mean 1 under pi0, so R_k = rho_k * R keeps the overall scale.
    """
    D = relative_depth(gdata, order)
    seen = D > 0
    n_seen = seen.sum(1)
    L = np.where(seen, np.log(np.maximum(D, min_depth)), 0.0).sum(1) \
        / np.maximum(n_seen, 1)
    usable = n_seen > 0
    if K == 1:
        return np.ones((len(L), 1)), np.ones(1), np.ones(1)
    g_u, pi0, m, _ = _gmm1d(L[usable], K)
    gamma0 = np.tile(pi0, (len(L), 1))
    gamma0[usable] = g_u
    rho = np.exp(m - float(pi0 @ m))
    return gamma0, pi0, rho


def _pass0(gdata, order, fit_kw, verbose):
    """Single-emission fits, used only as the starting point of the EM."""
    fits = {}
    for i, g in enumerate(order):
        t0 = time.time()
        X, mask, _ = prep(gdata[g])
        res = fit_effects(X, mask=mask, verbose=False, **fit_kw)
        fits[g] = light(res)
        if verbose:
            print(f"  [pass0 {i+1}/{len(order)}] {g}: loglik={res.loglik:.1f} "
                  f"conv={res.converged} ({time.time()-t0:.0f}s)", flush=True)
    return fits


def _split_fit(fit, rho, gamma0):
    """Turn a single-emission fit into a K-component starting point.

    R_k = rho_k * R with P shared is the amplification reading of the mixture:
    component k multiplies reads per cell by rho_k.  The components are then
    freed in the M-step.  The abundance is divided by the sequence's expected
    rho so a_n and the component do not both start carrying the same depth.
    """
    init = init_from_fit(fit)
    R = np.asarray(init.R, dtype=float)
    if R.ndim != 2:
        raise ValueError("pass 0 must be a single-emission fit; its R has "
                         f"shape {R.shape}. Re-fit, or start the EM from the "
                         "K-component fits with init_gamma instead of "
                         "splitting them again.")
    init.R = rho[:, None, None] * R[None]
    init.P = np.broadcast_to(np.asarray(init.P, dtype=float),
                             init.R.shape).copy()
    a = np.asarray(fit["a"] if isinstance(fit, dict) else fit.a, dtype=float)
    return init, a / np.exp(gamma0 @ np.log(rho))


def fit_mixture(data, groups=None, *, K=2, em_rounds=4, em_tol=1.0,
                warm_from=None, init_gamma=None, fits_out=None, verbose=True,
                ridge=True, mstep_kw=None, **fit_kw):
    """Fit every cell line under a shared K-component emission mixture.

    data, groups : as ``activity_table``.
    K            : number of emission components (1 is the plain model).
    em_rounds    : maximum EM iterations after the single-emission pass 0.
    em_tol       : stop when a round gains less than this many nats.  A round
                   that LOSES ground stops the loop too and is discarded: the
                   M-steps are warm-started and iteration-capped, so they do
                   not exactly maximize Q and the EM guarantee can fail.  What
                   comes back is always the best round, not the last.
    warm_from    : a fits pickle to use as pass 0 instead of fitting it.
    init_gamma   : (N, K) starting responsibilities, overriding the
                   depth-based initialization.
    ridge        : follow each M-step with a direct search along the
                   a_n <-> R_k ridge (see ``ridge_step``).  Cheap, and without
                   it the amplification split comes out several times too
                   small.
    mstep_kw     : optimizer overrides for the M-step refits, which are warm
                   starts and so want a shorter schedule than pass 0
                   (default: one warmup round and two finishing rounds).
    fit_kw       : passed to ``fit_effects``, for pass 0 and every M-step.

    Returns (fits, state): the per-line light fits, and the ``MixtureState``
    carrying pi, the pooled responsibilities and the per-line component
    log-likelihoods the readout needs.
    """
    t0 = time.time()
    gdata = data if isinstance(data, dict) else load_groups(data, groups=groups,
                                                            verbose=verbose)
    order = [g for g in (groups or gdata) if g in gdata]
    if not order:
        raise ValueError(f"no cell lines to fit (have {sorted(gdata)})")
    K = int(K)
    if em_rounds < 1:
        raise ValueError("em_rounds must be >= 1")
    clash = {"K", "gamma", "a_init", "init", "a_cap"} & set(fit_kw)
    if clash:
        raise ValueError(f"{sorted(clash)} are set by the EM, not by the "
                         "caller (use init_gamma / warm_from instead)")
    index = gdata[order[0]].index
    N = len(index)

    if verbose:
        print(f"[mixture] {len(order)} cell lines x {N} sequences, K={K}, "
              f"{em_rounds} EM rounds")

    fits0 = load_fits(warm_from) if warm_from else None
    if fits0 is None:
        if verbose:
            print("[mixture] pass 0: single-emission fits")
        fits0 = _pass0(gdata, order, fit_kw, verbose)
    fits0 = {g: (f.__dict__ if hasattr(f, "__dict__") else f)
             for g, f in fits0.items()}
    missing = [g for g in order if g not in fits0]
    if missing:
        raise ValueError(f"pass-0 fits are missing cell line(s) {missing}")

    gamma, pi, rho = init_components(gdata, order, K)
    if init_gamma is not None:
        gamma = np.asarray(init_gamma, dtype=float)
        if gamma.shape != (N, K):
            raise ValueError(f"init_gamma must be {(N, K)}, got {gamma.shape}")
        pi = gamma.mean(0)
    if verbose and K > 1:
        print(f"[mixture] init: pi={np.round(pi, 4)} "
              f"amplification factors rho={np.round(rho, 4)}")

    inits, a_inits, etas, caps = {}, {}, {}, {}
    for g in order:
        inits[g], a_inits[g] = _split_fit(fits0[g], rho, gamma)
        etas[g] = (fits0[g]["mu"], fits0[g]["sigma"])
        # pass 0 already escalated the abundance bound; starting the M-steps
        # from the converged caps keeps the escalation loop a no-op instead of
        # re-running it on every round of every line
        caps[g] = fits0[g].get("a_cap")

    # round 1 restarts every line from a freshly split pass-0 fit, so it gets
    # the full schedule; later rounds are warm starts and need far less
    later = dict(max_outer=1, finish_rounds=2)
    later.update(mstep_kw or {})
    fits, comp_ll, history = {}, {}, []
    prev_ll = -np.inf
    best = None                    # (loglik, fits, comp_ll, gamma, pi)
    for rnd in range(1, em_rounds + 1):
        # --- M-step ---------------------------------------------------------
        # Q splits into one term per cell line plus the pi term, so this is
        # just the old per-line problem with a gamma-weighted data term.
        pi = np.maximum(gamma.mean(0), _LOG_EPS)
        pi = pi / pi.sum()
        kw = dict(fit_kw) if rnd == 1 else dict(fit_kw, **later)
        for i, g in enumerate(order):
            tg = time.time()
            X, mask, _ = prep(gdata[g])
            res = fit_effects(X, mask=mask, K=K, gamma=gamma, init=inits[g],
                              a_init=a_inits[g], eta_init=etas[g],
                              a_cap=caps[g], verbose=False, **kw)
            fit, gain = light(res), 0.0
            if ridge:
                fit, gain = ridge_step(fit, X, mask, gamma)
            fits[g] = fit
            inits[g] = init_from_fit(fit)
            a_inits[g] = np.asarray(fit["a"], dtype=float)
            etas[g] = (fit["mu"], fit["sigma"])
            caps[g] = fit.get("a_cap")
            comp_ll[g] = component_loglik(fit, X, mask)
            if verbose:
                print(f"  [em{rnd} {i+1}/{len(order)}] {g}: Q={res.loglik:.1f}"
                      f"{f' (+{gain:.1f} ridge)' if gain else ''} "
                      f"conv={res.converged} ({time.time()-tg:.0f}s)",
                      flush=True)
        if fits_out:
            save_fits(fits_out, fits)

        # --- E-step: the only place every cell line meets --------------------
        acc = sum(comp_ll[g] for g in order)                       # (N, K)
        logpost = np.log(pi)[None, :] + acc
        mx = logpost.max(1)
        ll = float((mx + np.log(np.exp(logpost - mx[:, None]).sum(1))).sum())
        gamma = _softmax_rows(logpost)
        # the monotone quantity is the PENALIZED objective; the bare data
        # log-likelihood may dip while the M-step buys prior fit with it
        obj = ll - sum(penalty_nats(fits[g]) for g in order)
        dropped = obj < prev_ll - em_tol
        history.append(dict(round=rnd, loglik=ll, objective=obj, pi=pi.copy(),
                            decreased=bool(dropped), time=time.time() - t0))
        if verbose:
            print(f"[mixture] round {rnd}: objective={obj:.1f} "
                  f"(loglik={ll:.1f}) pi={np.round(pi, 4)} "
                  f"hard split={np.bincount(gamma.argmax(1), minlength=K)}"
                  f"{'  (DECREASED)' if dropped else ''}", flush=True)
        if best is None or obj > best[0]:
            best = (obj, {g: dict(fits[g]) for g in order}, dict(comp_ll),
                    gamma.copy(), pi.copy(), ll)
        gained = obj - prev_ll
        prev_ll = obj
        # EM raises the observed-data likelihood only when the M-step really
        # maximizes Q; these M-steps are warm-started and iteration-capped, so
        # a round CAN lose ground.  Keep the best one rather than the last.
        if dropped:
            if verbose:
                bestr = int(np.argmax([h["objective"] for h in history])) + 1
                print(f"[mixture] round {rnd} lost {best[0]-obj:.1f} nats "
                      f"against round {bestr}; keeping the best and stopping",
                      flush=True)
            break
        if rnd > 1 and gained < em_tol:
            if verbose:
                print(f"[mixture] converged (gain {gained:.2f} nats)",
                      flush=True)
            break

    obj, fits, comp_ll, gamma, pi, ll = best
    if fits_out:
        save_fits(fits_out, fits)
    state = MixtureState(K=K, pi=pi, gamma=gamma, comp_loglik=comp_ll,
                         loglik=ll, objective=obj, order=list(order),
                         index=index, history=history)
    return fits, state


def component_summary(fits, state=None, out=None):
    """Per cell line, the reads per cell `c = R(1-P)/P` of each component and
    the ratio between the extremes.  **The diagnostic to look at first.**

    The component is identified against the free per-object a_n only through the
    Fano factor (see the module docstring), so a fit can come back with the
    components merged -- `ratio` near 1 -- which means the mixture found
    nothing and is the single-emission model with extra parameters.  A ratio
    well above 1 and CONSISTENT across cell lines is the signature of a real
    amplification split, since amplification is a property of the sequence and
    should repeat in every library.
    """
    import pandas as pd

    order = list(state.order if state is not None else fits)
    rows = {}
    for g in order:
        f = fits[g]
        f = f if isinstance(f, dict) else f.__dict__
        R, P = np.asarray(f["R"]), np.asarray(f["P"])
        if R.ndim == 2:
            R, P = R[None], P[None]
        c = (R * (1 - P) / P).mean(axis=(1, 2))
        rows[g] = {**{f"c{j + 1}": c[j] for j in range(len(c))},
                   "ratio": float(c.max() / max(c.min(), 1e-300))}
    tab = pd.DataFrame(rows).T
    tab.index.name = "group"
    if out:
        tab.to_csv(out)
    return tab


def save_state(path, state):
    """Pickle a ``MixtureState``.  Needed to redo a readout later: the
    component weights come from every cell line at once and cannot be
    recovered from the per-line fits."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def components_table(state, out=None):
    """One row per sequence: the responsibilities, the hard component and its
    probability.  Global, not per cell line -- that is the point of the shared
    index."""
    import pandas as pd

    k, conf = state.assignment()
    cols = {f"gamma{j + 1}": state.gamma[:, j] for j in range(state.K)}
    tab = pd.DataFrame({**cols, "component": k + 1, "confidence": conf},
                       index=state.index)
    tab.index.name = "seq"
    if out:
        tab.to_csv(out)
    return tab
