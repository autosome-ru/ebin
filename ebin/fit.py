"""Maximum-marginal-likelihood fit of the normal-effects compound model.

Parameters split into a global block theta = [log R, logit P, logit phi] (S*B
channels) and a per-object block eta = [mu, log sigma, (log a | log kappa)].
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
* ``mu_prior``  Normal prior on the effect means.  sigma_prior bounds the
  effect SPREAD but not its LOCATION, so a single-bin low-count object still
  runs mu_n large and its Pi walks to a simplex vertex (activity exactly 1 or
  B).  The gauge pins Q50 = 0 and Q25 = -1, i.e. population sd ~ 1.5 and
  between-object sd ~ 1.1, so the gauge-consistent tau is ~1.3, not 3.
* the soft gauge pin, which costs nothing at the optimum (the likelihood
  gradient along the gauge is exactly zero) and is applied exactly on exit.

Abundance.  Without one, every object has the same expected latent total
(lambda_s), so any real depth variation has nowhere to go but Pi, which tilts
toward the bins with the largest channel scale c = R(1-P)/P -- Pi stops being a
profile and becomes an abundance proxy.  Either fit a free a_n
(``abundance=True``) or integrate it out under a Gamma prior
(``abundance_prior='gamma'``), which costs one global parameter instead of N and
propagates the abundance uncertainty into the effect.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.special import ndtri

from .model import LoglikBuilder, gamma_mixture_rule
from .initialize import initialize
from .effects import mixture_quantiles, normal_bin_probs, gauge_normalize
from .truncation import log_zero_prob, log_zero_prob_abund, log1mexp

jax.config.update("jax_enable_x64", True)

_GLOBAL_CYCLE = ["SLSQP", "TNC", "L-BFGS-B"]


@dataclass
class FitResult:
    R: np.ndarray                 # (S, B) emission size
    P: np.ndarray                 # (S, B) emission probability
    mu: np.ndarray                # (N,) effect means      (pinned gauge)
    sigma: np.ndarray             # (N,) effect sds        (pinned gauge)
    cuts: np.ndarray              # (B-1,) shared bin cut points
    Pi: np.ndarray                # (N, B) implied bin profile
    rates: np.ndarray             # (S, K) latent Poisson rates
    log_w: np.ndarray             # (S, K) their log weights
    phi: float                    # structural-zero probability (0 if truncated)
    loglik: float
    n_observed: int
    converged: bool
    observed: np.ndarray          # (N,) objects entering the fit
    a: np.ndarray = None          # (N,) fitted abundance (ones if not fitted)
    n_zero_dropped: int = 0       # all-zero rows removed by the truncation
    extras: dict = None           # kappa / abundance CV under a Gamma prior
    history: list = field(repr=False, default=None)
    config: dict = field(repr=False, default=None)


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
    """The channel block: theta = [log R (S*B), logit P (S*B), logit phi]."""

    def __init__(self, S, B, lambda_fix):
        self.S, self.B, self.lambda_fix = S, B, float(lambda_fix)
        n = S * B
        self.idx_R = slice(0, n)
        self.idx_P = slice(n, 2 * n)
        self.idx_phi = 2 * n
        self.n_params = 2 * n + 1

    def pack(self, init):
        th = np.zeros(self.n_params)
        th[self.idx_R] = np.log(np.asarray(init.R)).ravel()
        Pc = np.clip(np.asarray(init.P), 1e-12, 1 - 1e-12)
        th[self.idx_P] = np.log(Pc / (1 - Pc)).ravel()
        phi = float(np.clip(init.phi, 1e-6, 0.95))
        th[self.idx_phi] = np.log(phi / (1 - phi))
        return th

    def bounds(self):
        n = self.S * self.B
        return [(-20.0, 20.0)] * n + [(-25.0, 25.0)] * n + [(-14.0, 3.0)]

    def unpack(self, theta):
        R = jnp.exp(theta[self.idx_R]).reshape(self.S, self.B)
        P = jax.nn.sigmoid(theta[self.idx_P]).reshape(self.S, self.B)
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
                conditional=False, abundance=True, a_max=15.0,
                abundance_prior=None, n_abund_nodes=24,
                kappa_init=None, kappa_bounds=(0.5, 4096.0),
                sigma_prior=(0.0, 0.5), mu_prior=1.3, mu_center=None,
                pin_median=0.0, pin_left=-1.0, pin_weight=100.0,
                pi_floor=1e-12, init=None,
                max_outer=3, tol=1e-6, global_maxiter=80, eta_maxiter=40,
                finish_maxiter=5000, finish_rounds=6, verbose=True):
    """Fit the normal-effects model to one cell line's (N, S, B) counts.

    conditional : zero-TRUNCATED likelihood -- all-zero rows are dropped from
        both the likelihood and the mixture that sets the cuts, and each
        kept object is conditioned on Sum X > 0.  The default (False) is
        zero-INFLATION instead: all-zero rows are fitted with a structural-zero
        probability phi and so enter the mixture that sets the cuts, which is
        what aligns the cuts with the read-count structure.
    abundance / abundance_prior : fit a free per-object a_n, or integrate a_n
        out under a Gamma(kappa, mean=1) prior discretized on ``n_abund_nodes``
        Gauss-Laguerre nodes (mutually exclusive).  ``a_max`` bounds a_n and
        sizes the latent tau grid; under the Gamma prior it is only the grid
        ceiling, since the far nodes carry negligible weight.
    sigma_prior : (m, s) Normal prior on log sigma_n, or None.
    mu_prior, mu_center : sd and target of the Normal prior on mu_n (None
        disables).  ``mu_center`` may be an (N,) array.
    init : an ``InitResult`` to start from (see ``init_from_fit`` for warm
        starts); ``None`` runs the per-column ZINB initialization.

    Defaults reproduce the shipped configuration.
    """
    t0 = time.time()
    X = np.asarray(X)
    N, S, B = X.shape
    if B < 3:
        raise ValueError("B >= 3 is required for per-object sigma to be "
                         "identified")
    target_probs = np.arange(1, B) / B
    mid = (B - 1) // 2

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

    spec = _Globals(S, B, lambda_fix)
    # with an abundance the latent rate reaches a_max * lambda, so the
    # truncated tau grid must be sized for it
    rate_max = lambda_fix * (a_max if (abundance or abund_marg) else 1.0)
    builder = LoglikBuilder(X, mask=mask, rate_max=rate_max)

    observed = np.asarray(builder.mask).any(axis=(1, 2))
    if conditional:
        all_zero = np.asarray(builder.all_zero)
        kept = observed & ~all_zero
        n_zero_dropped = int((observed & all_zero).sum())
        n_obs = int(np.asarray(builder.mask)[kept].sum())
    else:
        kept = observed
        n_zero_dropped = 0
        n_obs = builder.n_observed
    if not kept.any():
        raise ValueError("no objects left to fit")

    scale = 1.0 / max(n_obs, 1)
    kept_j = jnp.asarray(kept)
    w_np = kept.astype(np.float64)
    w_np /= w_np.sum()
    w_mix = jnp.asarray(w_np)          # uniform over the objects that count
    i_kappa = 2 * N                    # eta = [mu, log sigma, log kappa]
    if verbose:
        print(f"[fit] N={N} S={S} B={B} kept={int(kept.sum())} "
              f"(missing={int((~observed).sum())}, "
              f"zero-dropped={n_zero_dropped}) conditional={conditional} "
              f"abundance={'gamma' if abund_marg else abundance} "
              f"tau_grid={builder.tau_full.shape[0]} "
              f"sigma_prior={sigma_prior} mu_prior={mu_prior}")

    # --- objective ----------------------------------------------------------
    def _pi_and_cuts(eta):
        mu = eta[:N]
        sig = jnp.exp(eta[N:2 * N])
        cuts = mixture_quantiles(mu, sig, w_mix, target_probs)
        return normal_bin_probs(mu, sig, cuts, floor=pi_floor), cuts

    def _rows_and_cuts(eta):
        """Rows fed to the latent Poisson: the profile, or a_n * profile."""
        Pi, cuts = _pi_and_cuts(eta)
        if not abundance:
            return Pi, cuts
        return jnp.exp(eta[2 * N:3 * N])[:, None] * Pi, cuts

    def _abund_rule(eta):
        """(nodes, log weights) of the Gamma(kappa, mean=1) abundance prior."""
        return gamma_mixture_rule(jnp.exp(eta[i_kappa]), 1.0, n_abund_nodes)

    def _data_neg_ll(theta, eta):
        R, P, rates, log_w, phi = spec.unpack(theta)
        M, _ = _rows_and_cuts(eta)
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

    def _gauge_pen(eta):
        """Soft pin of the effect-axis gauge (and of the a-scale ridge that a_n
        shares with R: only c*a*lambda is identified, so pin the geometric mean
        of a to 1)."""
        _, cuts = _pi_and_cuts(eta)
        pen = pin_weight * jnp.square(cuts[mid] - pin_median)
        if pin_left is not None and mid > 0:
            pen = pen + pin_weight * jnp.square(cuts[0] - pin_left)
        if abundance:
            la = eta[2 * N:3 * N]
            pen = pen + pin_weight * jnp.square(
                jnp.sum(jnp.where(kept_j, la, 0.0)) / jnp.sum(kept_j))
        return pen

    def _sigma_pen(eta):
        """-log prior on log sigma, on the same 1/n_obs scale as the data term
        (unlike the gauge pins, its strength relative to the likelihood
        matters)."""
        if sigma_prior is None:
            return 0.0
        m_s, s_s = sigma_prior
        quad = jnp.sum(jnp.where(kept_j, (eta[N:2 * N] - m_s) ** 2, 0.0))
        return scale * quad / (2.0 * s_s ** 2)

    mu_center_j = (jnp.asarray(mu_center, dtype=jnp.float64)
                   if mu_center is not None else jnp.float64(0.0))

    def _mu_pen(eta):
        """-log prior on the effect means, on the 1/n_obs scale."""
        if mu_prior is None:
            return 0.0
        quad = jnp.sum(jnp.where(kept_j, (eta[:N] - mu_center_j) ** 2, 0.0))
        return scale * quad / (2.0 * mu_prior ** 2)

    def _neg_obj(theta, eta):
        return (_data_neg_ll(theta, eta) + _gauge_pen(eta)
                + _sigma_pen(eta) + _mu_pen(eta))

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

    mu0, s0 = _eta_init(init.Pi, target_probs)
    eta = np.concatenate([mu0, np.log(s0)])
    e_bounds = [(-40.0, 40.0)] * N + [(-7.0, 7.0)] * N
    if abundance:
        # start a at the object's per-cell total relative to the group mean
        # (the quantity a exists to explain), geometric mean pinned to 1
        Xz = np.where(np.asarray(builder.mask), np.nan_to_num(X, nan=0.0), 0.0)
        cells = np.maximum(np.asarray(builder.mask).sum(axis=(1, 2)), 1)
        rate_n = Xz.sum(axis=(1, 2)) / cells        # depth-robust per-cell mean
        ref = np.exp(np.mean(np.log(rate_n[kept] + 1e-6)))
        la0 = np.log(np.clip(rate_n / max(ref, 1e-12), 1e-3, a_max * 0.9))
        la0[~kept] = 0.0
        la0 -= la0[kept].mean()
        eta = np.concatenate([eta, la0])
        e_bounds += [(np.log(1e-4), np.log(a_max))] * N
    if abund_marg:
        # start the Gamma shape from the object-total CV: a Gamma(kappa,
        # mean=1) has CV = 1/sqrt(kappa)
        if kappa_init is None:
            Xz = np.where(np.asarray(builder.mask),
                          np.nan_to_num(X, nan=0.0), 0.0)
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

    # start in the pinned gauge
    mu0, s0, _ = gauge_normalize(eta[:N], np.exp(eta[N:2 * N]),
                                 np.asarray(cuts_fn(jnp.asarray(eta))),
                                 pin_median=pin_median, pin_left=pin_left)
    eta[:N] = mu0
    eta[N:2 * N] = np.log(s0)

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
    tail = np.asarray(eta[2 * N:], dtype=float)          # log a | log kappa
    mu_hat = np.asarray(eta[:N], dtype=float)
    sig_hat = np.exp(np.asarray(eta[N:2 * N], dtype=float))
    mu_hat, sig_hat, cuts_hat = gauge_normalize(
        mu_hat, sig_hat, np.asarray(cuts_fn(jnp.asarray(eta))),
        pin_median=pin_median, pin_left=pin_left)
    eta_norm = np.concatenate([mu_hat, np.log(sig_hat), tail])
    ll_norm = current_ll(theta, eta_norm)
    if abs(ll_norm - ll) > max(1e-6 * abs(ll), 1e-3):
        # should never happen (exact invariance) -- keep the better point
        if verbose:
            print(f"[warn] gauge normalization moved loglik {ll:.4f} -> "
                  f"{ll_norm:.4f}; keeping the raw optimum")
        mu_hat = np.asarray(eta[:N], dtype=float)
        sig_hat = np.exp(np.asarray(eta[N:2 * N], dtype=float))
        cuts_hat = np.asarray(cuts_fn(jnp.asarray(eta)))
        eta_final = np.asarray(eta, dtype=float)
    else:
        ll = ll_norm
        eta_final = eta_norm
    record("gauge-normalized", ll)

    a_hat = np.exp(tail[:N]) if abundance else np.ones(N)
    kappa_hat = float(np.exp(eta_final[i_kappa])) if abund_marg else None
    R, P, rates, log_w, phi = spec.unpack(jnp.asarray(theta))
    Pi = np.asarray(normal_bin_probs(jnp.asarray(mu_hat), jnp.asarray(sig_hat),
                                     jnp.asarray(cuts_hat), floor=pi_floor))

    return FitResult(
        R=np.asarray(R), P=np.asarray(P), mu=mu_hat, sigma=sig_hat,
        cuts=cuts_hat, Pi=Pi, rates=np.asarray(rates), log_w=np.asarray(log_w),
        phi=0.0 if conditional else float(phi), loglik=ll, a=a_hat,
        n_observed=n_obs, converged=converged, observed=kept,
        n_zero_dropped=n_zero_dropped, history=history,
        extras=({} if not abund_marg else
                dict(kappa=kappa_hat, abund_cv=kappa_hat ** -0.5)),
        config=dict(lambda_fix=lambda_fix, conditional=conditional,
                    abundance=abundance, a_max=a_max,
                    abundance_prior=abundance_prior,
                    n_abund_nodes=n_abund_nodes, kappa=kappa_hat,
                    sigma_prior=sigma_prior, mu_prior=mu_prior,
                    pin_median=pin_median, pin_left=pin_left,
                    pin_weight=pin_weight, pi_floor=pi_floor))
