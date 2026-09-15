"""Free-Pi: the saturated profile, with no effect law.

Every object carries its own B-simplex row directly (B-1 free parameters via
softmax logits) instead of borrowing a shape from a shared latent law:

    T[n,s,b] ~ Poisson(a_n * Pi[n,b] * lambda_s),   Pi[n,:] = softmax(z[n,:]).

That is the same compound NB-Poisson likelihood ``fit_effects`` maximizes --
only the per-object block changes -- so the whole of ``model.LoglikBuilder``
applies verbatim, including the tau bucketing, the escalating abundance bound
and ``sum_mode="window"``.

There is NO effect law here, hence no cuts and no gauge: nothing ties one
object's profile to another's, and nothing ties one cell line's to another's.
That is the entire trade.  It fits each line better (3 free dof per object at
B = 4 instead of 2) and transfers better, and it loses cross-line comparability,
which is what the specificity/profile metric measures.

``pi_prior`` is the only thing standing between this and a degenerate fit.  The
free-Pi MLE is unbounded toward the simplex vertices: a low-count object whose
reads land in one bin wants Pi = (0,...,1,...,0), i.e. an activity of exactly 1
or exactly B.  A symmetric Dirichlet(alpha) adds (alpha-1) sum_b log Pi[n,b],
which diverges at the vertices, so the MAP profile is pulled a finite distance
inside; the pull is strongest for the objects with least data and vanishes for
well-determined ones.  It is the free-Pi analogue of ``sigma_prior``.

The readout is the plug-in ``E[bin] = sum_b b Pi[n,b]``.  There is no posterior
mean available: ``activity.posterior_activity`` integrates over (mu, sigma) of
an effect law that does not exist here, so ``activity_sd`` is undefined.

``abundance_mode="replicate"`` fits a separate a_{n,s} per (object, replicate)
instead of one a_n shared across them, i.e.

    T[n,s,b] ~ Poisson(a_{n,s} * Pi[n,b] * lambda_s).

The profile Pi[n,:] is still shared -- what stops being shared is the object's
DEPTH in each replicate.  With one a_n the model must predict the same total for
both replicates of an object; when they differ (uneven transfection, a sequence
that happened to be recovered in one prep and not the other) the emission has to
absorb the whole discrepancy, and the shallow replicate's few reads then argue
about the profile with a weight the data do not support.  Freeing a_{n,s} lets
each replicate match its own total, after which only the SHAPE across bins
carries information -- and the deep replicate dominates it, as it should.
lambda_s and R[s,b] cannot do this: they are global per-replicate levels, not
per (object, replicate).

The gauge changes with it.  Only c * a * lambda is identified with
c = R(1-P)/P, and with replicate-specific a each column a[:, s] rides its OWN
R[s, :] ridge, so the fit pins S geometric means to 1 instead of one.

``mom_target`` couples this line to the others.  The lines are fitted
independently -- there is nothing in the likelihood that makes one line's
activity scale comparable to another's, which is precisely what free-Pi gives
up.  A joint fit under the equality constraints

    mean_n E[bin]_n = m   and   var_n E[bin]_n = v      (the same m, v every line)

restores that comparability without touching the per-object freedom.  Since the
constraint couples the lines ONLY through the two scalars (m, v), the joint
problem decomposes: given (m, v) every line is a separate equality-constrained
fit.  This function solves that per-line subproblem by the method of
multipliers, i.e. it adds

    lam (M1 - m) + nu (M2 - v) + (rho/2) [ (M1 - m)^2 + (M2 - v)^2 ]

to the negative log-likelihood, with (lam, nu) = ``mom_mult`` the Lagrange
multipliers of the two constraints and rho = ``mom_rho`` the augmentation that
makes the subproblem well-conditioned before the multipliers have converged.
The outer loop -- update (lam, nu) by the violation, update the consensus
(m, v) so that the multipliers sum to zero across lines -- lives in
``scripts/moment_fit.py``; nothing here knows about the other cell lines.

The constrained activity is NOT an affine rescaling of the unconstrained one.
The multiplier tilts every object's profile by the gradient of its own
E[bin], which is largest where the profile is least determined, so the fit
pays for the moment move wherever it is cheapest in likelihood, not uniformly.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from .model import LoglikBuilder
from .batch import broadcast_tilt
from .initialize import initialize
from .fit import (FitResult, _Globals, _solver_options, _GLOBAL_CYCLE,
                  initial_abundance, abundance_caps, _eta_init)
from .truncation import log_zero_prob, log1mexp

jax.config.update("jax_enable_x64", True)


def _replicate_a0(X, mask, kept_rep):
    """(N, S) starting size factors for a replicate-specific abundance.

    ``initial_abundance`` per replicate: the object's per-cell read total in
    that replicate, relative to the geometric mean over the objects that count
    IN THAT REPLICATE.  Each column is referenced to its own geometric mean
    because each is pinned separately (they ride separate R[s, :] ridges), and
    because a globally shallow replicate would otherwise start every one of its
    a_{n,s} below 1 -- which is the shared model's failure mode, restated.

    With S = 1 this is exactly ``initial_abundance``.
    """
    Xz = np.where(np.asarray(mask, bool), np.nan_to_num(np.asarray(X), nan=0.0),
                  0.0)
    cells = np.maximum(np.asarray(mask, bool).sum(axis=2), 1)   # (N, S)
    rate = Xz.sum(axis=2) / cells
    a0 = np.ones_like(rate)
    for s in range(rate.shape[1]):
        sel = np.asarray(kept_rep, bool)[:, s]
        if not sel.any():
            continue
        ref = np.exp(np.mean(np.log(rate[sel, s] + 1e-6)))
        a0[:, s] = rate[:, s] / max(ref, 1e-12)
    return a0


def fit_free_pi(X, mask=None, *, lambda_fix=50.0, lambda_init=50.0,
                conditional=False, abundance=True, abundance_mode="shared",
                a_max="auto", a_cap=None, tilt=None,
                cap_headroom=3.0, tau_buckets=6, sum_mode="grid",
                K=1, gamma=None,
                pi_prior=1.2, pin_weight=100.0,
                mom_target=None, mom_mult=(0.0, 0.0), mom_rho=0.0,
                init=None, a_init=None,
                logits_init=None, max_outer=3, tol=1e-6, global_maxiter=80,
                eta_maxiter=40, finish_maxiter=5000, finish_rounds=6,
                verbose=True, bound_frac=0.001, max_rounds=6,
                a_max_ceiling=4096.0):
    """Fit the free-Pi model to one cell line's (N, S, B) counts.

    Arguments mirror ``fit_effects`` where they mean the same thing; the effect
    law's (``sigma_prior``, ``mu_prior``, ``shash``, the gauge pins) are absent
    and ``pi_prior`` replaces them.  ``a_max="auto"`` runs the same escalating
    abundance bound.

    ``mom_target=(m, v)`` turns on the cross-line moment constraint described in
    the module docstring: this line's activity is pushed toward mean m and
    variance v by the method of multipliers, with ``mom_mult=(lam, nu)`` the
    current Lagrange multipliers (nats per unit of the moment) and ``mom_rho``
    the augmentation weight (nats per unit squared).  ``mom_target=None`` -- the
    default -- leaves the objective bit-identical to the unconstrained fit.

    ``tilt`` is a fixed (N, S, B) -- or (S, B), or (N, B) -- multiplicative
    factor on the latent rate,

        T[n,s,b] ~ Poisson(a_n * Pi[n,b] * tilt[n,s,b] * lambda_s),

    the per-sample bin x covariate distortion of ``batch.py``.  It is an
    OFFSET, not a parameter: it is identified by replication across samples,
    not by one sample's counts, so it is estimated outside and fixed here.
    ``Pi`` then means the profile the object would have shown WITHOUT the
    sample's technical tilt, which is what the activity is read out of.

    Note what this can and cannot do.  With S = 1 the free-Pi rate is
    saturated: ``a_n * Pi[n,b] * tilt[n,b]`` and ``a'_n * Pi'[n,b]`` are the
    same model, so the tilt cannot change the likelihood and only relabels the
    profile (exactly, up to ``pi_prior``'s pull).  With S > 1 the profile is
    shared across replicates while the tilt is not, so it does real work.

    ``abundance_mode`` is ``"shared"`` (one a_n per object, the default) or
    ``"replicate"`` (one a_{n,s} per object and replicate -- see the module
    docstring).  With S = 1 the two are the same model and the replicate path
    reproduces the shared one exactly.  ``a`` on the result is then (N, S) and
    ``a_cap`` stays (N,): the cap sizes the object's tau grid, which is shared
    by its replicates, so it must bound the LARGEST of them.

    Returns a ``FitResult`` with ``mu``/``sigma``/``cuts`` recovered by probit
    least squares from the fitted Pi -- a SUMMARY of the profile, not fitted
    parameters, provided so the netCDF export and anything expecting an effect
    law keep working.
    """
    if abundance_mode not in ("shared", "replicate"):
        raise ValueError(f"unknown abundance_mode {abundance_mode!r}")
    if isinstance(a_max, str):
        if a_max != "auto":
            raise ValueError(f"a_max must be a number or 'auto', got {a_max!r}")
        return _adaptive(locals())
    t0 = time.time()
    X = np.asarray(X)
    N, S, B = X.shape
    a_max = float(a_max)
    replicate_a = abundance and abundance_mode == "replicate"
    if abundance_mode == "replicate" and not abundance:
        raise ValueError("abundance_mode='replicate' needs abundance=True")
    n_a = N * S if replicate_a else N

    obs_mask = ~np.isnan(X) if np.issubdtype(X.dtype, np.floating) \
        else np.ones(X.shape, bool)
    if mask is not None:
        obs_mask = obs_mask & np.asarray(mask, bool)
    observed = obs_mask.any(axis=(1, 2))
    all_zero = np.where(obs_mask, np.nan_to_num(X, nan=0.0), 0.0
                        ).sum(axis=(1, 2)) == 0
    kept = (observed & ~all_zero) if conditional else observed
    n_zero_dropped = int((observed & all_zero).sum()) if conditional else 0
    n_obs = int(obs_mask[kept].sum()) if conditional else int(obs_mask.sum())
    if not kept.any():
        raise ValueError("no objects left to fit")

    if init is None:
        if verbose:
            print("[init] fitting per-column ZINBs / quantile transform ...")
        init = initialize(X, mask=mask, lambda_init=lambda_init, verbose=False)

    # replicate s of object n informs a_{n,s} only where it has an observed
    # cell; that is also the set its gauge pin averages over
    kept_rep = kept[:, None] & obs_mask.any(axis=2)              # (N, S)
    if abundance:
        if a_cap is None:
            a_cap = (np.clip(cap_headroom * _replicate_a0(X, obs_mask,
                                                          kept_rep).max(axis=1),
                             1.0, a_max) if replicate_a
                     else abundance_caps(X, obs_mask, kept, a_max,
                                         headroom=cap_headroom))
        # ONE cap per object even in replicate mode: it sizes the tau grid, and
        # an object's replicates share that grid, so it has to bound the
        # largest of them
        a_cap = np.clip(np.asarray(a_cap, float).reshape(-1), 1e-3, a_max)
        rate_max = lambda_fix * a_cap
    else:
        a_cap, rate_max = None, lambda_fix
    # the tau grid must bound a_n * Pi * tilt * lambda for EVERY (s, b) the
    # likelihood touches, so the object's ceiling carries its largest tilt
    if tilt is not None:
        tilt_np = broadcast_tilt(tilt, N, S, B)
        rate_max = np.asarray(rate_max, float) * tilt_np.max(axis=(1, 2))
        tilt_j = jnp.asarray(tilt_np)
    builder = LoglikBuilder(X, mask=mask, rate_max=rate_max,
                            max_buckets=tau_buckets, sum_mode=sum_mode)
    K = int(K)
    if K > 1 and gamma is None:
        raise ValueError("K > 1 needs the (N, K) responsibilities; this is an "
                         "EM M-step, gamma is formed by the E-step")
    gamma_j = None if K == 1 else jnp.asarray(
        np.asarray(gamma, float).reshape(N, K))
    spec = _Globals(S, B, lambda_fix, K)
    scale = 1.0 / max(n_obs, 1)
    kept_j = jnp.asarray(kept)
    kept_rep_j = jnp.asarray(kept_rep)
    if verbose:
        grid = (f"tau_window={builder.window}" if sum_mode == "window" else
                f"tau_grid={builder.tau_lengths} "
                f"({builder.tau_work:.0f} nodes/object)")
        print(f"[free-Pi] N={N} S={S} B={B} kept={int(kept.sum())} "
              f"conditional={conditional} abundance={abundance}"
              f"({abundance_mode}) a_max={a_max:g} {grid} "
              f"pi_prior={pi_prior}")

    def _pi(eta):
        return jax.nn.softmax(eta[:N * B].reshape(N, B), axis=1)

    def _rows(eta):
        Pi = _pi(eta)
        if not abundance:
            return Pi if tilt is None else Pi[:, None, :] * tilt_j
        a = jnp.exp(eta[N * B:N * B + n_a])
        if replicate_a:
            # (N, S, B): the profile is shared, the depth is not
            rows = a.reshape(N, S)[:, :, None] * Pi[:, None, :]
            return rows if tilt is None else rows * tilt_j
        if tilt is not None:
            # (N, S, B): the profile is shared, the technical tilt is not
            return a[:, None, None] * Pi[:, None, :] * tilt_j
        return a[:, None] * Pi

    def _data_neg_ll(theta, eta):
        R, P, rates, log_w, phi = spec.unpack(theta)
        M = _rows(eta)

        def one(Rk, Pk):
            """(N,) per-object log-likelihood under one emission component."""
            if conditional:
                obj = builder.object_loglik(Rk, Pk, M, rates, log_w, 0.0)
                lz = jnp.minimum(log_zero_prob(Rk, Pk, M, rates, log_w,
                                               builder.mask), -1e-12)
                return jnp.where(kept_j, obj - log1mexp(lz), 0.0)
            return builder.object_loglik(Rk, Pk, M, rates, log_w, phi)

        if K == 1:
            total = jnp.sum(one(R[0], P[0]))
        else:
            # EM M-step: the responsibility-weighted data term.  A weighted SUM
            # over components, not a logsumexp -- Q separates over cell lines
            # and the component index is fixed by the E-step.
            total = jnp.sum(gamma_j * jnp.stack(
                [one(R[k], P[k]) for k in range(K)], axis=1))
        return -total * scale

    def _pi_pen(eta):
        """-log symmetric Dirichlet(pi_prior) on the profile rows, on the same
        1/n_obs scale as the data term.  From log_softmax, so it stays finite
        as a row approaches a vertex instead of overflowing there."""
        if pi_prior is None or pi_prior == 1.0:
            return 0.0
        lp = jax.nn.log_softmax(eta[:N * B].reshape(N, B), axis=1)
        return -scale * (pi_prior - 1.0) * jnp.sum(
            jnp.where(kept_j[:, None], lp, 0.0))

    def _a_pen(eta):
        """a_n shares a scale ridge with R, so pin its geometric mean to 1.

        With a replicate-specific abundance each column a[:, s] rides its own
        R[s, :] ridge, so there are S independent ridges and S pins -- one
        shared pin would leave S-1 flat directions.
        """
        if not abundance:
            return 0.0
        la = eta[N * B:N * B + n_a]
        if replicate_a:
            gm = (jnp.sum(jnp.where(kept_rep_j, la.reshape(N, S), 0.0), axis=0)
                  / jnp.maximum(jnp.sum(kept_rep_j, axis=0), 1.0))     # (S,)
            return pin_weight * jnp.sum(jnp.square(gm))
        return pin_weight * jnp.square(
            jnp.sum(jnp.where(kept_j, la, 0.0)) / jnp.sum(kept_j))

    bins_j = jnp.arange(1, B + 1, dtype=float)
    n_kept = float(max(int(kept.sum()), 1))
    if mom_target is not None:
        mom_target = (float(mom_target[0]), float(mom_target[1]))
        mom_mult = (float(mom_mult[0]), float(mom_mult[1]))
        # one rho per constraint: a variance violation is ~20x smaller in
        # absolute terms than a mean violation of the same practical size, so a
        # single augmentation weight conditions one of the two badly
        mom_rho = ((float(mom_rho), float(mom_rho))
                   if np.ndim(mom_rho) == 0 else
                   (float(mom_rho[0]), float(mom_rho[1])))

    def _moments(eta):
        """(mean, variance) of the plug-in activity E[bin] over kept objects.

        The population variance (1/n, not 1/(n-1)): with n in the tens of
        thousands the difference is far below the constraint tolerance, and
        this way the two moments are both plain averages of a per-object term,
        which is what makes the gradient one pass over the profile rows.
        """
        e = _pi(eta) @ bins_j
        w = jnp.where(kept_j, 1.0, 0.0)
        m1 = jnp.sum(w * e) / n_kept
        m2 = jnp.sum(w * jnp.square(e - m1)) / n_kept
        return m1, m2

    def _mom_pen(eta):
        """Augmented Lagrangian of the two cross-line moment constraints.

        On the 1/n_obs scale of the data term, so ``mom_mult`` and ``mom_rho``
        are in nats (per unit of the moment, and per unit squared): a multiplier
        of 1000 means the fit is willing to give up 1000 nats to move the mean
        activity by one bin.
        """
        m1, m2 = _moments(eta)
        d1, d2 = m1 - mom_target[0], m2 - mom_target[1]
        return scale * (mom_mult[0] * d1 + mom_mult[1] * d2
                        + 0.5 * (mom_rho[0] * d1 * d1 + mom_rho[1] * d2 * d2))

    def _neg_obj(theta, eta):
        v = _data_neg_ll(theta, eta) + _pi_pen(eta) + _a_pen(eta)
        # a Python-level branch, so an unconstrained fit traces the same graph
        # it did before this argument existed
        return v if mom_target is None else v + _mom_pen(eta)

    vg_global = jax.jit(jax.value_and_grad(_neg_obj, argnums=0))
    vg_eta = jax.jit(jax.value_and_grad(
        lambda eta, theta: _neg_obj(theta, eta), argnums=0))
    n_glob = spec.n_params
    vg_joint = jax.jit(jax.value_and_grad(
        lambda x: _neg_obj(x[:n_glob], x[n_glob:])))
    ll_fn = jax.jit(lambda th, e: -_data_neg_ll(th, e) / scale)
    pll_fn = jax.jit(lambda th, e: -_neg_obj(th, e) / scale)

    theta = spec.pack(init)
    g_bounds = spec.bounds()
    if logits_init is None:
        # START AT THE SATURATED SOLUTION.  Given (R, P, a) the free-Pi optimum
        # of a deep object is essentially Pi ∝ counts / v, with
        # v[b] = sum_s c[s,b] lambda_s the per-bin channel scale -- the reads
        # ARE the profile, tilted by the emission.  Starting from the ZINB
        # init's Pi instead leaves L-BFGS-B to walk N*B logits (150k on a
        # 30k-object line) to that point, and it does not get there: measured
        # on a deep line, the top decile ended 0.016/bin from the saturated
        # optimum, i.e. FURTHER from the data than the 2-parameter ordered
        # probit, which a saturated model cannot be when converged.
        Xz = np.where(obs_mask, np.nan_to_num(X, nan=0.0), 0.0).sum(1)   # (N,B)
        Rz, Pz = np.asarray(init.R, float), np.asarray(init.P, float)
        lz = np.asarray(init.Lambda, float).reshape(-1)
        if Rz.ndim == 3:            # (K, S, B): average the channel scale over
            Rz, Pz = Rz.mean(0), Pz.mean(0)      # components, this only starts it
        v = np.einsum("sb,s->b", Rz * (1 - Pz) / Pz, lz)
        share = Xz / np.maximum(v, 1e-12)[None, :]
        share = share / np.maximum(share.sum(1, keepdims=True), 1e-12)
        # objects with no reads have no profile to recover; give them the flat
        # row rather than a degenerate one
        flat = ~np.isfinite(share).all(1) | (Xz.sum(1) <= 0)
        share[flat] = 1.0 / B
        z0 = np.log(np.clip(share, 1e-8, None))
    else:
        z0 = np.asarray(logits_init, float).reshape(N, B).copy()
    z0 = z0 - z0.mean(1, keepdims=True)            # softmax is shift-invariant
    eta = z0.reshape(-1)
    e_bounds = [(-30.0, 30.0)] * (N * B)
    if abundance and replicate_a:
        cap = a_cap[:, None]
        a0 = (np.asarray(a_init, float).reshape(N, S) if a_init is not None
              else _replicate_a0(X, obs_mask, kept_rep))
        la0 = np.log(np.clip(a0, 1e-3, cap * 0.9))
        la0 = np.where(kept_rep, la0, 0.0)
        # centre WITHIN each replicate, to match the per-replicate pin
        for s in range(S):
            sel = kept_rep[:, s]
            if sel.any():
                la0[:, s] -= la0[sel, s].mean()
        la0 = np.where(kept_rep, np.clip(la0, np.log(1e-4), np.log(cap)), 0.0)
        eta = np.concatenate([eta, la0.reshape(-1)])           # row-major [n,s]
        e_bounds += [(np.log(1e-4), float(np.log(c)))
                     for c in a_cap for _ in range(S)]
    elif abundance:
        la0 = (np.log(np.clip(np.asarray(a_init, float), 1e-3, a_cap * 0.9))
               if a_init is not None
               else np.log(np.clip(initial_abundance(X, obs_mask, kept),
                                   1e-3, a_cap * 0.9)))
        la0 = np.asarray(la0, float).copy()
        la0[~kept] = 0.0
        la0 -= la0[kept].mean()
        la0 = np.clip(la0, np.log(1e-4), np.log(a_cap))
        eta = np.concatenate([eta, la0])
        e_bounds += [(np.log(1e-4), float(np.log(c))) for c in a_cap]

    history, cyc = [], 0
    ll = float(ll_fn(jnp.asarray(theta), jnp.asarray(eta)))
    pll = float(pll_fn(jnp.asarray(theta), jnp.asarray(eta)))

    def record(stage, ll):
        history.append(dict(stage=stage, loglik=ll, time=time.time() - t0))
        if verbose:
            print(f"[{time.time()-t0:7.1f}s] {stage:<24} loglik = {ll:.4f}",
                  flush=True)

    record("init", ll)
    for outer in range(1, max_outer + 1):
        prev = pll
        opt = _GLOBAL_CYCLE[cyc % len(_GLOBAL_CYCLE)]
        res = minimize(lambda th: tuple(map(np.asarray, vg_global(
            jnp.asarray(th), jnp.asarray(eta)))), theta, jac=True, method=opt,
            bounds=g_bounds, options=_solver_options(opt, global_maxiter))
        if -res.fun / scale >= pll - 1e-12:
            theta = res.x
        new = float(pll_fn(jnp.asarray(theta), jnp.asarray(eta)))
        if new - pll < tol * 10 * n_obs:
            cyc += 1
        pll = new
        ll = float(ll_fn(jnp.asarray(theta), jnp.asarray(eta)))
        record(f"outer{outer}:global({opt})", ll)

        res = minimize(lambda e: tuple(map(np.asarray, vg_eta(
            jnp.asarray(e), jnp.asarray(theta)))), eta, jac=True,
            method="L-BFGS-B", bounds=e_bounds,
            options=_solver_options("L-BFGS-B", eta_maxiter))
        if -res.fun / scale >= pll - 1e-12:
            eta = res.x
        ll = float(ll_fn(jnp.asarray(theta), jnp.asarray(eta)))
        pll = float(pll_fn(jnp.asarray(theta), jnp.asarray(eta)))
        record(f"outer{outer}:eta(L-BFGS-B)", ll)
        if pll - prev < tol * n_obs:
            break

    converged, finish_opt = False, "TNC"
    for rnd in range(1, finish_rounds + 1):
        prev = pll
        res = minimize(lambda x: tuple(map(np.asarray, vg_joint(jnp.asarray(x)))),
                       np.concatenate([theta, eta]), jac=True, method=finish_opt,
                       bounds=g_bounds + e_bounds,
                       options=_solver_options(finish_opt, finish_maxiter))
        if -res.fun / scale >= pll - 1e-12:
            theta, eta = res.x[:n_glob], res.x[n_glob:]
            ll = float(ll_fn(jnp.asarray(theta), jnp.asarray(eta)))
            pll = float(pll_fn(jnp.asarray(theta), jnp.asarray(eta)))
        record(f"finish{rnd}:joint({finish_opt})", ll)
        gain = pll - prev
        healthy = res.success or getattr(res, "status", -1) == 3
        if finish_opt == "TNC" and (not healthy or gain < tol * n_obs):
            finish_opt = "L-BFGS-B"
            if gain >= tol * n_obs:
                continue
        elif gain < tol * n_obs:
            converged = True
            break

    Pi = np.asarray(_pi(jnp.asarray(eta)))
    if not abundance:
        a_hat = np.ones(N)
    elif replicate_a:
        a_hat = np.exp(np.asarray(eta[N * B:N * B + n_a])).reshape(N, S)
    else:
        a_hat = np.exp(np.asarray(eta[N * B:N * B + N]))
    R, P, rates, log_w, phi = spec.unpack(jnp.asarray(theta))
    # a probit SUMMARY of the fitted profile, so downstream code that expects an
    # effect law keeps working -- these are not fitted parameters
    mu_s, sig_s = _eta_init(Pi, np.arange(1, B) / B)
    return FitResult(
        # (K, S, B) when K > 1; squeezed to (S, B) at K = 1 so every existing
        # consumer of a single-emission fit keeps its shape
        R=np.asarray(R)[0] if K == 1 else np.asarray(R),
        P=np.asarray(P)[0] if K == 1 else np.asarray(P), mu=mu_s, sigma=sig_s,
        cuts=np.asarray([float(v) for v in np.quantile(
            mu_s[kept], np.arange(1, B) / B)]),
        Pi=Pi, rates=np.asarray(rates), log_w=np.asarray(log_w),
        phi=0.0 if conditional else float(phi), loglik=ll, a=a_hat,
        a_cap=a_cap, n_observed=n_obs, converged=converged, observed=kept,
        n_zero_dropped=n_zero_dropped, history=history,
        extras=dict(free_pi=True, abundance_mode=abundance_mode,
                    # what the outer loop needs: where this line's two moments
                    # ended up, and under which multipliers it got there
                    mom_moments=tuple(float(x) for x in
                                      _moments(jnp.asarray(eta))),
                    mom_target=mom_target, mom_mult=mom_mult,
                    mom_rho=mom_rho),
        config=dict(lambda_fix=lambda_fix, conditional=conditional,
                    abundance=abundance, abundance_mode=abundance_mode,
                    a_max=a_max, abundance_prior=None,
                    n_abund_nodes=0, kappa=None, sigma_prior=None,
                    sigma_shared=False, mu_prior=None, shash=None,
                    eps_prior=None, free_pi=True, pi_prior=pi_prior,
                    mom_target=mom_target, K=K,
                    tau_buckets=tau_buckets, sum_mode=sum_mode,
                    pin_median=None, pin_left=None, pin_weight=pin_weight,
                    pi_floor=0.0))


def _adaptive(kw):
    """``a_max="auto"``: raise the per-object abundance bound until it stops
    binding, warm-starting each round (mirrors ``fit.fit_effects_adaptive``)."""
    kw = dict(kw)
    for k in ("t0",):
        kw.pop(k, None)
    bound_frac = kw.pop("bound_frac")
    max_rounds = int(kw.pop("max_rounds"))
    a_ceiling = float(kw.pop("a_max_ceiling"))
    ceiling = float(min(20.0, a_ceiling))
    kw.pop("a_max")
    verbose = kw.get("verbose", True)
    if not kw.get("abundance", True):
        return fit_free_pi(a_max=20.0, bound_frac=bound_frac,
                           max_rounds=1, **kw)
    res, a_cap = None, kw.pop("a_cap", None)
    for rnd in range(1, max_rounds + 1):
        if res is not None:
            # warm-start the GLOBAL block too.  Passing init=None re-ran the
            # per-column ZINB initialization every round AND threw away the
            # fitted (R, P, phi), so each escalation restarted the channel
            # block from scratch -- the ladder then pays for a cold fit per
            # round instead of a warm one.
            from .initialize import init_from_fit
            kw = dict(kw, init=init_from_fit(res), a_init=res.a,
                      logits_init=np.log(np.clip(res.Pi, 1e-12, None)))
        res = fit_free_pi(a_max=ceiling, a_cap=a_cap, bound_frac=bound_frac,
                          max_rounds=1, a_max_ceiling=a_ceiling, **kw)
        cap, a = res.a_cap, res.a
        # the cap is per OBJECT (it sizes that object's tau grid), so a
        # replicate-specific a is reduced over its replicates before the test
        a = a if a.ndim == 1 else a.max(axis=1)
        keep = np.asarray(res.observed, bool)
        at_bound = keep & (a >= (1.0 - 1e-3) * cap)
        frac = at_bound.sum() / max(int(keep.sum()), 1)
        if verbose:
            print(f"[a_max] round {rnd}: {at_bound.sum()} ({frac:.3%}) at the "
                  f"bound; ceiling={ceiling:g} max a={a[keep].max():.3g}",
                  flush=True)
        if frac <= bound_frac:
            break
        new_cap = np.where(at_bound, np.minimum(2.0 * cap, a_ceiling), cap)
        new_ceiling = float(min(max(ceiling, new_cap.max()), a_ceiling))
        if np.array_equal(new_cap, cap) and new_ceiling == ceiling:
            break
        a_cap, ceiling = new_cap, new_ceiling
    return res
