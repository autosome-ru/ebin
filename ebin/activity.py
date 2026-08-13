"""Posterior-mean activity E[bin] and its uncertainty.

The plug-in readout E[bin] = sum_b b * Pi[n,b] at the fitted (mu_n, sigma_n) is
a MODE.  For a low-information object the posterior over (mu_n, sigma_n) is
diffuse and skewed, so its mode sits near a simplex vertex (activity 1 or B,
regardless of depth) while the posterior MEAN is a mild tilt with a large SD --
the honest summary, and the one that ships.

E[bin] is a functional of theta = (mu_n, sigma_n), so its posterior mean is

    activity_n = sum_j w_{n,j} ebin(theta_j),  w_{n,j} ~ p(X_n | theta_j) p(theta_j)

on a shared (mu, log sigma) grid.  The abundance is profiled per grid point (a
size factor a = total_n / E[total | a = 1, theta_j]), so a deep object's
likelihood is sharply peaked and a 3-read object's is flat: depth enters
honestly instead of being frozen at the joint mode.  The prior is the same
penalty the fit used, so the posterior mean is coherent with the fitted model.

Under an emission mixture the grid gains a component axis and the summary
marginalizes over it as well, weighted by ``log_gamma``, the responsibilities
formed from the other cell lines (``MixtureState.log_gamma_loo``).  The
profiled abundance differs per component -- the same reads imply fewer cells
when the sequence amplifies well -- so it is profiled per (component, grid
point) too, and everything still comes from one set of weights.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .model import LoglikBuilder, as_components
from .effects import normal_bin_probs
from .truncation import log_zero_prob, log1mexp

jax.config.update("jax_enable_x64", True)


def _unit_nodes(n):
    """n symmetric nodes on [-1, 1]; a single node sits at the centre."""
    return np.zeros(1) if n == 1 else np.linspace(-1.0, 1.0, n)


def posterior_activity(res, X, mask, *, mu_center=0.0, tau_within="auto",
                       sigma_prior="auto", conditional="auto",
                       log_gamma=None, n_mu=121, n_ls=33,
                       mu_lim=12.0, ls_lim=(0.2, 5.0),
                       adaptive=True, adapt_k=5.0):
    """Posterior summary of every object of one cell line.

    res         : a fitted ``FitResult`` (supplies R, P, rates, cuts, config).
    X, mask     : the same (N, S, B) arrays the fit saw.
    mu_center   : scalar or (N,) shrinkage target of the mu prior (0 = neutral).
    tau_within  : sd of the mu prior; "auto" takes the fit's ``mu_prior``,
                  None gives a flat effect location.
    sigma_prior : (m, s) log-sigma prior; "auto" takes the fit's.
    conditional : whether the per-object likelihood is zero-truncated; "auto"
                  takes the fit's.  Under zero inflation phi is a global
                  constant and drops out of the per-object weights.
    log_gamma   : (N, K) unnormalized log weights of the emission components,
                  for a fit with K > 1.  Pass
                  ``state.log_gamma_loo(cell_line)`` so this line's own reads
                  are not counted once through the component and again through
                  the likelihood.  ``None`` weights the components equally.
    adaptive    : re-grid every object around its own posterior (two passes; see
                  the module docstring).  False = the plain shared grid, which
                  quantizes deep objects onto single nodes.
    adapt_k     : half-width of the per-object window in posterior SDs.  One
                  shared-grid cell is added on top: that is what brackets an
                  object whose first pass collapsed onto a single node and so
                  reported an SD of zero.

    Everything comes from the same posterior weights, so the summaries are
    mutually consistent (activity == bin_prob @ [1..B]).  Returns a
    dict of (N,) arrays, NaN off ``res.observed``:

      activity, activity_sd   posterior mean and SD of E[bin]
      activity_map            the plug-in mode, for reference
      mu, sigma               posterior-mean effect location and scale
      mu_sd, sigma_sd         their posterior SDs (standard errors)
      abundance               posterior-mean profiled size factor a_n
      activity_edge           weight on the adaptive window's boundary (0 under
                              adaptive=False); ~0 means the window held the
                              whole posterior
      n_eff                   posterior grid support (larger = more diffuse)
      tot                     observed reads
      component               most probable emission component (1-based; all 1
                              when K = 1)
      observed                (N,) bool

    plus bin_prob, the (N, B) posterior-mean distribution over bins, comp_prob,
    the (N, K) posterior over emission components, and cuts, the (B-1,) bin
    boundaries of this cell line -- one set for the whole line, not one per
    sequence.
    """
    cfg = res.config
    X = np.asarray(X)
    N, S, B = X.shape
    bvec = np.arange(1, B + 1, dtype=float)
    lambda_fix = cfg["lambda_fix"]
    # profile the abundance up to a_max whenever the fit modelled one, whether
    # as a free a_n or integrated out under the Gamma prior
    a_max = float(cfg["a_max"]
                  if (cfg["abundance"] or cfg.get("abundance_prior")) else 1.0)
    if tau_within == "auto":
        tau_within = cfg.get("mu_prior")
    if isinstance(sigma_prior, str):
        sigma_prior = cfg.get("sigma_prior")
    if conditional == "auto":
        conditional = cfg.get("conditional", True)
    if cfg.get("sigma_shared"):
        # the effect scale is a property of the cell line, not of the object:
        # there is nothing per-object to integrate over, so the grid collapses
        # onto the fitted value and the log-sigma prior does not apply.  This
        # overrides n_ls / ls_lim rather than honouring them -- a grid would
        # give this model a per-object sigma it does not have.
        s_fix = float(np.median(np.asarray(res.sigma, dtype=float)))
        ls_lim, n_ls, sigma_prior = (s_fix, s_fix), 1, None

    cuts = jnp.asarray(res.cuts)
    R, P = as_components(res.R), as_components(res.P)            # (K, S, B)
    K = R.shape[0]
    rates, log_w = jnp.asarray(res.rates), jnp.asarray(res.log_w)
    lam = np.asarray(res.rates)[:, 0]                          # (S,) 1 rate
    Rn, Pn = np.asarray(R), np.asarray(P)
    c = Rn * (1 - Pn) / Pn                                     # (K, S, B)

    obs_mask = ~np.isnan(X) if np.issubdtype(X.dtype, np.floating) \
        else np.ones(X.shape, bool)
    if mask is not None:
        obs_mask = obs_mask & np.asarray(mask, bool)
    total = np.where(obs_mask, np.nan_to_num(X, nan=0.0), 0.0).sum((1, 2))

    # The readout does not fit a_n, it PROFILES it: a = tot / (Pi . v) clipped
    # to a_max, with v[k,b] = sum_s c[k,s,b] lambda_s.  Pi is a simplex row, so
    # Pi . v >= min_b v[k,b] and the profiled abundance can never exceed
    # tot / min_{k,b} v[k,b], whatever the grid point.  That is an exact
    # per-object ceiling on the latent rate, and it is far below a_max for
    # every object but the deepest -- which are the only ones that then pay for
    # the long tau grid.
    v_kb = np.einsum("ksb,s->kb", c, lam)                      # (K, B)
    a_ceil = np.clip(total / max(float(v_kb.min()), 1e-12), 1e-4, a_max)
    builder = LoglikBuilder(X, mask=mask, rate_max=lambda_fix * a_ceil,
                            max_buckets=int(cfg.get("tau_buckets", 6) or 1))
    kept = np.asarray(res.observed)

    # ---- pass 1: shared (mu, log sigma) grid, its readout and expected total
    mu_g = np.linspace(-mu_lim, mu_lim, n_mu)
    ls_g = np.linspace(np.log(ls_lim[0]), np.log(ls_lim[1]), n_ls)
    MU, LS = np.meshgrid(mu_g, ls_g, indexing="ij")
    MU, LS = MU.ravel(), LS.ravel()
    G = MU.size
    Pi_g = np.asarray(normal_bin_probs(jnp.asarray(MU), jnp.exp(jnp.asarray(LS)),
                                       cuts))
    ebin_g = Pi_g @ bvec                                        # (G,)
    # E[tot | a = 1] under each component: (K, G).  Equivalently Pi_g @ v_kb --
    # the form the tau ceiling above is bounded from.
    denom_g = np.stack([(c[k][None] * Pi_g[:, None, :]).sum(2) @ lam
                        for k in range(K)])

    def ll_grid_for(Rk, Pk):
        """One jitted grid evaluation per component.  The emission tables are
        closed over, not passed in: XLA folds them into the kernel, and with
        K = 1 that reproduces the single-emission readout exactly."""
        @jax.jit
        def ll_grid(pi_row, a_col):
            M = a_col[:, None] * pi_row[None, :]
            obj = builder.object_loglik(Rk, Pk, M, rates, log_w, 0.0)
            if not conditional:
                return obj
            lz = jnp.minimum(
                log_zero_prob(Rk, Pk, M, rates, log_w, builder.mask), -1e-12)
            return obj - log1mexp(lz)
        return ll_grid

    ll = np.empty((N, K, G))
    a_grid = np.empty((N, K, G))
    for k in range(K):
        ll_grid = ll_grid_for(R[k], P[k])
        for j in range(G):
            a_col = np.clip(total / max(denom_g[k, j], 1e-12), 1e-4, a_max)
            a_grid[:, k, j] = a_col
            ll[:, k, j] = np.asarray(ll_grid(jnp.asarray(Pi_g[j]),
                                             jnp.asarray(a_col)))

    # per-object prior over the grid: N(mu; center_n, tau^2) * N(ls; m_s, s_s)
    center = np.broadcast_to(np.asarray(mu_center, float), (N,))
    logprior = np.zeros((N, G))
    if tau_within is not None:
        logprior = logprior - (MU[None, :] - center[:, None]) ** 2 \
            / (2.0 * tau_within ** 2)
    if sigma_prior is not None:
        m_s, s_s = sigma_prior
        logprior = logprior - ((LS - m_s) ** 2 / (2.0 * s_s ** 2))[None, :]

    lg_off = None
    if K > 1:
        lg = (np.zeros((N, K)) if log_gamma is None
              else np.asarray(log_gamma, dtype=float))
        if lg.shape != (N, K):
            raise ValueError(f"log_gamma must be {(N, K)}, got {lg.shape}")
        lg_off = lg - lg.max(1, keepdims=True)

    # (N, K, G) in float64 is already gigabytes at N = 30000; build the
    # posterior in place rather than in fresh copies
    logpost = ll
    logpost += logprior[:, None, :]
    if K > 1:
        logpost += lg_off[:, :, None]
    logpost = logpost.reshape(N, K * G)
    a_grid = a_grid.reshape(N, K * G)
    logpost -= logpost.max(1, keepdims=True)
    w = np.exp(logpost)
    w /= w.sum(1, keepdims=True)

    if adaptive:
        # ---- pass 2: one grid per object ---------------------------------
        # pass 1 located each posterior (marginally over the components); each
        # object is re-gridded over mu +- (k sd + one shared cell).  The floor
        # is what saves the collapsed objects: with all the weight on a single
        # node their sd reads 0, and the peak is then within one cell of it.
        MU_t, LS_t = np.tile(MU, K), np.tile(LS, K)
        mu_c, ls_c = w @ MU_t, w @ LS_t
        d_mu = 2.0 * mu_lim / (n_mu - 1) if n_mu > 1 else 0.0
        ls_lo, ls_hi = np.log(ls_lim[0]), np.log(ls_lim[1])
        d_ls = (ls_hi - ls_lo) / (n_ls - 1) if n_ls > 1 else 0.0
        hw_mu = adapt_k * np.sqrt(np.clip(w @ (MU_t ** 2) - mu_c ** 2,
                                          0, None)) + d_mu
        hw_ls = adapt_k * np.sqrt(np.clip(w @ (LS_t ** 2) - ls_c ** 2,
                                          0, None)) + d_ls
        # a window that runs off the fitted support is clipped, not translated
        lo_mu, hi_mu = (np.clip(mu_c - hw_mu, -mu_lim, mu_lim),
                        np.clip(mu_c + hw_mu, -mu_lim, mu_lim))
        lo_ls, hi_ls = (np.clip(ls_c - hw_ls, ls_lo, ls_hi),
                        np.clip(ls_c + hw_ls, ls_lo, ls_hi))
        mu_nodes = (0.5 * (lo_mu + hi_mu)[:, None]
                    + 0.5 * (hi_mu - lo_mu)[:, None] * _unit_nodes(n_mu)[None])
        ls_nodes = (0.5 * (lo_ls + hi_ls)[:, None]
                    + 0.5 * (hi_ls - lo_ls)[:, None] * _unit_nodes(n_ls)[None])

        v_j = jnp.asarray(v_kb)                     # E[tot | a=1] per (K, B)
        tot_j = jnp.asarray(total)

        @jax.jit
        def bin_probs(mu_col, ls_col):
            return normal_bin_probs(mu_col, jnp.exp(ls_col), cuts)

        @jax.jit
        def abund(Pi, v_row):
            return jnp.clip(tot_j / jnp.maximum(Pi @ v_row, 1e-12), 1e-4, a_max)

        def ll_col_for(Rk, Pk):
            @jax.jit
            def ll_col(Pi, a_col):
                M = a_col[:, None] * Pi
                obj = builder.object_loglik(Rk, Pk, M, rates, log_w, 0.0)
                if not conditional:
                    return obj
                lz = jnp.minimum(
                    log_zero_prob(Rk, Pk, M, rates, log_w, builder.mask), -1e-12)
                return obj - log1mexp(lz)
            return ll_col

        ll_cols = [ll_col_for(R[k], P[k]) for k in range(K)]
        logpost = np.empty((N, K, G))
        for i in range(n_mu):
            mu_col = jnp.asarray(mu_nodes[:, i])
            lp_mu = np.zeros(N) if tau_within is None else \
                -(mu_nodes[:, i] - center) ** 2 / (2.0 * tau_within ** 2)
            for t in range(n_ls):
                j = i * n_ls + t
                lp = lp_mu if sigma_prior is None else \
                    lp_mu - (ls_nodes[:, t] - m_s) ** 2 / (2.0 * s_s ** 2)
                Pi = bin_probs(mu_col, jnp.asarray(ls_nodes[:, t]))
                for k in range(K):
                    logpost[:, k, j] = np.asarray(
                        ll_cols[k](Pi, abund(Pi, v_j[k]))) + lp
        if K > 1:
            logpost += lg_off[:, :, None]
        logpost = logpost.reshape(N, K * G)
        logpost -= logpost.max(1, keepdims=True)
        w = np.exp(logpost)
        w /= w.sum(1, keepdims=True)
        w = w.reshape(N, K, G)

        # second sweep: redoing the readout is cheap, holding an (N, K, G, B)
        # tensor of per-object bin probs is not -- accumulate node by node
        acc = {q: np.zeros(N) for q in ("ebin", "ebin2", "a", "mu", "mu2",
                                        "sig", "sig2", "edge")}
        bin_prob = np.zeros((N, B))
        for i in range(n_mu):
            mu_col = jnp.asarray(mu_nodes[:, i])
            for t in range(n_ls):
                j = i * n_ls + t
                Pi = bin_probs(mu_col, jnp.asarray(ls_nodes[:, t]))
                for k in range(K):
                    acc["a"] += w[:, k, j] * np.asarray(abund(Pi, v_j[k]))
                Pi = np.asarray(Pi)
                wj = w[:, :, j].sum(1)
                e_j = Pi @ bvec
                mu_j, sig_j = mu_nodes[:, i], np.exp(ls_nodes[:, t])
                acc["ebin"] += wj * e_j
                acc["ebin2"] += wj * e_j ** 2
                acc["mu"] += wj * mu_j
                acc["mu2"] += wj * mu_j ** 2
                acc["sig"] += wj * sig_j
                acc["sig2"] += wj * sig_j ** 2
                bin_prob += wj[:, None] * Pi
                if i in (0, n_mu - 1) or t in (0, n_ls - 1):
                    acc["edge"] += wj

        comp_prob = w.sum(2)                                    # (N, K)
        ebin_pm, mu_pm, sigma_pm = acc["ebin"], acc["mu"], acc["sig"]
        sd = lambda m2, m: np.sqrt(np.clip(m2 - m ** 2, 0, None))  # noqa: E731
        return dict(
            comp_prob=np.where(kept[:, None], comp_prob, np.nan),
            component=np.where(kept, comp_prob.argmax(1) + 1.0, np.nan),
            activity=np.where(kept, ebin_pm, np.nan),
            activity_sd=np.where(kept, sd(acc["ebin2"], ebin_pm), np.nan),
            activity_map=np.where(kept, np.asarray(res.Pi) @ bvec, np.nan),
            activity_edge=np.where(kept, acc["edge"], np.nan),
            mu=np.where(kept, mu_pm, np.nan),
            mu_sd=np.where(kept, sd(acc["mu2"], mu_pm), np.nan),
            sigma=np.where(kept, sigma_pm, np.nan),
            sigma_sd=np.where(kept, sd(acc["sig2"], sigma_pm), np.nan),
            abundance=np.where(kept, acc["a"], np.nan),
            n_eff=np.where(kept, 1.0 / (w.reshape(N, K * G) ** 2).sum(1), np.nan),
            tot=np.where(kept, total, np.nan),
            bin_prob=np.where(kept[:, None], bin_prob, np.nan),
            observed=kept,
            cuts=np.asarray(res.cuts, float))

    comp_prob = w.reshape(N, K, G).sum(2)                       # (N, K)
    ebin_g = np.tile(ebin_g, K)
    Pi_g = np.tile(Pi_g, (K, 1))
    MU, LS = np.tile(MU, K), np.tile(LS, K)
    ebin_pm = w @ ebin_g
    ebin_sd = np.sqrt(np.clip(w @ (ebin_g ** 2) - ebin_pm ** 2, 0, None))

    sig_g = np.exp(LS)
    mu_pm = w @ MU
    sigma_pm = w @ sig_g
    return dict(
        comp_prob=np.where(kept[:, None], comp_prob, np.nan),
        component=np.where(kept, comp_prob.argmax(1) + 1.0, np.nan),
        activity=np.where(kept, ebin_pm, np.nan),
        activity_sd=np.where(kept, ebin_sd, np.nan),
        activity_map=np.where(kept, np.asarray(res.Pi) @ bvec, np.nan),
        activity_edge=np.where(kept, 0.0, np.nan),          # shared grid: n/a
        mu=np.where(kept, mu_pm, np.nan),
        mu_sd=np.where(kept, np.sqrt(np.clip(w @ (MU ** 2) - mu_pm ** 2,
                                             0, None)), np.nan),
        sigma=np.where(kept, sigma_pm, np.nan),
        sigma_sd=np.where(kept, np.sqrt(np.clip(w @ (sig_g ** 2)
                                                - sigma_pm ** 2, 0, None)),
                          np.nan),
        abundance=np.where(kept, (w * a_grid).sum(1), np.nan),
        n_eff=np.where(kept, 1.0 / (w ** 2).sum(1), np.nan),
        tot=np.where(kept, total, np.nan),
        bin_prob=np.where(kept[:, None], w @ Pi_g, np.nan),
        observed=kept,
        cuts=np.asarray(res.cuts, float))
