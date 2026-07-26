"""Posterior-mean activity E[bin] and its uncertainty.

The plug-in readout E[bin] = sum_b b * Pi[n,b] at the fitted (mu_n, sigma_n) is
a MODE.  For a low-information object the posterior over (mu_n, sigma_n) is
diffuse and skewed, so its mode sits near a simplex vertex (activity 1 or B,
regardless of depth) while the posterior MEAN is a mild tilt with a large SD --
the honest summary, and the one that ships.

E[bin] is a functional of theta = (mu_n, sigma_n), so its posterior mean is

    activity_n = sum_j w_{n,j} ebin(theta_j),  w_{n,j} ~ p(X_n | theta_j) p(theta_j)

on a shared (mu, log sigma) grid.  The abundance is PROFILED per grid point (a
size factor a = total_n / E[total | a = 1, theta_j]), so a deep object's
likelihood is sharply peaked and a 3-read object's is flat: depth enters
honestly instead of being frozen at the joint mode.  The prior is the same
penalty the fit used, so the posterior mean is coherent with the fitted model.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .model import LoglikBuilder
from .effects import normal_bin_probs
from .truncation import log_zero_prob, log1mexp

jax.config.update("jax_enable_x64", True)


def posterior_activity(res, X, mask, *, mu_center=0.0, tau_within="auto",
                       sigma_prior="auto", conditional="auto",
                       n_mu=121, n_ls=33,
                       mu_lim=12.0, ls_lim=(0.2, 5.0)):
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

    Everything comes from the SAME posterior weights, so the summaries are
    mutually consistent (activity == bin_prob @ [1..B], q2 == mu).  Returns a
    dict of (N,) arrays, NaN off ``res.observed``:

      activity, activity_sd   posterior mean and SD of E[bin]
      activity_map            the plug-in mode, for reference
      mu, sigma               posterior-mean effect location and scale
      mu_sd, sigma_sd         their posterior SDs (standard errors)
      q1, q2, q3              quartiles of the posterior-mean effect law
      abundance               posterior-mean profiled size factor a_n
      n_eff                   posterior grid support (larger = more diffuse)
      tot                     observed reads
      observed                (N,) bool

    plus bin_prob, the (N, B) posterior-mean distribution over bins.
    """
    cfg = res.config
    X = np.asarray(X)
    N, S, B = X.shape
    bvec = np.arange(1, B + 1, dtype=float)
    lambda_fix = cfg["lambda_fix"]
    # profile the abundance up to a_max whenever the fit modelled one, whether
    # as a free a_n or integrated out under the Gamma prior
    a_max = (cfg["a_max"]
             if (cfg["abundance"] or cfg.get("abundance_prior")) else 1.0)
    rate_max = lambda_fix * a_max
    if tau_within == "auto":
        tau_within = cfg.get("mu_prior")
    if isinstance(sigma_prior, str):
        sigma_prior = cfg.get("sigma_prior")
    if conditional == "auto":
        conditional = cfg.get("conditional", True)

    cuts = jnp.asarray(res.cuts)
    R, P = jnp.asarray(res.R), jnp.asarray(res.P)
    rates, log_w = jnp.asarray(res.rates), jnp.asarray(res.log_w)
    lam = np.asarray(res.rates)[:, 0]                          # (S,) K = 1
    c = np.asarray(res.R) * (1 - np.asarray(res.P)) / np.asarray(res.P)

    builder = LoglikBuilder(X, mask=mask, rate_max=rate_max)
    kept = np.asarray(res.observed)
    Xz = np.where(np.asarray(builder.mask), np.nan_to_num(X, nan=0.0), 0.0)
    total = Xz.sum((1, 2))                                     # observed reads

    # shared (mu, log sigma) grid, its activity readout and expected total
    mu_g = np.linspace(-mu_lim, mu_lim, n_mu)
    ls_g = np.linspace(np.log(ls_lim[0]), np.log(ls_lim[1]), n_ls)
    MU, LS = np.meshgrid(mu_g, ls_g, indexing="ij")
    MU, LS = MU.ravel(), LS.ravel()
    G = MU.size
    Pi_g = np.asarray(normal_bin_probs(jnp.asarray(MU), jnp.exp(jnp.asarray(LS)),
                                       cuts))
    ebin_g = Pi_g @ bvec                                        # (G,)
    denom_g = (c[None] * Pi_g[:, None, :]).sum(2) @ lam         # E[tot | a = 1]

    @jax.jit
    def ll_grid(pi_row, a_col):
        M = a_col[:, None] * pi_row[None, :]
        obj = builder.object_loglik(R, P, M, rates, log_w, 0.0)
        if not conditional:
            return obj
        lz = jnp.minimum(log_zero_prob(R, P, M, rates, log_w, builder.mask),
                         -1e-12)
        return obj - log1mexp(lz)

    ll = np.empty((N, G))
    a_grid = np.empty((N, G))
    for j in range(G):
        a_col = np.clip(total / max(denom_g[j], 1e-12), 1e-4, a_max)
        a_grid[:, j] = a_col
        ll[:, j] = np.asarray(ll_grid(jnp.asarray(Pi_g[j]), jnp.asarray(a_col)))

    # per-object prior over the grid: N(mu; center_n, tau^2) * N(ls; m_s, s_s)
    logprior = np.zeros((N, G))
    if tau_within is not None:
        center = np.broadcast_to(np.asarray(mu_center, float), (N,))
        logprior = logprior - (MU[None, :] - center[:, None]) ** 2 \
            / (2.0 * tau_within ** 2)
    if sigma_prior is not None:
        m_s, s_s = sigma_prior
        logprior = logprior - ((LS - m_s) ** 2 / (2.0 * s_s ** 2))[None, :]

    logpost = ll + logprior
    logpost -= logpost.max(1, keepdims=True)
    w = np.exp(logpost)
    w /= w.sum(1, keepdims=True)
    ebin_pm = w @ ebin_g
    ebin_sd = np.sqrt(np.clip(w @ (ebin_g ** 2) - ebin_pm ** 2, 0, None))

    sig_g = np.exp(LS)
    mu_pm = w @ MU
    sigma_pm = w @ sig_g
    z25 = 0.6744897501960817                           # N(0,1) third quartile
    return dict(
        activity=np.where(kept, ebin_pm, np.nan),
        activity_sd=np.where(kept, ebin_sd, np.nan),
        activity_map=np.where(kept, np.asarray(res.Pi) @ bvec, np.nan),
        mu=np.where(kept, mu_pm, np.nan),
        mu_sd=np.where(kept, np.sqrt(np.clip(w @ (MU ** 2) - mu_pm ** 2,
                                             0, None)), np.nan),
        sigma=np.where(kept, sigma_pm, np.nan),
        sigma_sd=np.where(kept, np.sqrt(np.clip(w @ (sig_g ** 2)
                                                - sigma_pm ** 2, 0, None)),
                          np.nan),
        q1=np.where(kept, w @ (MU - z25 * sig_g), np.nan),
        q2=np.where(kept, mu_pm, np.nan),
        q3=np.where(kept, w @ (MU + z25 * sig_g), np.nan),
        abundance=np.where(kept, (w * a_grid).sum(1), np.nan),
        n_eff=np.where(kept, 1.0 / (w ** 2).sum(1), np.nan),
        tot=np.where(kept, total, np.nan),
        bin_prob=np.where(kept[:, None], w @ Pi_g, np.nan),
        observed=kept)
