"""Batch-effect control: the per-sample bin x covariate tilt.

A sorted MPRA sample is not one library.  Each of the B bins is amplified and
sequenced as its OWN library, so any composition-dependent efficiency (PCR GC
bias being the obvious one) acts on the bins separately, and its contrast
ACROSS bins tilts every object's apparent profile:

    E x[n,s,b]  =  c[s,b] * a_n * Pi[n,b] * exp(g[s,b] . u_n)

with ``u_n`` a sequence covariate and ``g[s,b]`` the tilt carried by sample
``s``'s bin-``b`` library.  Only the contrast over b is identified -- a factor
common to all bins is an abundance and ``a_n`` eats it -- so ``g`` is centred
over bins throughout.

Within ONE sample a monotone tilt is indistinguishable from an activity shift:
"GC-rich sequences are less active here" and "GC-rich sequences lost reads in
the high bins here" predict the same counts.  What separates them is
REPLICATION, which shares the biology and redraws the technical part, so the
tilt is modelled hierarchically:

    g_obs(line, s) = g_bio(line) + g_tech(line, s),   g_tech ~ (0, sigma_t^2)

``sigma_t`` comes from the within-line differences, where biology cancels;
``variance_components`` splits the between-line spread and ``shrink_weight``
gives the fraction to remove.  The panel MEAN is deliberately left alone: a
tilt shared by every line is as consistent with covariate-coupled biology as
with a protocol-wide bias, and no replication separates them.

Applying the correction is a division on the profile,

    Pi_clean[n,b]  proportional to  Pi[n,b] * exp(-delta[b] . u_n),

a genuine model term rather than a post-hoc edit of the activity scalar: it
moves the whole B-vector and the activity follows.  ``fit_free_pi`` and
``fit_effects`` take the same factor as ``tilt=``, which is what makes the
correction identified when S > 1.

This module is the ESTIMATOR.  Deciding which samples to correct -- and whether
to correct rather than drop them -- is ``qc``; see ``docs/diagnostics.md``.
"""

import numpy as np
from scipy.optimize import minimize

__all__ = ["poly_basis", "sample_tilt", "variance_components", "shrink_weight",
           "apply_tilt", "tilt_factor", "tilt_axes", "strand_basis",
           "anti_control_delta", "broadcast_tilt"]


def broadcast_tilt(tilt, N, S, B):
    """Bring a tilt of shape (N, S, B), (S, B) or (N, B) to (N, S, B).

    (N, B) needs an explicit replicate axis: numpy aligns trailing dimensions,
    so broadcasting it against (N, S, B) would try to match N to S.  This is
    the shape ``qc.tilt_offsets`` returns -- one factor per object, shared by
    the line's replicates.  If N == S the two 2-D forms are indistinguishable
    and (S, B) is assumed; pass (N, S, B) explicitly in that case.
    """
    t = np.asarray(tilt, float)
    if t.ndim == 2 and t.shape == (N, B) and (N, B) != (S, B):
        t = t[:, None, :]
    return np.broadcast_to(t, (N, S, B))


def poly_basis(x, degree=2, weights=None):
    """(N, degree) orthonormal polynomial basis in ``x``, no constant column.

    Orthonormalised on the empirical distribution of ``x`` so the coefficients
    of different degrees are (nearly) uncorrelated and can be shrunk term by
    term.
    """
    x = np.asarray(x, float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    V = np.stack([x ** k for k in range(degree + 1)], 1)
    # weighted Gram-Schmidt
    out = []
    for k in range(degree + 1):
        v = V[:, k].copy()
        for u in out:
            v -= (w * v * u).sum() * u
        nrm = np.sqrt((w * v * v).sum())
        out.append(v / max(nrm, 1e-12))
    return np.stack(out[1:], 1)          # drop the constant


def strand_basis(seqs, degree=2):
    """``(U, n_sym)`` -- a covariate basis split into reverse-complement
    SYMMETRIC and ANTISYMMETRIC blocks.

    The plasmid is double stranded and both strands amplify together, so an
    amplification efficiency cannot tell a G from a C: it is a function of the
    unordered base pair, hence invariant under reverse complement, hence
    entirely inside the symmetric block.  The antisymmetric block -- GC skew
    ``f_G - f_C`` and AT skew ``f_A - f_T``, both sign-flipping under reverse
    complement -- is unreachable by dsDNA amplification and can only be written
    by a single-stranded step or by genuine biology.

    That makes the antisymmetric block a NEGATIVE CONTROL: correlated with a
    line's composition-coupled biology, causally excluded from its
    amplification artifact.  ``anti_control_delta`` uses exactly that.

    Columns ``[:n_sym]`` are symmetric (a degree-``degree`` polynomial in GC),
    columns ``[n_sym:]`` antisymmetric.  The blocks are orthonormalised
    SEPARATELY: mixing them would rotate symmetric signal into the
    antisymmetric coordinates and destroy the exclusion restriction.
    """
    seqs = np.asarray(seqs)
    A = np.array([list(s) for s in seqs])
    f = {nt: (A == nt).mean(1) for nt in "ACGT"}
    U_sym = poly_basis(f["G"] + f["C"], degree=degree)
    raw = np.stack([f["G"] - f["C"], f["A"] - f["T"]], 1)
    # a constant offset is harmless: it enters the tilt as a per-bin constant
    # and is absorbed by the multinomial's alpha[b]
    raw = raw - raw.mean(0, keepdims=True)
    w, V = np.linalg.eigh(np.cov(raw, rowvar=False))
    U_anti = raw @ (V / np.sqrt(np.maximum(w, 1e-30)))
    return np.concatenate([U_sym, U_anti], 1), U_sym.shape[1]


def anti_control_delta(G_obs, lab, n_sym, rank_sym=1, rank_anti=2):
    """Amplification tilt identified against the antisymmetric negative control.

    An alternative to ``variance_components``' replicate-pair split, using a
    within-sample control every sample provides rather than the handful of
    replicate pairs.  For each sample ``s``,

        A_sym(s)  =  gamma . A_anti(s)  +  tau(s)

    where ``A_sym``/``A_anti`` are the leading tilt-axis amplitudes of the two
    blocks.  ``gamma`` carries composition-coupled BIOLOGY from its
    antisymmetric expression into its symmetric one; amplification bias can
    only live in the residual ``tau``, and ``tau`` alone is corrected.
    ``gamma`` is fit LEAVE-ONE-SAMPLE-OUT so an outlying sample cannot shrink
    its own residual toward zero.

    Returns ``(delta, info)``: ``delta`` maps line -> (B, J) tilt to divide out
    (nonzero only in the symmetric columns), ``info`` the per-sample pieces.
    """
    G = np.asarray(G_obs, float)
    lab = np.asarray(lab)
    n_s, B, J = G.shape
    Gs, Ga = G[:, :, :n_sym], G[:, :, n_sym:]

    ax_s, A_s, mean_s, _ = tilt_axes(Gs, rank=rank_sym)
    _, A_a, _, _ = tilt_axes(Ga, rank=min(rank_anti, Ga.shape[1] * B))

    y = A_s[:, 0]
    X = np.c_[np.ones(n_s), A_a]
    pred = np.empty(n_s)
    for i in range(n_s):                       # leave-one-out gamma
        m = np.ones(n_s, bool); m[i] = False
        beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        pred[i] = X[i] @ beta
    tau = y - pred
    sd = tau.std(ddof=X.shape[1])

    delta = {}
    for l in dict.fromkeys(lab.tolist()):
        t = float(tau[lab == l].mean())
        d = np.zeros((B, J))
        d[:, :n_sym] = t * ax_s[0]             # symmetric columns only
        delta[l] = d
    return delta, dict(a_sym=y, a_anti=A_a, pred=pred, tau=tau, sd=float(sd),
                       axis_sym=ax_s[0], lab=lab)


def _multinomial_nll(par, U, Y, off, B, J):
    """-loglik of  q[n,b] ~ off[n,b] * exp(alpha[b] + g[b].u_n),  bin 0 = ref."""
    a = np.concatenate([[0.0], par[:B - 1]])
    G = np.vstack([np.zeros(J), par[B - 1:].reshape(B - 1, J)])
    eta = off + a[None, :] + U @ G.T                    # (N, B)
    m = eta.max(1, keepdims=True)
    e = np.exp(eta - m)
    Z = e.sum(1, keepdims=True)
    logq = eta - m - np.log(Z)
    nll = -(Y * logq).sum()
    q = e / Z
    Rres = Y - q * Y.sum(1, keepdims=True)              # (N, B) score residual
    ga = -Rres[:, 1:].sum(0)
    gG = -(U.T @ Rres[:, 1:]).T.reshape(-1)
    return nll, np.concatenate([ga, gG])


def sample_tilt(X, U, offset=None, min_reads=20, ridge=1e-6):
    """Fit one sample's bin x covariate tilt ``g`` (B, J), centred over bins.

    ``X`` is that sample's (N, B) counts, ``U`` the (N, J) covariate basis.
    ``offset`` is an optional (N, B) log-offset -- pass ``log(Pi)`` to measure
    the tilt of the counts RELATIVE to a fitted profile, or leave it None to
    measure the sample's total covariate response (biology included), which is
    what the hierarchical split is applied to.

    Conditioning on each object's total removes the abundance exactly, so this
    is a plain multinomial regression and needs no depth model.
    """
    X = np.asarray(X, float)
    N, B = X.shape
    U = np.asarray(U, float).reshape(N, -1)
    J = U.shape[1]
    tot = X.sum(1)
    keep = tot >= min_reads
    Y, Uk = X[keep], U[keep]
    off = np.zeros_like(Y) if offset is None else np.asarray(offset, float)[keep]
    # bin-library sizes are a per-bin constant and are absorbed by alpha[b],
    # so no separate normalisation is needed
    p0 = np.zeros((B - 1) * (1 + J))

    def f(par):
        nll, g = _multinomial_nll(par, Uk, Y, off, B, J)
        return nll + 0.5 * ridge * (par @ par), g + ridge * par

    res = minimize(f, p0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=500, ftol=1e-12, gtol=1e-10))
    G = np.vstack([np.zeros(J), res.x[B - 1:].reshape(B - 1, J)])
    return G - G.mean(0, keepdims=True), int(keep.sum()), res


def variance_components(G_obs, line, n_rep):
    """Split observed per-sample tilts into biological and technical variance.

    ``G_obs``   (n_samples, B, J) per-sample tilts
    ``line``    (n_samples,) line label of each sample
    ``n_rep``   dict line -> number of samples of that line

    Within-line pairs share the biology exactly, so their difference is
    ``tech_a - tech_b`` and  E[d^2] = 2 sigma_t^2.  The between-line spread of
    the line means is  sigma_bio^2 + sigma_t^2 * mean(1/S).

    Returns ``(sigma_t2, sigma_bio2, G_line, G_panel)`` with the variances
    per (B, J) cell.
    """
    G_obs = np.asarray(G_obs, float)
    line = np.asarray(line)
    lines = list(dict.fromkeys(line.tolist()))
    G_line = np.stack([G_obs[line == l].mean(0) for l in lines])   # (L, B, J)

    d2, npair = np.zeros(G_obs.shape[1:]), 0
    for l in lines:
        idx = np.where(line == l)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                d2 += (G_obs[idx[i]] - G_obs[idx[j]]) ** 2
                npair += 1
    if npair == 0:
        raise ValueError("no replicated line: sigma_t is not identified")
    sigma_t2 = d2 / npair / 2.0

    S = np.array([n_rep[l] for l in lines], float)
    var_obs = G_line.var(0, ddof=1)
    sigma_bio2 = np.maximum(var_obs - sigma_t2 * np.mean(1.0 / S), 0.0)
    return sigma_t2, sigma_bio2, dict(zip(lines, G_line)), G_line.mean(0)


def tilt_axes(G_obs, rank=1):
    """Low-rank basis of the per-sample tilts: ``(axes, amplitudes, mean, share)``.

    The observed tilts are not B*J independent numbers per sample: a
    composition-dependent efficiency acting on separately amplified bin
    libraries should look like one shape with a per-sample amplitude, and in
    practice one axis carries most of the between-sample variance.  Reducing to
    that amplitude puts every replicate pair behind ONE number instead of
    splitting them over B*J noise-dominated cells.

    The PANEL MEAN is removed before the decomposition and is never corrected.

    Caveat for anything downstream: the axis is the leading component of
    BETWEEN-sample variance, so it is aimed at the most anomalous samples and
    can be nearly orthogonal to an ordinary sample's technical deviation.  It
    is a good detector, not a general-purpose correction direction.
    """
    G = np.asarray(G_obs, float)
    n = G.shape[0]
    flat = G.reshape(n, -1)
    mean = flat.mean(0)
    _, s, vt = np.linalg.svd(flat - mean, full_matrices=False)
    axes = vt[:rank]                                   # (rank, B*J)
    amp = (flat - mean) @ axes.T                       # (n, rank)
    share = s ** 2 / (s ** 2).sum()
    return axes.reshape(rank, *G.shape[1:]), amp, mean.reshape(G.shape[1:]), share


def shrink_weight(sigma_t2, sigma_bio2, S):
    """James-Stein weight kept on a line's observed tilt, per (B, J) cell."""
    return sigma_bio2 / np.maximum(sigma_bio2 + sigma_t2 / float(S), 1e-30)


def tilt_factor(U, G):
    """(N, B) multiplicative factor exp(G . u), normalised to geometric mean 1
    over bins so it carries no abundance."""
    E = np.asarray(U, float) @ np.asarray(G, float).T          # (N, B)
    return np.exp(E - E.mean(1, keepdims=True))


def apply_tilt(Pi, U, G, sign=-1.0):
    """Divide (``sign=-1``) or multiply the fitted profile by the tilt.

    ``Pi`` (N, B) rows are renormalised, so this is a move inside the simplex:
    the object keeps its abundance and only its bin SHAPE changes.
    """
    Q = np.asarray(Pi, float) * tilt_factor(U, np.asarray(G) * sign)
    return Q / Q.sum(1, keepdims=True)
