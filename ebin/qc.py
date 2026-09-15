"""Library QC: which sequencing libraries are defective, and what to do.

Each sorting bin is amplified and sequenced as its own library, so a
composition-dependent efficiency (PCR GC bias) acts on the bins separately and
tilts every object's apparent profile.  ``batch.sample_tilt`` measures that
tilt per library; this module decides which of them are anomalous.

The panel's per-library tilts are not B x J independent numbers -- one axis
carries most of the between-library variance -- so each library is reduced to
its amplitude ``a_s`` on that axis and modelled as

    a_s = mu + b_line + t_s + e_s,   b_line ~ N(0, sigma_bio^2),  e_s ~ N(0, v_s)

with ``v_s`` the exact multinomial Fisher variance of the amplitude.  Within-line
replicate differences identify the technical term, because biology cancels in
them exactly.

Two models for ``t_s``, and they disagree about what to do:

``gaussian``  t_s ~ N(0, sigma_tech^2)  -- every library loses the same fraction
              of its deviation.
``mixture``   t_s ~ (1-pi) N(0, sigma_ok^2) + pi N(0, sigma_bad^2) -- a library
              either behaves or fails.  The E-step is exact (S <= 2 means at
              most four assignments per line), not a mean-field iteration.

Prefer ``mixture``.  On the panel this was developed against, the within-line
differences are six small numbers and one at z = -10.9; a variance is the wrong
summary of that, and the Gaussian's own leave-one-out puts sigma_tech outside
the interval it reports.  ``removal_interval`` bootstraps the fraction removed
and it is uninformative ([1.3%, 97.6%] over lines), so treat the scan as a
DETECTOR -- it says *which* libraries are bad, not *how much* to correct.  The
supported responses, in order of evidence: drop the flagged libraries
(``drop_defective``), or correct them alone (``tilt_offsets``).  Correcting the
whole panel is not supported: the panel-mean tilt is as consistent with
GC-dependent biology as with a protocol bias and is never touched.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .batch import sample_tilt, strand_basis, tilt_axes, tilt_factor
from .data import bin_order, read_counts

__all__ = ["LibraryScan", "scan_libraries", "library_tilts", "fit_reml",
           "fit_failure_mixture", "tilt_offsets", "drop_defective",
           "removal_interval", "DEFECT_THRESHOLD"]

DEFECT_THRESHOLD = 0.3

# V is exactly singular when both components sit at zero and the measurement
# variances are ~1e-7; a jitter far below anything the data resolves keeps the
# optimizer feasible.
_JIT = 1e-12


# --------------------------------------------------------------------------
# per-library tilt amplitudes
# --------------------------------------------------------------------------
def library_tilts(counts, sequences=None, *, degree=2, min_reads=20,
                  groups=None, verbose=False):
    """Measure every library's tilt and reduce it to one amplitude.

    Returns ``(frame, info)``.  ``frame`` has one row per (cell line,
    replicate) with ``alpha`` (amplitude on the leading symmetric axis),
    ``var`` (its multinomial Fisher variance) and ``a_anti`` (the amplitude on
    the reverse-complement ANTISYMMETRIC axis, which dsDNA amplification cannot
    reach -- carried as a negative control, never corrected).  ``info`` holds
    the axis, the covariate basis and the raw (n_lib, B, J) tilts.
    """
    df = counts if isinstance(counts, pd.DataFrame) else read_counts(counts)
    df = df.fillna(0.0)
    seqs = df.index.to_numpy() if sequences is None else np.asarray(sequences)
    U, n_sym = strand_basis(seqs, degree=degree)

    libs = [c[:2] for c in df.columns]
    libs = [l for l in dict.fromkeys(libs) if groups is None or l[0] in groups]
    if not libs:
        raise ValueError("no libraries selected")

    G, raw = [], []
    for line, rep in libs:
        sub = df[line][rep]
        b_names = list(dict.fromkeys(sub.columns))
        X = sub[[b_names[i] for i in bin_order(b_names)]].to_numpy(float)
        g, n_kept, res = sample_tilt(X, U, min_reads=min_reads)
        G.append(g)
        raw.append((X, res.x))
        if verbose:
            print(f"  [tilt] {line:<24}{rep:<12} {n_kept:6d} objects")
    G = np.stack(G)

    axis, A_sym, _, share = tilt_axes(G[:, :, :n_sym], rank=1)
    _, A_anti, _, _ = tilt_axes(G[:, :, n_sym:], rank=1)
    var = _amplitude_variance(raw, U, axis[0], n_sym, min_reads)

    frame = pd.DataFrame(dict(line=[l for l, _ in libs],
                              replicate=[r for _, r in libs],
                              alpha=A_sym[:, 0], var=var,
                              se=np.sqrt(var), a_anti=A_anti[:, 0]))
    return frame, dict(axis=axis[0], basis=U, n_sym=n_sym, tilts=G,
                       share=float(share[0]))


def _amplitude_variance(raw, U, axis, n_sym, min_reads):
    """Var(alpha) = w' I^-1 w, I the multinomial Fisher information of the
    per-library regression and w the linear functional reading the amplitude
    off the centred tilt (the axis is already bin-centred, so the centring map
    drops out)."""
    N, B = raw[0][0].shape
    J = U.shape[1]
    ncoef = (B - 1) * (1 + J)
    w = np.zeros(ncoef)
    for b in range(1, B):
        w[(b - 1) * (1 + J) + 1: (b - 1) * (1 + J) + 1 + n_sym] = axis[b, :n_sym]
    Z = np.c_[np.ones(N), U]

    out = []
    for X, par in raw:
        keep = X.sum(1) >= min_reads
        a = np.concatenate([[0.0], par[:B - 1]])
        Gr = np.vstack([np.zeros(J), par[B - 1:].reshape(B - 1, J)])
        eta = a[None, :] + U @ Gr.T
        e = np.exp(eta - eta.max(1, keepdims=True))
        q = (e / e.sum(1, keepdims=True))[keep]
        Zk, Tk = Z[keep], X.sum(1)[keep]
        ZZ = Zk[:, :, None] * Zk[:, None, :]
        I = np.zeros((ncoef, ncoef))
        for b in range(1, B):
            for c in range(1, B):
                wbc = Tk * ((q[:, b] if b == c else 0.0) - q[:, b] * q[:, c])
                sl_b = slice((b - 1) * (1 + J), (b - 1) * (1 + J) + 1 + J)
                sl_c = slice((c - 1) * (1 + J), (c - 1) * (1 + J) + 1 + J)
                I[sl_b, sl_c] = np.einsum("n,nij->ij", wbc, ZZ)
        out.append(float(w @ np.linalg.pinv(I) @ w))
    return np.array(out)


# --------------------------------------------------------------------------
# the two policies
# --------------------------------------------------------------------------
def _blocks(line):
    idx = {}
    for i, l in enumerate(line):
        idx.setdefault(l, []).append(i)
    return [np.array(v) for v in idx.values()], list(idx)


def _profile_mu(y, v, blocks, s2b, s2t, w):
    num = den = 0.0
    Vinvs = []
    for bl in blocks:
        n = len(bl)
        V = s2b * np.ones((n, n)) + np.diag(s2t / w[bl] + v[bl] + _JIT)
        Vi = np.linalg.inv(V)
        Vinvs.append(Vi)
        num += np.ones(n) @ Vi @ y[bl]
        den += np.ones(n) @ Vi @ np.ones(n)
    return num / den, den, Vinvs


def _neg_reml(par, y, v, blocks, restricted, w):
    s2b, s2t = float(par[0]) ** 2, float(par[1]) ** 2
    mu, den, Vinvs = _profile_mu(y, v, blocks, s2b, s2t, w)
    ll = 0.0
    for bl, Vi in zip(blocks, Vinvs):
        V = s2b * np.ones((len(bl),) * 2) + np.diag(s2t / w[bl] + v[bl] + _JIT)
        r = y[bl] - mu
        ll -= 0.5 * (np.linalg.slogdet(V)[1] + r @ Vi @ r)
    return -(ll - 0.5 * np.log(den) if restricted else ll)


def fit_reml(y, v, line, *, restricted=True, fix_tech=None, w=None):
    """REML for (sigma_bio, sigma_tech), with the BLUP of the sample term.

    ``v`` is the known measurement variance of each ``y``.  At S = 1 the BLUP
    is the James-Stein shrinkage of that line's deviation from the panel mean.
    """
    y, v = np.asarray(y, float), np.asarray(v, float)
    blocks, names = _blocks(np.asarray(line))
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    best = None
    for s0 in ((0.10, 0.10), (0.02, 0.20), (0.20, 0.02), (0.15, 0.05)):
        if fix_tech is not None:
            r = minimize(lambda p: _neg_reml([p[0], fix_tech], y, v, blocks,
                                             restricted, w),
                         [s0[0]], method="L-BFGS-B", bounds=[(0.0, 3.0)])
            val, par = r.fun, np.array([r.x[0], fix_tech])
        else:
            r = minimize(_neg_reml, np.array(s0), method="L-BFGS-B",
                         args=(y, v, blocks, restricted, w),
                         bounds=[(0.0, 3.0)] * 2)
            val, par = r.fun, r.x
        if best is None or val < best[0]:
            best = (val, par)
    val, par = best
    s2b, s2t = par[0] ** 2, par[1] ** 2
    mu, _, Vinvs = _profile_mu(y, v, blocks, s2b, s2t, w)
    t_hat, t_var = np.zeros_like(y), np.zeros_like(y)
    for bl, Vi in zip(blocks, Vinvs):
        d = s2t / w[bl]
        t_hat[bl] = d * (Vi @ (y[bl] - mu))
        t_var[bl] = np.maximum(d - d * np.diag(Vi) * d, 0.0)
    return dict(sigma_bio=abs(par[0]), sigma_tech=abs(par[1]), mu=mu, nll=val,
                t_hat=t_hat, t_var=t_var, blocks=blocks, names=names)


def _block_terms(r, v, s2_assign, s2b):
    """One line's block under a fixed assignment: log-density plus the first
    AND second posterior moments.  The variances are not optional -- an EM that
    updates a component from squared BLUPs alone collapses it to zero."""
    n = len(r)
    Sig = s2b * np.ones((n, n)) + np.diag(s2_assign + v)
    Si = np.linalg.inv(Sig)
    ll = -0.5 * (np.linalg.slogdet(Sig)[1] + r @ Si @ r + n * np.log(2 * np.pi))
    t = s2_assign * (Si @ r)
    vt = np.maximum(s2_assign - s2_assign ** 2 * np.diag(Si), 0.0)
    one = np.ones(n)
    b = float(s2b * one @ (Si @ r))
    vb = max(s2b - s2b ** 2 * float(one @ Si @ one), 0.0)
    return ll, t, vt, b, vb


def fit_failure_mixture(y, v, line, *, iters=600, tol=1e-12, pi_floor=1e-3):
    """Exact-E-step EM for (pi, sigma_ok, sigma_bad, sigma_bio, mu).

    Returns those, plus ``p_failed`` per library and the posterior sample term
    ``t_hat``.  For a replicated line the assignment is informed by the
    replicates' disagreement; for a singleton only by its distance from the
    panel, which is weak -- read such a flag as "this deviation is not a draw
    from the other lines' distribution", not as a diagnosis.
    """
    y, v, line = np.asarray(y, float), np.asarray(v, float), np.asarray(line)
    lines = list(dict.fromkeys(line.tolist()))
    idx = {l: np.where(line == l)[0] for l in lines}
    pi, s2_ok, s2_bad, s2b, mu = 0.15, 0.03 ** 2, 0.30 ** 2, 0.12 ** 2, y.mean()

    for _ in range(iters):
        post_t = np.zeros_like(y)
        r_bad = np.zeros_like(y)
        # component -> [sum of weights, sum of weight * E t^2]
        acc = {0: [0.0, 0.0], 1: [0.0, 0.0]}
        eb2 = num_mu = den_mu = 0.0
        for l in lines:
            ii = idx[l]
            n = len(ii)
            r = y[ii] - mu
            recs, lls = [], []
            for tag in itertools.product((0, 1), repeat=n):
                s2a = np.array([s2_bad if t else s2_ok for t in tag])
                ll, t, vt, b, vb = _block_terms(r, v[ii], s2a, s2b)
                ll += sum(np.log(pi) if k else np.log(1 - pi) for k in tag)
                lls.append(ll)
                recs.append((np.array(tag), t, vt, b, vb))
            lls = np.array(lls)
            w = np.exp(lls - lls.max())
            w /= w.sum()
            for wi, (tag, t, vt, b, vb) in zip(w, recs):
                post_t[ii] += wi * t
                r_bad[ii] += wi * tag
                eb2 += wi * (b ** 2 + vb)
                for j, k in enumerate(tag):
                    acc[int(k)][0] += wi
                    acc[int(k)][1] += wi * (t[j] ** 2 + vt[j])
            s2a = r_bad[ii] * s2_bad + (1 - r_bad[ii]) * s2_ok
            Si = np.linalg.inv(s2b * np.ones((n, n)) + np.diag(s2a + v[ii]))
            num_mu += float(np.ones(n) @ Si @ y[ii])
            den_mu += float(np.ones(n) @ Si @ np.ones(n))

        new = (num_mu / den_mu, float(np.clip(r_bad.mean(), pi_floor, 0.5)),
               max(acc[0][1] / max(acc[0][0], 1e-9), 1e-10),
               max(acc[1][1] / max(acc[1][0], 1e-9), 1e-10),
               max(eb2 / len(lines), 1e-10))
        step = max(abs(a - b) for a, b in
                   zip(new, (mu, pi, s2_ok, s2_bad, s2b)))
        mu, pi, s2_ok, s2_bad, s2b = new
        if step < tol:
            break

    return dict(mu=mu, pi=pi, sigma_ok=np.sqrt(s2_ok),
                sigma_bad=np.sqrt(s2_bad), sigma_bio=np.sqrt(s2b),
                t_hat=post_t, p_failed=r_bad, lines=lines)


def _removed(sigma_bio, sigma_t):
    """Fraction of a singleton line's deviation a policy removes."""
    return sigma_t ** 2 / (sigma_bio ** 2 + sigma_t ** 2)


# --------------------------------------------------------------------------
# the scan
# --------------------------------------------------------------------------
@dataclass
class LibraryScan:
    """Per-library tilt amplitudes with both policies fitted.

    ``table`` is the per-library frame (``line``, ``replicate``, ``alpha``,
    ``se``, ``p_defective``, ``t_mixture``, ``t_gaussian``); ``axis`` the
    leading tilt direction; ``basis`` the (N, J) covariate basis the tilt is
    expressed in.
    """
    table: pd.DataFrame
    axis: np.ndarray
    basis: np.ndarray
    n_sym: int
    tilts: np.ndarray
    share: float
    mixture: dict
    gaussian: dict
    threshold: float = DEFECT_THRESHOLD

    @property
    def defective(self):
        """Libraries over the threshold, as ``(line, replicate)`` pairs."""
        t = self.table
        return [tuple(r) for r in
                t.loc[t.p_defective > self.threshold,
                      ["line", "replicate"]].to_numpy()]

    @property
    def defective_lines(self):
        return list(dict.fromkeys(l for l, _ in self.defective))

    def summary(self):
        m, g = self.mixture, self.gaussian
        rows = self.table.sort_values("p_defective", ascending=False)
        return "\n".join([
            f"{len(self.table)} libraries, {self.table.line.nunique()} cell "
            f"lines; leading axis = {self.share:.1%} of tilt variance",
            f"  gaussian  sigma_bio {g['sigma_bio']:.4f}  sigma_tech "
            f"{g['sigma_tech']:.4f}  -> every library loses "
            f"{_removed(g['sigma_bio'], g['sigma_tech']):.1%}",
            f"  mixture   sigma_bio {m['sigma_bio']:.4f}  sigma_ok "
            f"{m['sigma_ok']:.4f}  sigma_bad {m['sigma_bad']:.4f}  "
            f"pi {m['pi']:.3f}",
            f"            -> clean {_removed(m['sigma_bio'], m['sigma_ok']):.1%}"
            f", defective {_removed(m['sigma_bio'], m['sigma_bad']):.1%}",
            f"  measurement se: median {self.table.se.median():.2}, max "
            f"{self.table.se.max():.2} vs between-library sd "
            f"{self.table.alpha.std():.3f}",
            f"  flagged at P > {self.threshold}: "
            + (", ".join(f"{l} {r}" for l, r in self.defective) or "none"),
            rows.head(6).to_string(index=False,
                                   float_format=lambda x: f"{x:9.4f}")])


def scan_libraries(counts, sequences=None, *, degree=2, min_reads=20,
                   groups=None, threshold=DEFECT_THRESHOLD, verbose=False):
    """Flag defective sequencing libraries from their bin x covariate tilt.

        scan = scan_libraries("counts.csv")
        print(scan.summary())
        scan.defective                  # [('Jurkat', '052323/2'), ...]

    Needs at least one replicated cell line: the technical component is
    identified by within-line differences and nothing else.
    """
    frame, info = library_tilts(counts, sequences, degree=degree,
                                min_reads=min_reads, groups=groups,
                                verbose=verbose)
    y, v, line = (frame.alpha.to_numpy(), frame["var"].to_numpy(),
                  frame.line.to_numpy())
    if frame.groupby("line").size().max() < 2:
        raise ValueError("no replicated cell line: the technical variance is "
                         "not identified, so no library can be called "
                         "defective")
    G = fit_reml(y, v, line)
    M = fit_failure_mixture(y, v, line)

    frame = frame.assign(deviation=y - M["mu"], p_defective=M["p_failed"],
                         t_mixture=M["t_hat"], t_gaussian=G["t_hat"])
    scan = LibraryScan(table=frame, axis=info["axis"], basis=info["basis"],
                       n_sym=info["n_sym"], tilts=info["tilts"],
                       share=info["share"], mixture=M, gaussian=G,
                       threshold=threshold)
    if verbose:
        print(scan.summary())
    return scan


def removal_interval(scan, n=300, seed=0):
    """Bootstrap over CELL LINES of the fraction each policy removes.

    Lines, not libraries: the line is the unit the design replicates, and
    resampling libraries would break the replicate pairs that carry the whole
    identification.  Expect a wide, possibly bimodal interval -- that width is
    the honest uncertainty on the correction strength, and it is why the scan
    should be used to select libraries rather than to scale a correction.
    """
    rng = np.random.default_rng(seed)
    t = scan.table
    y, v, line = (t.alpha.to_numpy(), t["var"].to_numpy(), t.line.to_numpy())
    lines = list(dict.fromkeys(line.tolist()))
    idx = {l: np.where(line == l)[0] for l in lines}
    out = []
    for _ in range(n):
        yy, vv, ll = [], [], []
        for k, j in enumerate(rng.choice(len(lines), len(lines), replace=True)):
            ii = idx[lines[j]]
            yy.append(y[ii])
            vv.append(v[ii])
            # relabel: a line may be drawn more than once
            ll += [f"L{k}"] * len(ii)
        ll = np.array(ll)
        if not (pd.Series(ll).value_counts() > 1).any():
            continue                            # no replicated line drawn
        yy, vv = np.concatenate(yy), np.concatenate(vv)
        try:
            G = fit_reml(yy, vv, ll)
            M = fit_failure_mixture(yy, vv, ll, iters=200)
        except np.linalg.LinAlgError:
            continue
        out.append((G["sigma_tech"], _removed(G["sigma_bio"], G["sigma_tech"]),
                    M["sigma_ok"], M["sigma_bad"], M["pi"],
                    _removed(M["sigma_bio"], M["sigma_ok"]),
                    _removed(M["sigma_bio"], M["sigma_bad"])))
    return pd.DataFrame(out, columns=["sigma_tech", "gaussian_removed",
                                      "sigma_ok", "sigma_bad", "pi",
                                      "mixture_removed_clean",
                                      "mixture_removed_defective"])


# --------------------------------------------------------------------------
# acting on the scan
# --------------------------------------------------------------------------
def tilt_offsets(scan, lines, policy="mixture", scale=1.0):
    """``{cell line: (N, B) multiplicative factor}`` ready for ``tilt=``.

        scan = scan_libraries("counts.csv")
        tilt = tilt_offsets(scan, scan.defective_lines)
        table, _ = activity_table("counts.csv", tilt=tilt)

    The factor divides the estimated technical tilt out of the latent rate, so
    it moves the whole B-vector and the activity follows; it is a model term,
    not a post-hoc edit of the activity scalar.

    ``lines`` is REQUIRED and says which cell lines to correct -- there is no
    default, because correcting the whole panel removes deviations the data
    gives no evidence are technical, and that should never happen by omission.
    Pass ``scan.defective_lines`` for the flagged ones, or an explicit list.
    Apply the SAME policy to every line you correct: a mixed policy interferes
    through any downstream multi-task model and is worse than either pure one.
    """
    col = {"mixture": "t_mixture", "gaussian": "t_gaussian"}[policy]
    keep = set(lines)
    unknown = keep - set(scan.table.line)
    if unknown:
        raise ValueError(f"not in the scan: {sorted(unknown)}")
    out = {}
    for l, sub in scan.table.groupby("line", sort=False):
        if l not in keep:
            continue
        g = np.zeros((scan.tilts.shape[1], scan.basis.shape[1]))
        amp = scale * float(sub[col].mean())
        g[:, :scan.n_sym] = amp * scan.axis[:, :scan.n_sym]
        out[l] = tilt_factor(scan.basis, -g)
    return out


def drop_defective(counts, scan=None, *, keep_lines=(), **scan_kw):
    """Delete the flagged libraries from a count table.

    The only response to the scan with no free parameter: it uses just the part
    the design identifies.  A flagged library that is a cell line's ONLY
    library removes that cell line -- ``keep_lines`` protects one you need
    downstream.  Returns ``(table, dropped)``.
    """
    df = counts if isinstance(counts, pd.DataFrame) else read_counts(counts)
    scan = scan_libraries(df, **scan_kw) if scan is None else scan
    drop = [(l, r) for l, r in scan.defective if l not in set(keep_lines)]
    if not drop:
        return df, []
    mask = [c[:2] not in set(drop) for c in df.columns]
    return df.loc[:, mask], drop
