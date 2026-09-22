"""Hierarchical shrinkage of a cell line's covariate response curves.

A cell line's activity column carries a response to sequence composition (GC)
and to read depth.  Most of that is biology, but part of it is the library, and
a line whose curve is far from the panel's is partly reporting its own prep.
Decompose every line as

    z_l  =  s_l(GC)  +  r_l(depth)  +  resid_l

and shrink the two curves toward the panel mean with an ESTIMATED weight:

    s_new = s_bar + (1 - w) (s_l - s_bar),    w = s2_tech / (s2_tech + s2_bio)

A typical line barely moves; an anomalous one moves most.  "Why this line?"
becomes an output rather than a choice -- which is what separates this from
substituting a hand-picked donor line's curve.

WHERE THE WEIGHT COMES FROM.  Not from any downstream metric: from the
replicated cell lines.  Fit the same curves to each REPLICATE separately and
difference them -- biology is identical in both replicates of a line, so it
cancels, and what is left is what re-running the library would have changed.
The replicate curves are fitted on a model-free per-replicate activity (the
observed mean bin of the raw counts), because the fitted model shares one
profile across a line's replicates and cannot produce a per-replicate curve at
all.  The weight is a RATIO of variances, so it transfers to the fitted
activity provided the relative noise structure is similar; that is an
assumption, stated here rather than hidden.

A replicate difference also contains estimation noise, which would inflate
s2_tech and the weight with it.  ``replicate_curves`` therefore also fits each
library on two disjoint halves of the sequences; those differ by pure counting
noise and calibrate it out.

TWO WARNINGS, both learned the hard way on the panel this was developed
against:

*  A replicate pair whose second member is a DEFECTIVE library does not measure
   "what re-running this library would change" -- it measures the failure.  One
   such pair carried 75% of the GC curve's sum of squares and inflated the
   weight threefold.  ``curve_shrinkage`` uses every pair unless you pass
   ``drop=`` or ``scan=``; ``plot_curve_shrinkage`` panel C shows which pairs
   the weight is resting on so the call is yours to make.
*  Once such a pair is excluded the weight is small and the effect on anything
   downstream is near-null.  The gains reported at the larger weight scaled
   with shrinkage STRENGTH, not with the estimation being right.  The DIAGNOSIS
   -- which lines have anomalous curves -- survives; the magnitude does not.
   Treat this as a diagnostic first and a correction second.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import bin_order, read_counts

__all__ = ["CurveShrinkage", "replicate_curves", "curve_shrinkage",
           "shrink_activity", "gc_content", "additive"]


def gc_content(sequences):
    """GC fraction of each sequence.

    Validated, because a blank or stray row in a hand-edited table otherwise
    reaches the curve fit as a NaN covariate and surfaces as an unreadable
    ``SVD did not converge in Linear Least Squares``.
    """
    s = pd.Index(sequences).astype(str).str.upper()
    n = s.str.len().to_numpy()
    bad = (n == 0) | (s.str.count("[ACGTN]").to_numpy() != n)
    if bad.any():
        show = ", ".join(repr(v[:30]) for v in s[bad][:3])
        raise ValueError(
            f"{bad.sum()} of {len(s)} index entries are not DNA sequences "
            f"({show}): a hand-edited table usually means a blank or shifted "
            f"row.  Pass sequences= if the index is not the sequence.")
    return ((s.str.count("G") + s.str.count("C")) / n).to_numpy()


# --------------------------------------------------------------------------
# the additive decomposition
# --------------------------------------------------------------------------
def _smooth_lowess(x, y, fit, frac, it=3):
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as e:                                # pragma: no cover
        raise ImportError("method='lowess' needs statsmodels "
                          "(pip install ebin[hier]); method='linear' does "
                          "not") from e
    xf, yf = x[fit], y[fit]
    sm = lowess(yf, xf, frac=frac, it=it, delta=0.01 * (xf.max() - xf.min()),
                return_sorted=True)
    gx, i = np.unique(sm[:, 0], return_index=True)
    gy = sm[i, 1]
    return lambda q: np.interp(np.clip(q, gx[0], gx[-1]), gx, gy)


def _smooth_linear(x, y, fit):
    b, a0 = np.polyfit(np.asarray(x)[fit], np.asarray(y)[fit], 1)
    return lambda q: a0 + b * np.asarray(q, float)


MIN_FIT_ROWS = 20          # below this a curve is noise, not a curve


def additive(y, covs, fit, frac=0.3, rounds=4, method="lowess"):
    """Backfit ``y ~ sum_k f_k(cov_k)``; returns the fitted 1-D curves.

    Each term is centred on the rows in ``fit``, so the curves carry no level.
    ``method="linear"`` makes every term a straight line, which reduces the
    decomposition to a multiple regression -- useful as a control on how much
    of a curve is just a slope.
    """
    fit = np.asarray(fit, bool)
    if fit.sum() < MIN_FIT_ROWS:
        raise ValueError(f"only {int(fit.sum())} usable rows, need at least "
                         f"{MIN_FIT_ROWS}: nothing here defines a curve")
    for k, x in enumerate(covs):
        if not np.isfinite(np.asarray(x, float)[fit]).all():
            raise ValueError(f"covariate {k} is not finite on all "
                             f"{int(fit.sum())} fitted rows")
    f = [np.zeros(len(y)) for _ in covs]
    fun = [None] * len(covs)
    for _ in range(rounds):
        for k, x in enumerate(covs):
            partial = y - sum(f[j] for j in range(len(covs)) if j != k)
            g = (_smooth_linear(x, partial, fit) if method == "linear"
                 else _smooth_lowess(x, partial, fit, frac))
            c = g(x[fit]).mean()
            fun[k] = (lambda g=g, c=c: lambda q: g(q) - c)()
            f[k] = fun[k](x)
    return fun


def _standardize(v, fit):
    out = np.full(len(v), np.nan)
    out[fit] = (v[fit] - v[fit].mean()) / v[fit].std()
    return out


# --------------------------------------------------------------------------
# stage A: one curve pair per sequencing library, from raw counts
# --------------------------------------------------------------------------
def replicate_curves(counts, sequences=None, *, n_grid=41, frac=0.3,
                     min_reads=20, method="lowess", groups=None, seed=0,
                     verbose=False):
    """Model-free GC and depth curves for every library, plus a noise estimate.

    Returns ``(frame, grids)``.  ``frame`` has one row per (line, replicate)
    with ``s``/``r`` (the two curves on the shared grids) and ``s_half``/
    ``r_half`` (the difference between two disjoint-half fits of the SAME
    library, which is pure estimation noise).
    """
    df = counts if isinstance(counts, pd.DataFrame) else read_counts(counts)
    df = df.fillna(0.0)
    seqs = df.index.to_numpy() if sequences is None else np.asarray(sequences)
    gc = gc_content(seqs)

    libs = [c[:2] for c in df.columns]
    libs = [l for l in dict.fromkeys(libs) if groups is None or l[0] in groups]
    gcg = np.quantile(gc, np.linspace(0.02, 0.98, n_grid))
    depg = np.linspace(-2.5, 2.5, n_grid)
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(gc)) < len(gc) // 2

    rows = []
    for line, rep in libs:
        sub = df[line][rep]
        b_names = list(dict.fromkeys(sub.columns))
        X = sub[[b_names[i] for i in bin_order(b_names)]].to_numpy(float)
        tot = X.sum(1)
        ok = (tot >= min_reads) & np.isfinite(gc)
        if ok.sum() < MIN_FIT_ROWS:
            raise ValueError(
                f"library {line}/{rep}: only {int(ok.sum())} sequences reach "
                f"min_reads={min_reads}, so it carries no curve.  Lower "
                f"min_reads, or take that library out of the table "
                f"(--groups excludes a whole cell line)")
        ebin = np.full(len(tot), np.nan)
        ebin[ok] = (X[ok] @ np.arange(1, X.shape[1] + 1)) / tot[ok]
        z = _standardize(ebin, ok)
        d = np.nan_to_num(_standardize(np.log10(np.maximum(tot, 1.0)), ok))
        y = np.nan_to_num(z)

        def curves(fit):
            f = additive(y, [gc, d], fit, frac, method=method)
            return f[0](gcg), f[1](depg)

        s, r = curves(ok)
        sA, rA = curves(ok & half)
        sB, rB = curves(ok & ~half)
        rows.append(dict(line=line, replicate=rep, s=s, r=r,
                         s_half=sA - sB, r_half=rA - rB))
        if verbose:
            print(f"  [curve] {line:<24}{rep:<12} |s| {np.linalg.norm(s):6.3f}"
                  f"  |r| {np.linalg.norm(r):6.3f}"
                  f"  noise {np.linalg.norm(sA - sB):6.3f}/"
                  f"{np.linalg.norm(rA - rB):6.3f}")
    return pd.DataFrame(rows), dict(gc=gcg, depth=depg)


# --------------------------------------------------------------------------
# stage B: variance components and the weight
# --------------------------------------------------------------------------
def _components(frame, key, drop):
    """A full-data replicate difference carries 2 s2_tech + 2 s2_noise; a
    split-half difference within ONE library carries 4 s2_noise (two half-data
    curves, each with twice a full curve's estimation variance).  So
    s2_tech = Var(rep diff)/2 - Var(split-half diff)/4.

    EVERY within-line pair is used, so a line with three libraries contributes
    its three pairs rather than being skipped.  The noise term is averaged over
    the libraries that actually contribute a pair.
    """
    C = np.stack(frame[key].to_numpy())
    H = np.stack(frame[key + "_half"].to_numpy())
    lines = frame.line.to_numpy()
    reps = [l for l in dict.fromkeys(lines.tolist())
            if (lines == l).sum() >= 2 and l not in drop]
    if not reps:
        raise ValueError("no usable replicate pair: the technical curve "
                         "variance is not identified")

    D, pairs, used = [], [], set()
    for l in reps:
        idx = np.where(lines == l)[0]
        used.update(idx.tolist())
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                D.append(C[idx[a]] - C[idx[b]])
                pairs.append(l if len(idx) == 2 else f"{l}[{a + 1}v{b + 1}]")
    D = np.stack(D)

    s2_noise = (H[sorted(used)] ** 2).mean(0) / 4.0
    s2_raw = (D ** 2).mean(0) / 2.0
    s2_tech = np.maximum(s2_raw - s2_noise, 1e-12)

    uniq = list(dict.fromkeys(lines.tolist()))
    means = np.stack([C[lines == l].mean(0) for l in uniq])
    n_rep = np.array([(lines == l).sum() for l in uniq], float)
    s2_bio = np.maximum(means.var(0, ddof=1) - s2_tech / n_rep.mean(), 1e-12)
    return dict(s2_tech=s2_tech, s2_bio=s2_bio, s2_noise=s2_noise,
                s2_raw=s2_raw, pairs=pairs, diffs=D, line_curves=means,
                lines=uniq, n_rep=dict(zip(uniq, n_rep.astype(int))))


@dataclass
class CurveShrinkage:
    """Per-grid-point variance components and shrinkage weights.

    ``weight(term, x, n_rep)`` gives the weight at the covariate values ``x``
    for a line with ``n_rep`` libraries.
    """
    gc: dict
    depth: dict
    grids: dict
    dropped: tuple = ()
    scale: float = 1.0

    def weight(self, term, x, n_rep=1):
        c = self.gc if term == "gc" else self.depth
        t = c["s2_tech"] / max(n_rep, 1)
        w = np.clip(self.scale * t / (t + c["s2_bio"]), 0.0, 1.0)
        g = self.grids[term]
        return np.interp(np.clip(x, g[0], g[-1]), g, w)

    def summary(self):
        out = [f"excluded pairs: {', '.join(self.dropped) or 'none'}"]
        for term, c in (("gc", self.gc), ("depth", self.depth)):
            w = c["s2_tech"] / (c["s2_tech"] + c["s2_bio"])
            out.append(
                f"  {term:<6} pairs {len(c['pairs']):2d} "
                f"({', '.join(dict.fromkeys(p.split('[')[0] for p in c['pairs']))})   "
                f"sigma_tech {np.sqrt(c['s2_tech']).mean():.4f}  "
                f"sigma_bio {np.sqrt(c['s2_bio']).mean():.4f} "
                f"over {len(c['lines'])} lines  "
                f"noise {c['s2_noise'].sum() / c['s2_raw'].sum():5.1%} of the "
                f"raw difference   w(S=1) {w.mean():.3f}")
        n = len(self.gc["lines"])
        if n < 5:
            out.append(f"  CAUTION: sigma_bio is a variance over {n} cell "
                       f"lines, so the weight is barely determined")
        return "\n".join(out)


def curve_shrinkage(frame, scan=None, *, drop=(), scale=1.0):
    """Variance components and shrinkage weights from the replicate curves.

    By default EVERY replicate pair is used.  ``drop`` names cell lines to
    leave out of the variance estimate, and ``scan`` (a
    :class:`ebin.qc.LibraryScan`) drops the ones it flags -- both are opt-in,
    and neither touches the data or the output, only which pairs estimate
    sigma_tech.

    Worth knowing before you decide: a pair whose second member is a defective
    library does not measure "what re-running this library would change", it
    measures the failure.  On one real panel a single such pair carried 75% of
    the GC curve's sum of squares and inflated the weight threefold.  Panel C
    of ``plot_curve_shrinkage`` shows which pairs the weight rests on, and
    ``summary()`` reports what was excluded.

    ``scale`` multiplies the weight.  It exists to run the sensitivity check,
    not to tune: the weight is what the replicates say, and a scale above 1 is
    a claim the data does not make.
    """
    drop = set(drop) | set(scan.defective_lines if scan is not None else ())
    if isinstance(frame, tuple):
        frame, grids = frame
    else:
        raise TypeError("pass the (frame, grids) pair returned by "
                        "replicate_curves()")
    return CurveShrinkage(gc=_components(frame, "s", drop),
                          depth=_components(frame, "r", drop),
                          grids=grids, dropped=tuple(sorted(drop)),
                          scale=scale)


# --------------------------------------------------------------------------
# applying it
# --------------------------------------------------------------------------
def shrink_activity(table, shrinkage, counts=None, *, sequences=None,
                    depth=None, terms="both", frac=0.3, method="lowess",
                    field="activity", restore_sd=False, loo=True,
                    verbose=False):
    """Shrink every cell line's curves in a fitted activity table.

    ``table`` is what ``activity_table`` returns.  Read depth comes from
    ``counts`` (per line, summed over replicates and bins) or from an explicit
    ``depth`` frame.  Returns a copy of the table with ``field`` replaced.

    ``restore_sd=False`` (the default) keeps the column's original mean and SD
    scaling but does not renormalize the shrunk column back up -- shrinking a
    curve is meant to reduce the spread, and rescaling it away undoes the
    correction.  ``loo=True`` leaves each line out of its own panel mean.
    """
    lines = [c for c in dict.fromkeys(table.columns.get_level_values(0))]
    if len(lines) < 2:
        raise ValueError(
            f"the activity table has {len(lines)} cell line(s): a line is "
            f"shrunk toward the mean of the OTHER lines, so with one column "
            f"there is nothing to shrink toward and the call would be a silent "
            f"no-op.  Shrink the whole panel, then take the column you want")
    seqs = table.index.to_numpy() if sequences is None else np.asarray(sequences)
    gc = gc_content(seqs)
    dep = _depth_frame(table, counts, depth, lines)
    n_rep = shrinkage.gc["n_rep"]

    dec, rows = {}, []
    for c in lines:
        y = table[(c, field)].to_numpy(float)
        d_raw = dep[c].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(d_raw) & np.isfinite(gc)
        z = np.nan_to_num(_standardize(y, ok))
        d = np.nan_to_num(_standardize(np.log10(np.maximum(d_raw, 1.0)), ok))
        fun = additive(z, [gc, d], ok, frac, method=method)
        dec[c] = dict(fun=fun, ok=ok, d=d, y=y,
                      resid=z - fun[0](gc) - fun[1](d))

    S_gc = np.stack([dec[c]["fun"][0](gc) for c in lines])
    n_l = len(lines)
    out = table.copy()
    for i, c in enumerate(lines):
        D = dec[c]
        ok, d, y = D["ok"], D["d"], D["y"]
        s_l, r_l = D["fun"][0](gc), D["fun"][1](d)
        s_bar = ((n_l * S_gc.mean(0) - S_gc[i]) / (n_l - 1) if loo and n_l > 1
                 else S_gc.mean(0))
        R = np.stack([dec[k]["fun"][1](d) for k in lines])
        r_bar = ((R.sum(0) - R[i]) / (n_l - 1) if loo and n_l > 1 else R.mean(0))
        S_c = n_rep.get(c, 1)

        if terms in ("both", "gc"):
            wg = shrinkage.weight("gc", gc, S_c)
            s_new = s_bar + (1.0 - wg) * (s_l - s_bar)
        else:
            wg, s_new = np.zeros(len(gc)), s_l
        if terms in ("both", "depth"):
            wd = shrinkage.weight("depth", d, S_c)
            r_new = r_bar + (1.0 - wd) * (r_l - r_bar)
        else:
            wd, r_new = np.zeros(len(gc)), r_l

        zn = D["resid"] + (s_new + r_new) - (s_new + r_new)[ok].mean()
        sc = 1.0 / zn[ok].std() if restore_sd else 1.0
        new = y.copy()
        new[ok] = y[ok].mean() + y[ok].std() * sc * (zn[ok] - zn[ok].mean())
        out[(c, field)] = new
        rows.append(dict(line=c, n_rep=S_c, w_gc=float(wg[ok].mean()),
                         w_depth=float(wd[ok].mean()),
                         amp_gc=float(s_l[ok].std()),
                         amp_gc_new=float(s_new[ok].std()),
                         r_old_new=float(np.corrcoef(y[ok], new[ok])[0, 1])))
    report = pd.DataFrame(rows).sort_values("r_old_new")
    if verbose:
        print(report.to_string(index=False,
                              float_format=lambda x: f"{x:8.4f}"))
    return out, report


def _depth_frame(table, counts, depth, lines):
    if depth is not None:
        return depth
    if counts is None:
        raise ValueError("pass counts= (or depth=) so the depth curve has a "
                         "covariate")
    df = counts if isinstance(counts, pd.DataFrame) else read_counts(counts)
    df = df.fillna(0.0)
    missing = [c for c in lines if c not in df.columns.get_level_values(0)]
    if missing:
        raise ValueError(
            f"the activity table has cell line(s) {missing} that the count "
            f"table does not.  Pass the counts the table was fitted from")
    dep = pd.DataFrame({c: df[c].to_numpy(float).sum(1) for c in lines},
                       index=df.index).reindex(table.index)
    if not np.isfinite(dep.to_numpy(float)).any():
        raise ValueError("no sequence of the activity table is in the count "
                         "table: the two indexes do not match")
    return dep
