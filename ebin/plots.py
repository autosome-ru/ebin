"""Diagnostic plots for a fitted activity table.

Each function takes the table returned by ``activity_table`` (or read back from
the CSV with ``pd.read_csv(path, index_col=0, header=[0, 1])``), draws one panel
per cell line, and returns the figure.  matplotlib is imported lazily, so it is
only needed if you plot.
"""

import numpy as np
import pandas as pd

from .data import load_groups, prep


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return matplotlib, plt


INK, MUTE, GRID, COOL, WARN = ("#1c1c22", "#8b8b96", "#e6e6ec",
                                       "#4c78a8", "#d1495b")


def _style():
    """House rcParams, applied per-figure so importing ebin changes nothing."""
    _, plt = _plt()
    return plt.rc_context({
        "font.size": 9, "axes.edgecolor": MUTE, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white"})


def _clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _label(ax, text):
    """Panel identifier, drawn inside the axes instead of as a title."""
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=INK)


def _defect_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "defect", ["#cfd6e4", "#8ea3c4", "#e2a33c", WARN])


def _lines(table, groups=None):
    have = list(dict.fromkeys(table.columns.get_level_values(0)))
    return [g for g in (groups or have) if g in have]


def _grid(n, ncols=5, size=(3.2, 2.7)):
    _, plt = _plt()
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(size[0] * ncols, size[1] * nrows))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, list(axes.ravel()[:n])


def _finish(fig, out, layout=True):
    if layout:
        fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved] {out}")
    return fig


def raw_mass_center(counts, groups=None):
    """Depth-normalized empirical E[bin] straight from the counts.

    Every (replicate, bin) column is divided by its own library size — the
    bins are separately sequenced, so this is the normalization that has to
    happen before the bins are comparable — then replicates are summed and each
    sequence's profile is normalized to give ``sum_b b * p_b``.  No model:
    this is the yardstick to check the fitted activity against.
    """
    gdata = counts if isinstance(counts, dict) else \
        load_groups(counts, groups=groups, verbose=False)
    out, index = {}, None
    for g, gd in gdata.items():
        X, mask, _ = prep(gd)
        Xz = np.where(mask, np.nan_to_num(X), 0.0)
        p = Xz / np.maximum(Xz.sum(0)[None], 1e-9)      # per-channel depth
        y = p.sum(1)                                    # over replicates, (N,B)
        s = y.sum(1)
        b = np.arange(1, X.shape[2] + 1, dtype=float)
        out[g] = np.where(s > 0, (y * b).sum(1) / np.maximum(s, 1e-30), np.nan)
        index = gd.index
    return pd.DataFrame(out, index=index)


def gc_content(sequences):
    """GC fraction of each sequence."""
    s = pd.Index(sequences).astype(str).str.upper()
    letters = s.str.count("[ACGTN]").to_numpy()
    if not np.all(letters == s.str.len().to_numpy()):
        raise ValueError("sequences must be DNA strings; pass sequences= "
                         "explicitly if the table index is not the sequence")
    return ((s.str.count("G") + s.str.count("C")) / s.str.len()).to_numpy()


def marginal_density(mu, sigma, x, weights=None, chunk=4096):
    """(pdf, cdf) of the marginal effect law on the grid ``x``.

    The marginal is the N-component normal mixture the model actually
    parameterizes -- one component per sequence, not an averaged-out curve:

        G(x) = sum_n w_n * Phi((x - mu_n) / sigma_n),   sum_n w_n = 1.

    ``weights`` default to uniform, which is what the fit uses; pass e.g. the
    fitted abundances to weight sequences by how many cells they contribute.
    Accumulated in chunks, so the N x len(x) array never materializes.
    """
    from scipy.stats import norm
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    ok = np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    w = np.ones(mu.size) if weights is None else np.asarray(weights, float)
    ok &= np.isfinite(w)
    mu, sigma, w = mu[ok], sigma[ok], w[ok]
    w = w / w.sum()
    pdf = np.zeros_like(x)
    cdf = np.zeros_like(x)
    for i in range(0, mu.size, chunk):
        s = slice(i, i + chunk)
        z = (x[None, :] - mu[s, None]) / sigma[s, None]
        pdf += (w[s, None] * norm.pdf(z) / sigma[s, None]).sum(0)
        cdf += (w[s, None] * norm.cdf(z)).sum(0)
    return pdf, cdf


def plot_marginal_distribution(table, groups=None, n_bins=4, cuts=None,
                               weights=None, out=None, ncols=5, grid=1500,
                               pad=3.0):
    """The marginal effect law of each cell line, sliced into the sorting bins.

    The curve is the N-component mixture the model estimates,
    G(x) = sum_n w_n Phi((x - mu_n)/sigma_n) — one normal per sequence, drawn
    as the density it defines — and the bins are its quantile cells.  Each bin's
    slice is filled in its own colour, and the x ticks are the cut points (the
    quartiles of G for B = 4) labelled with their values.  Every slice holds
    1/B of the mass by construction, so unequal-looking areas mean the density
    is concentrated there, not that the bins are unbalanced.

    cuts    : {cell line: array} to draw the fit's own cut points; by default
        they are read off G itself, which is how the fit defines them.
    weights : {cell line: array} of mixture weights; uniform by default, as in
        the fit.
    """
    matplotlib, plt = _plt()
    lines = _lines(table, groups)
    B = n_bins
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, B))
    probs = np.arange(1, B) / B

    fig, axes = _grid(len(lines), ncols, (3.4, 2.8))
    for ax, g in zip(axes, lines):
        mu = table[(g, "mu")].to_numpy(float)
        sd = table[(g, "sigma")].to_numpy(float)
        m = np.isfinite(mu) & np.isfinite(sd)
        lo = np.percentile(mu[m], 0.2) - pad * np.median(sd[m])
        hi = np.percentile(mu[m], 99.8) + pad * np.median(sd[m])
        x = np.linspace(lo, hi, grid)
        w = None if weights is None else np.asarray(weights[g], float)[m]
        pdf, cdf = marginal_density(mu[m], sd[m], x, weights=w)
        q = np.asarray(cuts[g]) if cuts is not None else np.interp(probs, cdf, x)

        edges = np.concatenate([[x[0]], q, [x[-1]]])
        for b in range(B):
            sel = (x >= edges[b]) & (x <= edges[b + 1])
            ax.fill_between(x[sel], pdf[sel], color=colors[b], linewidth=0)
        ax.plot(x, pdf, color="0.15", lw=1.0)
        for v in q:
            ax.axvline(v, color="0.15", lw=0.8, ls="--")
        ax.set_xticks(q)
        ax.set_xticklabels([f"{v:.2f}" for v in q], fontsize=7, rotation=45,
                           ha="right", rotation_mode="anchor")
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        _label(ax, f"{g}  (n={int(m.sum())})")
    # reserve a strip at the top for the bin legend (a fixed ~0.35 inch,
    # whatever the number of rows)
    top = 1.0 - 0.35 / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(handles=[matplotlib.patches.Patch(color=colors[b],
                                                 label=f"bin {b + 1}")
                        for b in range(B)],
               loc="upper center", ncol=B, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    return _finish(fig, out, layout=False)


def plot_activity_vs_raw(table, counts, groups=None, out=None, ncols=5):
    """Fitted activity against the depth-normalized raw mass centre, coloured
    by the fitted abundance.

    The two agree for well-measured sequences; the low-abundance ones are where
    the model shrinks and the raw estimate is noise."""
    from scipy.stats import spearmanr
    _, plt = _plt()
    lines = _lines(table, groups)
    raw = raw_mass_center(counts, lines).reindex(table.index)
    ab = np.concatenate([table[(g, "abundance")].to_numpy(float) for g in lines])
    ab = np.log10(np.clip(ab[np.isfinite(ab)], 1e-4, None))
    # a shared scale so panels are comparable; 5-95% because the a_min floor
    # piles up a long left tail that would otherwise eat the whole colormap
    vmin, vmax = np.percentile(ab, [5, 95])

    fig, axes = _grid(len(lines), ncols)
    sc = None
    for ax, g in zip(axes, lines):
        x = raw[g].to_numpy(float)
        y = table[(g, "activity")].to_numpy(float)
        c = np.log10(np.clip(table[(g, "abundance")].to_numpy(float), 1e-4, None))
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
        sc = ax.scatter(x[m], y[m], c=c[m], s=2, alpha=0.4, linewidths=0,
                        cmap="viridis", vmin=vmin, vmax=vmax, rasterized=True)
        lim = [min(np.nanmin(x[m]), np.nanmin(y[m])),
               max(np.nanmax(x[m]), np.nanmax(y[m]))]
        ax.plot(lim, lim, color="0.4", lw=0.8, ls="--")
        rho = spearmanr(x[m], y[m]).statistic
        _label(ax, f"{g}  rho={rho:.3f}")
        ax.set_xlabel("raw E[bin] (depth-normalized)", fontsize=7)
        ax.set_ylabel("activity", fontsize=7)
        ax.tick_params(labelsize=7)
    if sc is not None:
        fig.colorbar(sc, ax=axes, shrink=0.6, label="log10 abundance")
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved] {out}")
    return fig


def plot_gc_vs_abundance(table, sequences=None, groups=None, out=None,
                         ncols=5, n_trend=25):
    """GC content against the fitted abundance, with the binned median trend.

    A strong trend means the size factor is picking up a sequencing/cloning
    bias rather than a biological one."""
    from scipy.stats import spearmanr
    _, plt = _plt()
    lines = _lines(table, groups)
    gc = gc_content(table.index if sequences is None else sequences)

    fig, axes = _grid(len(lines), ncols)
    for ax, g in zip(axes, lines):
        y = table[(g, "abundance")].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(gc) & (y > 0)
        ax.scatter(gc[m], y[m], s=2, alpha=0.3, linewidths=0, color="#3b6ea5",
                   rasterized=True)
        edges = np.quantile(gc[m], np.linspace(0, 1, n_trend + 1))
        idx = np.clip(np.digitize(gc[m], edges[1:-1]), 0, n_trend - 1)
        med = np.array([np.median(y[m][idx == k]) if (idx == k).any() else np.nan
                        for k in range(n_trend)])
        ax.plot(0.5 * (edges[:-1] + edges[1:]), med, color="#c0392b", lw=1.4)
        ax.set_yscale("log")
        _label(ax, f"{g}  rho={spearmanr(gc[m], y[m]).statistic:.3f}")
        ax.set_xlabel("GC content", fontsize=7)
        ax.set_ylabel("abundance", fontsize=7)
        ax.tick_params(labelsize=7)
    return _finish(fig, out)


def plot_components(state, counts, groups=None, out=None, bins=80):
    """The evidence for an emission mixture, in one figure.

    Left: the cross-line mean of log relative depth -- the sequence-intrinsic
    part of the read count, which is where a bimodal amplification shows up --
    with the fitted components stacked on it.  A mixture that is doing real
    work splits a visibly bimodal histogram; one that is splitting a smooth
    unimodal one is fitting a tail, not a population.  Right: how confident the
    assignment is, which says whether the cell lines agree.
    """
    from .mixture import relative_depth
    _, plt = _plt()
    gdata = counts if isinstance(counts, dict) else \
        load_groups(counts, groups=groups, verbose=False)
    order = [g for g in (groups or state.order) if g in gdata]
    D = relative_depth(gdata, order)
    seen = D > 0
    L = np.where(seen, np.log10(np.maximum(D, 1e-6)), 0.0).sum(1) \
        / np.maximum(seen.sum(1), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    k, conf = state.assignment()
    edges = np.linspace(np.nanpercentile(L, 0.2), np.nanpercentile(L, 99.8),
                        bins + 1)
    axes[0].hist([L[k == j] for j in range(state.K)], bins=edges, stacked=True,
                 label=[f"component {j+1} ({state.pi[j]:.1%})"
                        for j in range(state.K)])
    axes[0].set_xlabel("mean log10 relative depth across cell lines")
    axes[0].set_ylabel("sequences")
    axes[0].legend(fontsize=8)
    axes[1].hist(conf, bins=60, color="#3b6ea5")
    axes[1].set_xlabel("posterior probability of the assigned component")
    axes[1].set_ylabel("sequences")
    axes[1].set_yscale("log")
    return _finish(fig, out)


def plot_library_qc(scan, out=None):
    """Two panels: every library's tilt amplitude with its Fisher 95% interval,
    and its P(defective).  Sorted by amplitude, coloured by P(defective).

    The intervals are on the left panel to show the flags are not measurement
    error -- they are invisible at this scale against the between-library
    spread.
    """
    _, plt = _plt()
    t = scan.table.copy()
    t["tag"] = t.line.astype(str) + "  " + t.replicate.astype(str)
    S = t.sort_values("alpha")
    from matplotlib.colors import Normalize
    cmap, norm = _defect_cmap(), Normalize(0, 1)
    yy = np.arange(len(S))
    with _style():
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 1.1 + 0.30 * len(t)),
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
        ax = axes[0]
        ax.axvline(S.alpha.mean(), color=MUTE, ls="--", lw=1.0, zorder=1)
        ax.hlines(yy, 0, S.alpha, color=GRID, lw=1.2, zorder=1)
        ax.errorbar(S.alpha, yy, xerr=1.96 * S.se, fmt="none", ecolor=MUTE,
                    elinewidth=1.0, zorder=2)
        ax.scatter(S.alpha, yy, s=52, c=S.p_defective, cmap=cmap, norm=norm,
                   edgecolor="white", lw=0.6, zorder=3)
        ax.set_yticks(yy)
        ax.set_yticklabels(S.tag, fontsize=7.5)
        ax.set_ylim(-0.8, len(S) - 0.2)
        ax.set_xlabel("tilt amplitude")
        _clean(ax)

        ax = axes[1]
        ax.hlines(yy, 0, S.p_defective, color=GRID, lw=2.0, zorder=1)
        ax.scatter(S.p_defective, yy, s=48, c=S.p_defective, cmap=cmap,
                   norm=norm, edgecolor="white", lw=0.6, zorder=3)
        ax.set_yticks(yy)
        ax.set_yticklabels(S.tag, fontsize=7.5)
        ax.set_xlim(0, 1.06)
        ax.set_ylim(-0.8, len(S) - 0.2)
        ax.set_xlabel("P(defective)")
        _clean(ax)
        return _finish(fig, out)


def plot_curve_shrinkage(curves, shrinkage, out=None, term="gc"):
    """Where the shrinkage weight comes from, and what it does.

    A  every library's curve, so the panel spread is visible;
    B  the variance split per grid point -- biological, technical, and the
       estimation noise the split-half fits calibrate out -- with the weight
       that ratio implies;
    C  the within-line replicate differences the technical term is estimated
       from.  If one pair dominates this panel, the weight rests on that pair
       and the excluded-pair list is the thing to check.
    """
    _, plt = _plt()
    frame, grids = curves if isinstance(curves, tuple) else (curves, None)
    key = "s" if term == "gc" else "r"
    comp = shrinkage.gc if term == "gc" else shrinkage.depth
    g = shrinkage.grids[term]
    C = np.stack(frame[key].to_numpy())
    label = "GC content" if term == "gc" else "read depth (z)"
    with _style():
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
        ax = axes[0]
        for row, c in zip(frame.itertuples(), C):
            flag = row.line in shrinkage.dropped
            ax.plot(g, c, lw=1.8 if flag else 0.9,
                    color=WARN if flag else COOL, alpha=0.95 if flag else 0.45,
                    zorder=3 if flag else 2)
        ax.plot(g, C.mean(0), color=INK, lw=2.2, zorder=4, label="panel mean")
        ax.set_xlabel(label)
        ax.set_ylabel("activity (z)")
        ax.legend(fontsize=7.5, frameon=False)

        _clean(ax, grid="both")

        ax = axes[1]
        ax.fill_between(g, 0, comp["s2_bio"], color=COOL, alpha=0.45,
                        label="biological")
        ax.fill_between(g, comp["s2_bio"], comp["s2_bio"] + comp["s2_tech"],
                        color=WARN, alpha=0.45, label="technical")
        ax.plot(g, comp["s2_noise"], color=MUTE, ls=":", lw=1.4,
                label="estimation noise (removed)")
        ax.set_xlabel(label)
        ax.set_ylabel("variance per grid point")
        ax.legend(fontsize=7.5, frameon=False)
        w = comp["s2_tech"] / (comp["s2_tech"] + comp["s2_bio"])
        _label(ax, f"w(S=1) = {w.mean():.3f}")
        _clean(ax, grid="both")

        ax = axes[2]
        norms = np.linalg.norm(comp["diffs"], axis=1)
        order = np.argsort(-norms)
        xx = np.arange(len(order))
        ax.bar(xx, norms[order], width=0.62, color=COOL, zorder=3)
        ax.set_xticks(xx)
        ax.set_xticklabels([comp["pairs"][i] for i in order], rotation=38,
                           ha="right", fontsize=7.5)
        ax.set_ylabel("|replicate difference|")
        share = norms.max() ** 2 / max((norms ** 2).sum(), 1e-30)
        _label(ax, f"top pair = {share:.0%} of the sum of squares")
        _clean(ax, grid="y")
        return _finish(fig, out)


def plot_all(table, counts, outdir=".", groups=None, state=None, scan=None,
             curves=None, shrinkage=None):
    """Write the diagnostics into ``outdir``."""
    import os
    os.makedirs(outdir, exist_ok=True)
    p = lambda n: os.path.join(outdir, n)
    plot_marginal_distribution(table, groups, out=p("marginal_effect_law.png"))
    plot_activity_vs_raw(table, counts, groups, out=p("activity_vs_raw.png"))
    plot_gc_vs_abundance(table, groups=groups, out=p("gc_vs_abundance.png"))
    if state is not None and state.K > 1:
        plot_components(state, counts, groups, out=p("components.png"))
    if scan is not None:
        plot_library_qc(scan, out=p("library_qc.png"))
    if curves is not None and shrinkage is not None:
        for term in ("gc", "depth"):
            plot_curve_shrinkage(curves, shrinkage, term=term,
                                 out=p(f"curve_shrinkage_{term}.png"))
