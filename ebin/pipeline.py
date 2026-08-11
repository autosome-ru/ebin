"""End-to-end driver: fit every cell line, read out the activity, write tables."""

import os
import time

import numpy as np
import pandas as pd

from .data import load_groups, prep
from .fit import fit_effects, StoredFit, light, save_fits, load_fits  # noqa: F401
from .initialize import init_from_fit
from .activity import posterior_activity
from .mixture import fit_mixture, components_table, save_state

# everything the readout produces per sequence goes to the table.  The bin cuts
# are NOT here: there is one set per cell line, not one per sequence -- see
# cuts_table().  ``component`` / ``comp_prob`` are appended only under an
# emission mixture, where they are not constant.
TABLE_FIELDS = ("activity", "activity_sd", "activity_map",
                "mu", "mu_sd", "sigma", "sigma_sd",
                "abundance", "n_eff", "tot", "bin_prob")
MIXTURE_FIELDS = TABLE_FIELDS + ("component", "comp_prob")

# fields that are (N, ·) rather than (N,): written as one column per column of
# the array, named with this prefix (bin_prob -> bin1 ... binB)
WIDE_FIELDS = {"bin_prob": "bin", "comp_prob": "comp"}

FIELD_DESC = {
    "activity": "posterior-mean E[bin] (the activity target)",
    "activity_sd": "posterior SD of E[bin] (per-object confidence)",
    "activity_map": "plug-in MAP E[bin] (point estimate, reference)",
    "mu": "posterior-mean effect location (gauge units)",
    "mu_sd": "posterior SD of mu (standard error)",
    "sigma": "posterior-mean effect scale (gauge units)",
    "sigma_sd": "posterior SD of sigma (standard error)",
    "abundance": "posterior-mean profiled abundance / size factor a_n",
    "n_eff": "posterior effective grid support (larger = more uncertain)",
    "tot": "total observed reads for this object",
    "component": "most probable emission component (1-based)",
    "bin_prob": "posterior-mean probability of falling in each bin",
    "comp_prob": "posterior probability of each emission component",
    "cuts": "bin cut points of this cell line: quantiles of the effect "
            "mixture at j/B, shared by every sequence",
}


def activity_table(data, groups=None, *, out=None, fits_out=None,
                   warm_from=None, readout=None, fields=None,
                   K=1, verbose=True, **fit_kw):
    """Fit each cell line and assemble the activity table.

    data       : path to the count table, or a DataFrame / dict of GroupData.
    groups     : cell lines to run, in output order (default: all, file order).
    out        : CSV path; rewritten after every line so a long run can be
                 inspected while it goes.
    fits_out   : pickle path for the fitted parameters (reusable readouts).
    warm_from  : pickle of earlier fits to warm-start from (same data, e.g.
                 refitting under the Gamma abundance prior).
    readout    : kwargs for ``posterior_activity`` (defaults reproduce the
                 shipped readout: the fit's own priors on a 121 x 33 grid).
    fields     : which per-object scalars to write; all of them by default.
    K          : emission components.  K = 1 fits the cell lines independently.
                 K > 1 shares one component index per sequence across all of
                 them, which couples the lines and so runs the EM in
                 ``mixture.fit_mixture`` -- see ``mixture_activity_table``,
                 which returns the fitted mixture as well.
    fit_kw     : passed to ``fit_effects`` (or, for K > 1, on to
                 ``fit_mixture``: ``em_rounds``, ``mstep_kw``, ``init_gamma``).

    Returns (table, results) where ``table`` has (cell line, field) columns --
    including the per-bin distribution, as ``bin1 ... binB`` -- and ``results``
    maps cell line -> the readout dict, which holds the same fields in their
    natural shape plus the line's cut points.
    """
    if int(K) > 1:
        table, results, _ = mixture_activity_table(
            data, groups, out=out, fits_out=fits_out, warm_from=warm_from,
            readout=readout, fields=fields, K=K, verbose=verbose, **fit_kw)
        return table, results
    fields = TABLE_FIELDS if fields is None else fields
    gdata = data if isinstance(data, dict) else load_groups(data, groups=groups,
                                                            verbose=verbose)
    order = [g for g in (groups or gdata) if g in gdata]
    if not order:
        raise ValueError(f"no cell lines to fit (have {sorted(gdata)})")
    warm = load_fits(warm_from) if warm_from else {}
    index = gdata[order[0]].index

    fits, results, cols = {}, {}, {}
    for i, g in enumerate(order):
        t0 = time.time()
        X, mask, _ = prep(gdata[g])
        init = init_from_fit(warm[g].__dict__) if g in warm else None
        res = fit_effects(X, mask=mask, init=init, verbose=False, **fit_kw)
        act = posterior_activity(res, X, mask, **(readout or {}))
        fits[g], results[g] = light(res), act
        cols.update(_columns(act, g, fields, index))
        if out:
            _write_table(cols, index, out)
        if fits_out:
            save_fits(fits_out, fits)
        if verbose:
            a = act["activity"]
            print(f"[{i+1}/{len(order)}] {g}: n={int(np.isfinite(a).sum())} "
                  f"loglik={res.loglik:.1f} conv={res.converged} "
                  f"phi={res.phi:.4f} activity mean={np.nanmean(a):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return _write_table(cols, index, out), results


def mixture_activity_table(data, groups=None, *, out=None, fits_out=None,
                           warm_from=None, readout=None, fields=None, K=2,
                           verbose=True, **mix_kw):
    """Fit the shared K-component emission mixture and read out the activity.

    The fits come from ``mixture.fit_mixture`` (EM: the component index is one
    per sequence for the whole dataset, so only the E-step sees every cell line
    at once).  Each line is then read out conditioned on the responsibilities
    formed from the OTHER lines, which is what keeps a line's own reads from
    being counted twice.

    Returns (table, results, state).  With ``out`` set, the per-sequence
    component assignment is written next to it as ``<out>_components.csv``.
    """
    gdata = data if isinstance(data, dict) else load_groups(data, groups=groups,
                                                            verbose=verbose)
    fields = (MIXTURE_FIELDS if fields is None else fields)
    fits, state = fit_mixture(gdata, groups, K=K, warm_from=warm_from,
                              fits_out=fits_out, verbose=verbose, **mix_kw)

    index = state.index
    results, cols = {}, {}
    for i, g in enumerate(state.order):
        t0 = time.time()
        X, mask, _ = prep(gdata[g])
        act = posterior_activity(StoredFit(fits[g]), X, mask,
                                 log_gamma=state.log_gamma_loo(g),
                                 **(readout or {}))
        results[g] = act
        cols.update(_columns(act, g, fields, index))
        if out:
            _write_table(cols, index, out)
        if verbose:
            a = act["activity"]
            print(f"[{i+1}/{len(state.order)}] {g}: "
                  f"n={int(np.isfinite(a).sum())} "
                  f"activity mean={np.nanmean(a):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if out:
        stem = out.rsplit(".", 1)[0]
        components_table(state, out=stem + "_components.csv")
        # the readout cannot be redone without this: the component weights come
        # from every cell line at once and are not in the per-line fits
        save_state(stem + "_state.pkl", state)
        if verbose:
            print(f"[saved] {stem}_components.csv  shared component per "
                  f"sequence\n[saved] {stem}_state.pkl  mixture state "
                  f"(readout_table needs it)")
    return _write_table(cols, index, out), results, state


def readout_table(data, fits_path, groups=None, *, out=None, readout=None,
                  fields=None, state=None, verbose=True):
    """Redo the readout from saved fits -- no refit.

    Useful for trying a different readout grid or prior, or for rebuilding the
    table (or the netCDF) without paying for the fits again.  ``state`` is the
    ``MixtureState`` that goes with the fits; it is required for K > 1, since
    the component weights are not recoverable from one cell line.
    """
    gdata = data if isinstance(data, dict) else load_groups(data, groups=groups,
                                                            verbose=verbose)
    fits = load_fits(fits_path)
    order = [g for g in (groups or fits) if g in fits and g in gdata]
    if not order:
        raise ValueError(f"no cell lines in both the data and {fits_path}")
    K = max(int(fits[g].config.get("K", 1)) for g in order)
    if K > 1 and state is None:
        raise ValueError(f"{fits_path} holds a K={K} emission mixture; pass the "
                         "MixtureState it was fitted with (state=...)")
    fields = ((MIXTURE_FIELDS if K > 1 else TABLE_FIELDS)
              if fields is None else fields)
    index = gdata[order[0]].index

    results, cols = {}, {}
    for g in order:
        X, mask, _ = prep(gdata[g])
        extra = {} if state is None else dict(log_gamma=state.log_gamma_loo(g))
        act = posterior_activity(fits[g], X, mask, **extra, **(readout or {}))
        results[g] = act
        cols.update(_columns(act, g, fields, index))
        if verbose:
            print(f"  {g}: activity median "
                  f"{np.nanmedian(act['activity']):.3f}", flush=True)
    return _write_table(cols, index, out), results


def cuts_table(results, out=None):
    """The bin cut points, one row per cell line.

    These are the only quartiles in the model: quantiles of the effect mixture
    at j/B, shared by every sequence of the line.  The gauge pins the first two
    (-1 and 0), so the informative part is where the upper cuts land.
    """
    tab = pd.DataFrame({g: np.asarray(r["cuts"], float)
                        for g, r in results.items()}).T
    tab.columns = [f"q{j + 1}" for j in range(tab.shape[1])]
    tab.index.name = "group"
    if out:
        tab.to_csv(out)
    return tab


def _columns(act, g, fields, index):
    """The (cell line, field) columns of one readout.

    A field that is (N, ·) -- the per-bin distribution, the per-component
    posterior -- becomes one column per column of the array rather than being
    dropped: nothing the model estimates per sequence is too wide for a table
    with B = 4 bins.
    """
    cols = {}
    n = len(index)
    for f in fields:
        if f == "cuts":
            raise ValueError("'cuts' is one set of numbers per cell line, not "
                             "per sequence, so it is not a table field -- use "
                             "cuts_table(results), or the CLI's <out>_cuts.csv")
        v = np.asarray(act[f])
        if v.shape[:1] != (n,):
            raise ValueError(f"field {f!r} has shape {v.shape}, which is not "
                             f"one row per sequence ({n})")
        if v.ndim == 1:
            cols[(g, f)] = pd.Series(v, index=index)
            continue
        stem = WIDE_FIELDS.get(f, f)
        for j in range(v.shape[1]):
            cols[(g, f"{stem}{j + 1}")] = pd.Series(v[:, j], index=index)
    return cols


def _write_table(cols, index, out=None):
    tab = pd.DataFrame(cols, index=index)
    tab.columns = pd.MultiIndex.from_tuples(tab.columns,
                                            names=["group", "score"])
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        tab.to_csv(out)
    return tab


def write_netcdf(path, results, index, order=None, attrs=None):
    """Write the readout to netCDF: every (cell_type, seq) field plus the
    per-bin distribution bin_prob (cell_type, seq, bin) and, under an emission
    mixture, comp_prob (cell_type, seq, component)."""
    import xarray as xr

    order = list(order or results)
    first = results[order[0]]
    B = first["bin_prob"].shape[1]
    K = first["comp_prob"].shape[1] if "comp_prob" in first else 1
    # the wide fields get their own dimension below, not the (cell_type, seq) one
    scalars = [f for f in MIXTURE_FIELDS if f in first and f not in WIDE_FIELDS]
    if K == 1:
        scalars = [f for f in scalars if f != "component"]
    stack = lambda f: np.stack([results[g][f] for g in order])
    data = {"bin_prob": (("cell_type", "seq", "bin"),
                         stack("bin_prob").astype(np.float32),
                         {"description": FIELD_DESC["bin_prob"]}),
            "cuts": (("cell_type", "cut"), stack("cuts"),
                     {"description": FIELD_DESC["cuts"]}),
            **{f: (("cell_type", "seq"), stack(f).astype(np.float32),
                   {"description": FIELD_DESC[f]}) for f in scalars}}
    coords = {"cell_type": order, "seq": np.asarray(index),
              "bin": np.arange(1, B + 1), "cut": np.arange(1, B)}
    if K > 1:
        # the axis is "comp", not "component": that name is already the
        # per-sequence variable, and xarray rejects the collision
        data["comp_prob"] = (("cell_type", "seq", "comp"),
                             stack("comp_prob").astype(np.float32),
                             {"description": FIELD_DESC["comp_prob"]})
        coords["comp"] = np.arange(1, K + 1)
    ds = xr.Dataset(data, coords=coords, attrs=attrs or {})
    ds.to_netcdf(path)
    return ds
