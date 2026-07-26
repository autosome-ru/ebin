"""End-to-end driver: fit every cell line, read out the activity, write tables."""

import os
import pickle
import time

import numpy as np
import pandas as pd

from .data import load_groups, prep
from .fit import fit_effects
from .initialize import init_from_fit
from .activity import posterior_activity

# every per-object scalar the readout produces; all of it goes to the table
TABLE_FIELDS = ("activity", "activity_sd", "activity_map",
                "mu", "mu_sd", "sigma", "sigma_sd", "q1", "q2", "q3",
                "abundance", "n_eff", "tot")

FIELD_DESC = {
    "activity": "posterior-mean E[bin] (the activity target)",
    "activity_sd": "posterior SD of E[bin] (per-object confidence)",
    "activity_map": "plug-in MAP E[bin] (point estimate, reference)",
    "mu": "posterior-mean effect location (gauge units)",
    "mu_sd": "posterior SD of mu (standard error)",
    "sigma": "posterior-mean effect scale (gauge units)",
    "sigma_sd": "posterior SD of sigma (standard error)",
    "q1": "1st quartile of the posterior-mean effect distribution",
    "q2": "2nd quartile (median = mu) of the effect distribution",
    "q3": "3rd quartile of the posterior-mean effect distribution",
    "abundance": "posterior-mean profiled abundance / size factor a_n",
    "n_eff": "posterior effective grid support (larger = more uncertain)",
    "tot": "total observed reads for this object",
    "bin_prob": "posterior-mean probability of falling in each bin",
}


class StoredFit:
    """The part of a fit the readout needs, as loaded from disk."""

    def __init__(self, d):
        self.__dict__.update(d)


def light(res):
    """A ``FitResult`` without the optimizer history: everything the readout
    reads, plus the fitted per-object effect law and abundance."""
    return dict(R=res.R, P=res.P, rates=res.rates, log_w=res.log_w,
                cuts=res.cuts, Pi=res.Pi, mu=res.mu, sigma=res.sigma, a=res.a,
                observed=res.observed, phi=res.phi, loglik=res.loglik,
                converged=res.converged, extras=res.extras, config=res.config)


def save_fits(path, fits):
    with open(path, "wb") as f:
        pickle.dump(fits, f)


def load_fits(path):
    with open(path, "rb") as f:
        return {g: StoredFit(d) for g, d in pickle.load(f).items()}


def activity_table(data, groups=None, *, out=None, fits_out=None,
                   warm_from=None, readout=None, fields=TABLE_FIELDS,
                   verbose=True, **fit_kw):
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
    fit_kw     : passed to ``fit_effects``.

    Returns (table, results) where ``table`` has (cell line, field) columns and
    ``results`` maps cell line -> the full readout dict (including the (N, B)
    ``bin_prob``, which is too wide for the table).
    """
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
        for f in fields:
            cols[(g, f)] = pd.Series(act[f], index=index)
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


def readout_table(data, fits_path, groups=None, *, out=None, readout=None,
                  fields=TABLE_FIELDS, verbose=True):
    """Redo the readout from saved fits -- no refit.

    Useful for trying a different readout grid or prior, or for rebuilding the
    table (or the netCDF) without paying for the fits again.
    """
    gdata = data if isinstance(data, dict) else load_groups(data, groups=groups,
                                                            verbose=verbose)
    fits = load_fits(fits_path)
    order = [g for g in (groups or fits) if g in fits and g in gdata]
    if not order:
        raise ValueError(f"no cell lines in both the data and {fits_path}")
    index = gdata[order[0]].index

    results, cols = {}, {}
    for g in order:
        X, mask, _ = prep(gdata[g])
        act = posterior_activity(fits[g], X, mask, **(readout or {}))
        results[g] = act
        for f in fields:
            cols[(g, f)] = pd.Series(act[f], index=index)
        if verbose:
            print(f"  {g}: activity median "
                  f"{np.nanmedian(act['activity']):.3f}", flush=True)
    return _write_table(cols, index, out), results


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
    per-bin distribution bin_prob (cell_type, seq, bin)."""
    import xarray as xr

    order = list(order or results)
    B = results[order[0]]["bin_prob"].shape[1]
    binp = np.stack([results[g]["bin_prob"] for g in order]).astype(np.float32)
    ds = xr.Dataset(
        {"bin_prob": (("cell_type", "seq", "bin"), binp,
                      {"description": FIELD_DESC["bin_prob"]}),
         **{f: (("cell_type", "seq"),
                np.stack([results[g][f] for g in order]).astype(np.float32),
                {"description": FIELD_DESC[f]})
            for f in TABLE_FIELDS}},
        coords={"cell_type": order, "seq": np.asarray(index),
                "bin": np.arange(1, B + 1)},
        attrs=attrs or {})
    ds.to_netcdf(path)
    return ds
