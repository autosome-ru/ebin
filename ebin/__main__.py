"""Command line: python -m ebin counts.csv -o activity.csv"""

import argparse

from .pipeline import activity_table, write_netcdf, TABLE_FIELDS


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m ebin",
        description="Fit the EBin model to a bin-count table and write the "
                    "per-cell-line activity table. Defaults are the shipped "
                    "configuration.")
    p.add_argument("data", help="count table, 3-level header "
                                "(cell line, replicate, bin)")
    p.add_argument("-o", "--out", default="activity.csv", help="output CSV")
    p.add_argument("--groups", help="comma-separated cell lines to run, in "
                                    "output order (default: all)")
    p.add_argument("--zero-truncation", action="store_true",
                   help="drop all-zero rows instead of fitting them as "
                        "structural zeros")
    p.add_argument("--abundance", choices=("free", "gamma"), default="free",
                   help="per-object abundance: a free a_n, or integrated out "
                        "under a Gamma prior")
    p.add_argument("--nodes", type=int, default=24,
                   help="quadrature nodes for the Gamma abundance integral")
    p.add_argument("--a-max", type=float, default=15.0)
    p.add_argument("--lambda-fix", type=float, default=50.0)
    p.add_argument("--mu-prior", type=float, default=1.3,
                   help="sd of the prior on the effect location (0 disables)")
    p.add_argument("--sigma-prior", default="0,0.5",
                   help="mean,sd of the prior on log sigma ('none' disables)")
    p.add_argument("--grid", default="121x33",
                   help="readout grid, n_mu x n_log_sigma")
    p.add_argument("--fields", help="comma-separated subset of the output "
                                    "fields (default: all of them)")
    p.add_argument("--save-fits", help="pickle the fitted parameters here")
    p.add_argument("--warm-from", help="warm-start from a fits pickle")
    p.add_argument("--netcdf", help="also write the full posterior to netCDF")
    p.add_argument("--plots", metavar="DIR",
                   help="also write the diagnostic plots into DIR")
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    n_mu, n_ls = (int(v) for v in a.grid.lower().split("x"))
    sigma_prior = None if a.sigma_prior.lower() == "none" else \
        tuple(float(v) for v in a.sigma_prior.split(","))
    fit_kw = dict(conditional=a.zero_truncation, a_max=a.a_max,
                  lambda_fix=a.lambda_fix, sigma_prior=sigma_prior,
                  mu_prior=a.mu_prior or None)
    if a.abundance == "gamma":
        fit_kw.update(abundance=False, abundance_prior="gamma",
                      n_abund_nodes=a.nodes)

    table, results = activity_table(
        a.data, groups=a.groups.split(",") if a.groups else None,
        out=a.out, fits_out=a.save_fits, warm_from=a.warm_from,
        readout=dict(n_mu=n_mu, n_ls=n_ls),
        fields=tuple(a.fields.split(",")) if a.fields else TABLE_FIELDS,
        verbose=not a.quiet, **fit_kw)
    print(f"[saved] {a.out}  {table.shape[0]} sequences x "
          f"{len(results)} cell lines")
    if a.netcdf:
        write_netcdf(a.netcdf, results, table.index, order=list(results),
                     attrs=dict(model="EBin", readout=f"{n_mu}x{n_ls} grid"))
        print(f"[saved] {a.netcdf}")
    if a.plots:
        from .plots import plot_all
        plot_all(table, a.data, a.plots, groups=list(results))


if __name__ == "__main__":
    main()
