"""Command line.

    ebin fit  counts.csv -o activity.csv     fit and write the activity table
    ebin qc   counts.csv -o qc/              flag defective libraries
    ebin hier counts.csv -a activity.csv     shrink the per-line curves

``ebin counts.csv`` with no subcommand is ``ebin fit``.
"""

import argparse
import os
import sys


def _add_fit_options(p):
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
    p.add_argument("--a-max", default="auto",
                   help="ceiling on the abundance a_n: a number pins it, "
                        "'auto' (default) starts at 20 and doubles the bound "
                        "of whatever sits on it until under --a-max-tol do")
    p.add_argument("--a-max-tol", type=float, default=0.001,
                   help="fraction of sequences allowed to sit at the "
                        "abundance bound before it is raised again")
    p.add_argument("--sum-mode", choices=("grid", "window"), default="grid",
                   help="how the latent count is summed out: the dense "
                        "truncated grid, or a constant number of nodes centred "
                        "on the mode of the summand (cost independent of depth "
                        "and of --a-max)")
    p.add_argument("--tau-buckets", type=int, default=6,
                   help="distinct latent-count grid lengths the sequences are "
                        "bucketed onto (1 = one grid sized by the deepest)")
    p.add_argument("--lambda-fix", type=float, default=50.0)
    p.add_argument("--mu-prior", type=float, default=1.3,
                   help="sd of the prior on the effect location (0 disables)")
    p.add_argument("--sigma-prior", default="0,0.5",
                   help="mean,sd of the prior on log sigma ('none' disables)")
    p.add_argument("--sigma-shared", action="store_true",
                   help="one effect scale for the whole cell line instead of "
                        "one per sequence (homoskedastic ordered probit)")
    p.add_argument("--shash", choices=("global", "eps"), default=None,
                   help="sinh-arcsinh effect law instead of the Normal: "
                        "'global' fits one (delta, eps) per cell line, 'eps' "
                        "a shared delta with a per-sequence skew")
    p.add_argument("--eps-prior", default="0,0.5",
                   help="mean,sd of the prior on the sinh-arcsinh skew "
                        "('none' disables); only bites for --shash eps")
    p.add_argument("-K", "--components", type=int, default=1,
                   help="NB emission components, sharing one component per "
                        "sequence across all cell lines (1 = no mixture)")
    p.add_argument("--em-rounds", type=int, default=4,
                   help="EM rounds for the emission mixture")
    p.add_argument("--grid", default="121x33",
                   help="readout grid, n_mu x n_log_sigma")
    p.add_argument("--shared-grid", action="store_true",
                   help="read the posterior off one grid shared by every "
                        "sequence, instead of re-gridding each on its own "
                        "posterior; quantizes deep sequences onto single nodes")
    p.add_argument("--fields", help="comma-separated subset of the output "
                                    "fields (default: all of them)")
    p.add_argument("--save-fits", help="pickle the fitted parameters here")
    p.add_argument("--warm-from", help="warm-start from a fits pickle")
    p.add_argument("--netcdf", help="also write the full posterior to netCDF")
    p.add_argument("--plots", metavar="DIR",
                   help="also write the diagnostic plots into DIR")
    p.add_argument("--drop-defective", action="store_true",
                   help="run the library QC first and delete the libraries it "
                        "flags before fitting.  The only response to the scan "
                        "with no free parameter; note that dropping a cell "
                        "line's only library removes the cell line")
    p.add_argument("--correct-defective", action="store_true",
                   help="instead of dropping them, divide the estimated "
                        "technical tilt out of the flagged LINES' profiles.  "
                        "Mutually exclusive with --drop-defective")
    p.add_argument("--defect-threshold", type=float, default=0.3,
                   help="P(defective) above which a library is flagged")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="ebin", description="Fit the EBin model to a bin-count table. "
                                 "Defaults are the shipped configuration.")
    sub = p.add_subparsers(dest="cmd")
    _add_fit_options(sub.add_parser("fit", help="fit and write the activity "
                                                "table"))

    q = sub.add_parser("qc", help="flag defective sequencing libraries")
    q.add_argument("data")
    q.add_argument("-o", "--out", default="qc",
                   help="directory for the scan table and figure")
    q.add_argument("--groups", help="comma-separated cell lines to scan")
    q.add_argument("--degree", type=int, default=2,
                   help="polynomial degree of the GC basis")
    q.add_argument("--min-reads", type=int, default=20,
                   help="objects below this total are dropped from the "
                        "per-library tilt regression")
    q.add_argument("--defect-threshold", type=float, default=0.3)
    q.add_argument("--bootstrap", type=int, default=0,
                   help="bootstrap resamples over cell lines for the interval "
                        "on the removal fraction (0 = skip; 300 is plenty)")
    q.add_argument("--no-plot", action="store_true")

    h = sub.add_parser("hier", help="shrink per-line covariate response curves")
    h.add_argument("data")
    h.add_argument("-a", "--activity",
                   help="activity CSV to shrink (default: only estimate and "
                        "report the weights)")
    h.add_argument("-o", "--out", default="hier",
                   help="directory for the report and figures")
    h.add_argument("--groups")
    h.add_argument("--smoother", choices=("lowess", "linear"), default="lowess",
                   help="'linear' makes every curve a straight line, which is "
                        "the control on how much of a curve is just a slope")
    h.add_argument("--frac", type=float, default=0.3, help="LOWESS bandwidth")
    h.add_argument("--terms", choices=("both", "gc", "depth"), default="both")
    h.add_argument("--scale", type=float, default=1.0,
                   help="multiplier on the estimated weight; for sensitivity "
                        "checks only -- above 1 is a claim the data does not "
                        "make")
    h.add_argument("--exclude-defective", action="store_true",
                   help="run the library QC and leave the lines it flags out "
                        "of the VARIANCE ESTIMATE (nothing is dropped from the "
                        "data or the output).  Off by default: a pair whose "
                        "second member failed measures the failure rather than "
                        "the technical noise, but which pairs to trust is your "
                        "call -- panel C of the figure shows where the weight "
                        "is coming from")
    h.add_argument("--drop-lines", help="comma-separated cell lines to leave "
                                        "out of the variance estimate")
    h.add_argument("--netcdf", help="write the shrunk activity to netCDF "
                                    "(needs -a)")
    h.add_argument("--no-plot", action="store_true")

    if argv and argv[0] not in {"fit", "qc", "hier", "-h", "--help"}:
        argv = ["fit"] + list(argv)          # bare `ebin counts.csv`
    a = p.parse_args(argv)
    if a.cmd is None:
        p.print_help()
        raise SystemExit(2)
    return a


def _fit_kwargs(a):
    sigma_prior = None if a.sigma_prior.lower() == "none" else \
        tuple(float(v) for v in a.sigma_prior.split(","))
    eps_prior = None if a.eps_prior.lower() == "none" else \
        tuple(float(v) for v in a.eps_prior.split(","))
    a_max = a.a_max if a.a_max.lower() == "auto" else float(a.a_max)
    kw = dict(conditional=a.zero_truncation, a_max=a_max,
              tau_buckets=a.tau_buckets, sum_mode=a.sum_mode,
              lambda_fix=a.lambda_fix, sigma_prior=sigma_prior,
              sigma_shared=a.sigma_shared, mu_prior=a.mu_prior or None,
              shash=a.shash, eps_prior=eps_prior)
    if a_max == "auto":
        kw["bound_frac"] = a.a_max_tol
    if a.abundance == "gamma":
        kw.update(abundance=False, abundance_prior="gamma",
                  n_abund_nodes=a.nodes)
    return kw


def cmd_fit(a):
    from .data import read_counts
    from .pipeline import activity_table, mixture_activity_table, \
        write_netcdf, cuts_table

    if a.drop_defective and a.correct_defective:
        raise SystemExit("--drop-defective and --correct-defective are "
                         "mutually exclusive")
    groups = a.groups.split(",") if a.groups else None
    data, fit_kw = a.data, _fit_kwargs(a)
    if a.drop_defective or a.correct_defective:
        from .qc import scan_libraries, drop_defective, tilt_offsets
        # scanned on the WHOLE table even when only some lines are fitted: the
        # technical component is identified by the replicated lines, and
        # restricting the panel throws that identification away
        data = read_counts(a.data)
        scan = scan_libraries(data, threshold=a.defect_threshold,
                              verbose=not a.quiet)
        if not scan.defective:
            print("[qc] nothing flagged; fitting the table as given")
        elif a.drop_defective:
            data, dropped = drop_defective(data, scan)
            print("[qc] dropped " + ", ".join(f"{l}/{r}" for l, r in dropped))
            left = set(data.columns.get_level_values(0))
            gone = [g for g in (groups or ()) if g not in left]
            if gone:
                print(f"[qc] {', '.join(gone)} had no library left and is not "
                      f"fitted; --correct-defective keeps it")
                groups = [g for g in groups if g in left]
                if not groups:
                    raise SystemExit(
                        "every cell line you asked for lost its only library; "
                        "use --correct-defective, or --groups with a line that "
                        "survives")
        else:
            fit_kw["tilt"] = tilt_offsets(scan, lines=scan.defective_lines)
            print("[qc] correcting " + ", ".join(fit_kw["tilt"]))

    n_mu, n_ls = (int(v) for v in a.grid.lower().split("x"))
    common = dict(groups=groups, out=a.out, fits_out=a.save_fits,
                  warm_from=a.warm_from,
                  readout=dict(n_mu=n_mu, n_ls=n_ls,
                               adaptive=not a.shared_grid),
                  fields=tuple(a.fields.split(",")) if a.fields else None,
                  verbose=not a.quiet)
    state = None
    if a.components > 1:
        table, results, state = mixture_activity_table(
            data, K=a.components, em_rounds=a.em_rounds, **common, **fit_kw)
    else:
        table, results = activity_table(data, **common, **fit_kw)
    print(f"[saved] {a.out}  {table.shape[0]} sequences x "
          f"{len(results)} cell lines")
    cuts_out = a.out.rsplit(".", 1)[0] + "_cuts.csv"
    cuts_table(results, out=cuts_out)
    print(f"[saved] {cuts_out}  bin cut points per cell line")
    if a.netcdf:
        write_netcdf(a.netcdf, results, table.index, order=list(results),
                     attrs=dict(model="EBin",
                                readout=f"{n_mu}x{n_ls} grid"
                                + (" (shared)" if a.shared_grid
                                   else " (per sequence)")))
        print(f"[saved] {a.netcdf}")
    if a.plots:
        from .plots import plot_all
        plot_all(table, data, a.plots, groups=list(results), state=state)


def cmd_qc(a):
    from .qc import scan_libraries, removal_interval
    os.makedirs(a.out, exist_ok=True)
    scan = scan_libraries(a.data, groups=a.groups.split(",") if a.groups
                          else None, degree=a.degree, min_reads=a.min_reads,
                          threshold=a.defect_threshold, verbose=True)
    path = os.path.join(a.out, "library_scan.csv")
    scan.table.to_csv(path, index=False)
    print(f"[saved] {path}")
    if a.bootstrap:
        boot = removal_interval(scan, n=a.bootstrap)
        boot.to_csv(os.path.join(a.out, "removal_bootstrap.csv"), index=False)
        q = boot.quantile([0.025, 0.5, 0.975]).T
        q.columns = ["2.5%", "median", "97.5%"]
        print("\n  bootstrap over cell lines:")
        print(q.to_string(float_format=lambda x: f"{x:9.4f}"))
    if not a.no_plot:
        from .plots import plot_library_qc
        plot_library_qc(scan, out=os.path.join(a.out, "library_qc.png"))


def cmd_hier(a):
    import pandas as pd
    from .hier import replicate_curves, curve_shrinkage, shrink_activity
    from .qc import scan_libraries
    os.makedirs(a.out, exist_ok=True)
    groups = a.groups.split(",") if a.groups else None

    scan = None
    if a.exclude_defective:
        scan = scan_libraries(a.data, groups=groups)
        print(f"[qc] leaving {', '.join(scan.defective_lines) or 'nothing'} "
              f"out of the variance estimate")
    curves = replicate_curves(a.data, groups=groups, frac=a.frac,
                              method=a.smoother, verbose=True)
    sh = curve_shrinkage(curves, scan=scan, scale=a.scale,
                         drop=a.drop_lines.split(",") if a.drop_lines else ())
    print("\n" + sh.summary())

    if a.activity:
        table = pd.read_csv(a.activity, index_col=0, header=[0, 1])
        new, report = shrink_activity(table, sh, counts=a.data, terms=a.terms,
                                      frac=a.frac, method=a.smoother,
                                      verbose=True)
        path = os.path.join(a.out, os.path.basename(a.activity))
        new.to_csv(path)
        report.to_csv(os.path.join(a.out, "shrinkage_report.csv"), index=False)
        print(f"[saved] {path}")
        if a.netcdf:
            from .pipeline import table_to_netcdf
            table_to_netcdf(a.netcdf, new, attrs=dict(
                model=f"EBin {__import__('ebin').__version__}",
                arm=f"hier_{a.smoother}", smoother=a.smoother,
                shrinkage_terms=a.terms, shrinkage_scale=str(a.scale),
                excluded_from_variance=", ".join(sh.dropped) or "none",
                source=os.path.basename(a.activity)))
            print(f"[saved] {a.netcdf}")
    elif a.netcdf:
        raise SystemExit("--netcdf needs -a/--activity: there is nothing to "
                         "write without a table to shrink")
    if not a.no_plot:
        from .plots import plot_curve_shrinkage
        for term in ("gc", "depth"):
            plot_curve_shrinkage(curves, sh, term=term,
                                 out=os.path.join(a.out, f"curves_{term}.png"))


def main(argv=None):
    a = _parse_args(sys.argv[1:] if argv is None else list(argv))
    {"fit": cmd_fit, "qc": cmd_qc, "hier": cmd_hier}[a.cmd](a)


if __name__ == "__main__":
    main()
