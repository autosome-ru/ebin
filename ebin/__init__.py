"""EBin -- posterior bin-expectation normalization for sorting-based MPRA.

Reads raw per-bin read counts and returns, for every sequence in every cell
line, the posterior-mean expected bin E[bin] (the activity) together with its
posterior SD.  The model is a compound NB-Poisson likelihood over a latent
Gaussian effect law per sequence; see model.py, effects.py and activity.py.

    from ebin import activity_table
    table, _ = activity_table("counts.csv", out="activity.csv")
"""

import os

# jax grabs 75% of VRAM otherwise; float64 throughout (the likelihood sums
# thousands of log terms per object)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from .data import GroupData, read_counts, load_groups, prep  # noqa: E402,F401
from .initialize import initialize, init_from_fit, InitResult  # noqa: E402,F401
from .fit import fit_effects, FitResult  # noqa: E402,F401
from .activity import posterior_activity  # noqa: E402,F401
from .pipeline import (activity_table, readout_table, save_fits,  # noqa: E402,F401
                       load_fits, write_netcdf)
from .plots import (plot_marginal_distribution, plot_activity_vs_raw,  # noqa: E402,F401
                    plot_gc_vs_abundance, plot_all, raw_mass_center,
                    gc_content, marginal_density)

__version__ = "1.0.1"
