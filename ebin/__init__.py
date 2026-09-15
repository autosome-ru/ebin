"""EBin -- posterior bin-expectation normalization for sorting-based MPRA.

Reads raw per-bin read counts and returns, for every sequence in every cell
line, the posterior-mean expected bin E[bin] (the activity) together with its
posterior SD.  The model is a compound NB-Poisson likelihood over a latent
Gaussian effect law per sequence; see model.py, effects.py and activity.py.

    from ebin import activity_table
    table, _ = activity_table("counts.csv", out="activity.csv")

Set ``K`` above 1 for a K-component NB emission mixture whose component is
chosen once per sequence for the whole dataset -- see mixture.py.

Two tools sit alongside the fit and are documented in their own modules:
``qc`` flags defective sequencing libraries from their bin x covariate tilt,
and ``hier`` shrinks a cell line's covariate response curves toward the panel.
Both are diagnostics first -- see ``docs/diagnostics.md``.
"""

import os

# jax grabs 75% of VRAM otherwise; float64 throughout (the likelihood sums
# thousands of log terms per object)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from .data import GroupData, read_counts, load_groups, prep  # noqa: E402,F401
from .initialize import initialize, init_from_fit, InitResult  # noqa: E402,F401
from .fit import (fit_effects, fit_effects_adaptive, FitResult,  # noqa: E402,F401
                  save_fits, load_fits, StoredFit, light,
                  abundance_caps, initial_abundance)
from .activity import posterior_activity  # noqa: E402,F401
from .freepi import fit_free_pi  # noqa: E402,F401
from .batch import (poly_basis, sample_tilt, variance_components,  # noqa: E402,F401
                    shrink_weight, apply_tilt, tilt_factor, tilt_axes)
from .model import logmarg_window, mode_scale  # noqa: E402,F401
from .mixture import (fit_mixture, MixtureState, components_table,  # noqa: E402,F401
                      component_summary, component_loglik, relative_depth,
                      ridge_step, penalty_nats, save_state, load_state)
from .pipeline import (activity_table, mixture_activity_table,  # noqa: E402,F401
                       readout_table, write_netcdf, cuts_table,
                       table_to_netcdf)
from .qc import (scan_libraries, LibraryScan, library_tilts,  # noqa: E402,F401
                 fit_failure_mixture, fit_reml, tilt_offsets, drop_defective,
                 removal_interval)
from .hier import (replicate_curves, curve_shrinkage,  # noqa: E402,F401
                   shrink_activity, CurveShrinkage, additive)
from .plots import (plot_marginal_distribution, plot_activity_vs_raw,  # noqa: E402,F401
                    plot_gc_vs_abundance, plot_components, plot_all,
                    plot_library_qc, plot_curve_shrinkage,
                    raw_mass_center, gc_content, marginal_density)

__version__ = "1.3.0"
