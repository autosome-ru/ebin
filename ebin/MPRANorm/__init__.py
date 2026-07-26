# -*- coding: utf-8 -*-
from .heuristic import fraction_estimator, fraction_map_estimator
from .distfit import fit as distfit
from .compound import infer_latent_counts
from .compound_omega import infer_latent_counts as infer_latent_counts_omega
from .compound_profile import infer_latent_counts as infer_latent_counts_profile
from .compound_beta import infer_latent_counts as infer_latent_counts_beta
from .compound_indep import infer_latent_counts as infer_latent_counts_indep