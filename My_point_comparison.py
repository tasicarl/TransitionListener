from os import cpu_count
from TransitionListener import observability
from time import time
import numpy as np
import scanner

DAISY_TYPE = 2
STABLE_VEV = 1e2
N_TRY_MAX = 3
ARGS_SHIFT = 1e-3
ARG_INDEX_TO_SHIFT = 0
PLOT_POTENTIAL_AT_T = False
PLOT_T = 1208
VERBOSE = False
OVERVIEW_DETECTOR_NAME = "LISA"


# ("lambda", "g", "vev", "xi_nuc", "Gamma_GeV")
INPUT_VALS = [2e-2, 1, 1e3, 1, 1e-16]
scan = scanner.Full_calculation("", "", ("", ""), INPUT_VALS, DAISY_TYPE, STABLE_VEV, N_TRY_MAX, ARGS_SHIFT, ARG_INDEX_TO_SHIFT, True, 0, "", "", "", 0, 0, True, VERBOSE, 0, PLOT_POTENTIAL_AT_T, PLOT_T, OVERVIEW_DETECTOR_NAME)
GW_params1 = scan.GW_params
print(scan.GW_params)

INPUT_VALS = [2e-2, 1, 1e3, 2, 1e-16]
scan = scanner.Full_calculation("", "", ("", ""), INPUT_VALS, DAISY_TYPE, STABLE_VEV, N_TRY_MAX, ARGS_SHIFT, ARG_INDEX_TO_SHIFT, True, 0, "", "", "", 0, 0, True, VERBOSE, 0, PLOT_POTENTIAL_AT_T, PLOT_T, OVERVIEW_DETECTOR_NAME)
GW_params2 = scan.GW_params
print(scan.GW_params)

INPUT_VALS = [2e-2, 1, 1e3, 5, 1e-16]
scan = scanner.Full_calculation("", "", ("", ""), INPUT_VALS, DAISY_TYPE, STABLE_VEV, N_TRY_MAX, ARGS_SHIFT, ARG_INDEX_TO_SHIFT, True, 0, "", "", "", 0, 0, True, VERBOSE, 0, PLOT_POTENTIAL_AT_T, PLOT_T, OVERVIEW_DETECTOR_NAME)
GW_params3 = scan.GW_params
print(scan.GW_params)

INPUT_VALS = [2e-2, 1, 1e3, 5, 1e-18]
scan = scanner.Full_calculation("", "", ("", ""), INPUT_VALS, DAISY_TYPE, STABLE_VEV, N_TRY_MAX, ARGS_SHIFT, ARG_INDEX_TO_SHIFT, True, 0, "", "", "", 0, 0, True, VERBOSE, 0, PLOT_POTENTIAL_AT_T, PLOT_T, OVERVIEW_DETECTOR_NAME)
GW_params4 = scan.GW_params
print(scan.GW_params)

INPUT_VALS = [2e-2, 1, 1e3, 5, 1e-20]
scan = scanner.Full_calculation("", "", ("", ""), INPUT_VALS, DAISY_TYPE, STABLE_VEV, N_TRY_MAX, ARGS_SHIFT, ARG_INDEX_TO_SHIFT, True, 0, "", "", "", 0, 0, True, VERBOSE, 0, PLOT_POTENTIAL_AT_T, PLOT_T, OVERVIEW_DETECTOR_NAME)
GW_params5 = scan.GW_params
print(scan.GW_params)

# List of the parameters necessary to calculate the GW signal
GW_params = np.array((
	GW_params1,
	GW_params2,
	GW_params3,
	GW_params4,
	GW_params5))

# Plot labels
labels = (
	R"$\xi$ = 1, $\Gamma$ = $10^{-16}$ GeV",
	R"$\xi$ = 2, $\Gamma$ = $10^{-16}$ GeV",
	R"$\xi$ = 5, $\Gamma$ = $10^{-16}$ GeV",
	R"$\xi$ = 5, $\Gamma$ = $10^{-18}$ GeV",
	R"$\xi$ = 5, $\Gamma$ = $10^{-20}$ GeV")

# Styles of the to be plotted lines
linestyles = (
	"dotted",
	"dashed",
	"solid",
	"solid",
	"solid")

# Colors of the to be plotted lines
colors = (
	"darkorange",
	"darkorange",
	"darkorange",
	"darkblue",
	"darkred")

# Opacities of the to be plotted lines
alphas = (
	1,
	1,
	1,
	1,
	1)

# Plot title and filename
title = "Effect of heating up the dark sector"
outputname = "Fancy_overview_plot"

# Number of lines
N = 5

obs = observability.GW_spectra_comparison()
obs.plot_N_spectra(GW_params, labels, linestyles, colors, alphas, title, outputname, N)