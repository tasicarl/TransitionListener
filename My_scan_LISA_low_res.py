from os import cpu_count
import numpy as np
import scanner

if __name__ == "__main__": 
	DAISY_TYPE = 2
	STABLE_VEV = 1e2
	N_TRY_MAX = 3
	ARGS_SHIFT = 1e-5
	ARG_INDEX_TO_SHIFT = 1
	SINGLE_POINT = False
	GRID_N = 3
	SCALE = "loglog"
	GRID_PARAMS_NAMES = ("xi_nuc", "Gamma_GeV")
	GRID_PARAMS_PLOT_NAMES = (R"$\xi_\mathrm{nuc}$", R"$\Gamma$ / GeV")
	SINGLE_PROCESS = False
	VERBOSE = True if SINGLE_POINT else False
	NUMBER_CPUS_USED = cpu_count() - 1
	PLOT_POTENTIAL_AT_T = False
	PLOT_T = 1208
	DESCRIPTION = "Scan_LISA_BP_at_low_resolution"
	OVERVIEW_TITLE = R"Some complicated title that also allows for greek letters as $\zeta$"
	INPUT_NAMES = ("lambda", "g", "vev", "xi_nuc", "Gamma_GeV")
	INPUT_VALS = [1.5e-3, 0.5, 2e3, 0, 0] # vev in Gev
	X_MINMAX = [0, 1]
	Y_MINMAX = [-19, -12]
	scanner.Full_calculation(DESCRIPTION, OVERVIEW_TITLE, INPUT_NAMES, INPUT_VALS, DAISY_TYPE, STABLE_VEV, N_TRY_MAX, ARGS_SHIFT, ARG_INDEX_TO_SHIFT, SINGLE_POINT, GRID_N, SCALE, GRID_PARAMS_NAMES, GRID_PARAMS_PLOT_NAMES, X_MINMAX, Y_MINMAX, SINGLE_PROCESS, VERBOSE, NUMBER_CPUS_USED, PLOT_POTENTIAL_AT_T, PLOT_T, "LISA")

