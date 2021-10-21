from os import cpu_count
import scanner

INPUT_NAMES = ("lambda", "g", "vev", "xi_nuc", "Gamma_GeV")
INPUT_VALS = [2e-3, 0.5, 2e3, 1, 1e-18]
DAISY_TYPE = 2 
STABLE_VEV = 1e2
N_TRY_MAX = 3
ARGS_SHIFT = 1e-3
ARG_INDEX_TO_SHIFT = 0
SINGLE_POINT = True
VERBOSE = True
DESCRIPTION = "Single_parameter_point"
OVERVIEW_TITLE = R"Single parameter point"
GRID_N = 0
SCALE = "loglog"
GRID_PARAMS_NAMES = ("lambda", "Gamma_GeV")
GRID_PARAMS_PLOT_NAMES = (r"$\lambda$", r"$\Gamma$ / GeV")
X_MINMAX = [-4, -1]
Y_MINMAX = [-26, -10]
SINGLE_PROCESS = False
NUMBER_CPUS_USED = cpu_count() - 1
PLOT_POTENTIAL_AT_T = True
PLOT_T = 10
OVERVIEW_DETECTOR_NAME = ""
scanner.Full_calculation(DESCRIPTION, OVERVIEW_TITLE, INPUT_NAMES, INPUT_VALS, DAISY_TYPE, STABLE_VEV, N_TRY_MAX, ARGS_SHIFT, ARG_INDEX_TO_SHIFT, SINGLE_POINT, GRID_N, SCALE, GRID_PARAMS_NAMES, GRID_PARAMS_PLOT_NAMES, X_MINMAX, Y_MINMAX, SINGLE_PROCESS, VERBOSE, NUMBER_CPUS_USED, PLOT_POTENTIAL_AT_T, PLOT_T, OVERVIEW_DETECTOR_NAME)


