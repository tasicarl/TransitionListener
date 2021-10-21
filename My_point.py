from os import cpu_count
import scanner

'''

This script executes scanner.py with the simple task of checking whether the model defined in tl_dark_photon_model.py can give observable stochastic gravitational wave signals
for a given point in parameter space. Some more comments on the input parameters for scanner.py are given. Note also the list of paramters at the end of this file which can be
used to tweak the accuracy of the performed calcualtions. All of this also applies to the case of grid scans, where the runtime of the code however can easily exceed one day for
extensive scans on a grid larger than 50 x 50, for increased accuracy. As default, a trade-off has been found that prefers more precise results over low computational costs.

'''


# Set the names of the model parameters and their respective default values
INPUT_NAMES = ("lambda", "g", "vev", "xi_nuc", "Gamma_GeV")
INPUT_VALS = [1.5e-3, 0.5, 2e3, 5, 1e-16]

# Set 0 if no daisy resummation wanted, 1 for Parwani method and 2 for Arnold-Espinoza method. 
DAISY_TYPE = 2

# The VEV that is used as an input for TransitionListener. The physical VEV in GeV is set above.
# This incluences how the (internal) CONVERSION_FACTOR is going to be set.
STABLE_VEV = 1e2

# If an error occurs, try again with slightly changed input model parameters, but only N_TRY_MAX times
N_TRY_MAX = 3

# Relative shift and the index of the parameter to be shifted in case of an error
ARGS_SHIFT = 1e-3
ARG_INDEX_TO_SHIFT = 0

# Set True if only one model parameter point is to be analyzed. Else, whole array
# of model configurations will be scanned over.
SINGLE_POINT = True

# Set True if you want to see what TransitionListener is doing currently
# Warning: Can have considerable impact on runtime!
VERBOSE = True

# Description of scan to be put in filenames and plot titles
# The latter also includes matplotlib-LaTeX code
DESCRIPTION = "Single_parameter_point"
OVERVIEW_TITLE = R"Single parameter point"

# In case of scan over square grid of model parameters (SINGLE_POINT = False), set to >+ 3
GRID_N = 10

# Scale of x and y axis of the grid scan {"lin", "loglin", "linlog", "loglog"}
SCALE = "loglog"

# Names of parameters to be scannes over, taken from INPUT_NAMES and in matplotlib-LaTeX version for plots
GRID_PARAMS_NAMES = ("lambda", "Gamma_GeV")
GRID_PARAMS_PLOT_NAMES = (r"$\lambda$", r"$\Gamma$ / GeV")

# Ranges for grid scan of x and y parameter. If linear axis, corresponds to parameter. If log axis,
# the parameter range corresponds to the log10 value (e.g., -3 corresponds to 0.001)
X_MINMAX = [-4, -1]
Y_MINMAX = [-26, -10]

# If SINGLE_PROCESS == True, all will be calcualted usign one CPU. Else, number of CPUs set below
SINGLE_PROCESS = False
NUMBER_CPUS_USED = cpu_count() - 1

# If PLOT_POTENTIAL_AT_T == True, the only thing that the program is going to do is to plot the
# effective potential of the defined model at a given temeperature PLOT_T (in units of STABLE_VEV)
PLOT_POTENTIAL_AT_T = False
PLOT_T = 1000

# Name of the detector that is used to geenrate an overview plot of four panels, including the
# respective detector's signal-to-noise ratio
OVERVIEW_DETECTOR_NAME = "LISA"


scanner.Full_calculation(DESCRIPTION, OVERVIEW_TITLE, INPUT_NAMES, INPUT_VALS, DAISY_TYPE, STABLE_VEV, N_TRY_MAX, ARGS_SHIFT, ARG_INDEX_TO_SHIFT, SINGLE_POINT, GRID_N, SCALE, GRID_PARAMS_NAMES, GRID_PARAMS_PLOT_NAMES, X_MINMAX, Y_MINMAX, SINGLE_PROCESS, VERBOSE, NUMBER_CPUS_USED, PLOT_POTENTIAL_AT_T, PLOT_T, OVERVIEW_DETECTOR_NAME)



'''

To increase the precision of the analysis it is possible to also change the following hidden parameters at the cost of getting a larger runtime:

1) In ./TransitionListener/transitionFinder.py modify
	rel_slope_error = 0.01 # to lower value
    N_first_try = 50 # to higher value
    N_second_try = 100 # to higher value
    N_cutted_fit = 15 # to higher value
to obtain a better computation of beta/H. The parameter rel_slope_error sets the limit up to which the slope of the euclidean bounce action S(T) for T < Tnuc can mismatch
the value for T > Tnuc. If the slopes are in agreement with each other, a weighted mean of them is used in the calcualtion of beta/H. N_first_try is the number of points of
support that are computed initially for calculating the slope above and below Tnuc. If the relative error of the slopes is larger then rel_slope_error an additional
N_second_try points of support are calculated to compute the derivative. The separated linear fits of S(T) are only done for T < Tnuc and T > Tnuc separately, if the number
of points of support exceeds N_cutted_fit on both sides. If not, a linear fit is computed, ignoring that S(T) might have a jump (due to numerics...) around Tnuc.
A value of N_cutted_fit above 5 is highly recommended.

2) In ./tl_dark_photon_model.py decrease
	self.x_eps = .001
    self.T_eps = .001
for higher working precision when evaluating the effective potential. Both values are used at various places throughout the program while dealing with the phase tracing
in the effective potential. These can easily be reduced without changing the runtime too much. The result is an increased precision of the critical temperature and the Higgs
field's vacuum expectation value in dependence of temperature.

3) In ./TransitionListener/generic_potential.py in method calcGWTrans() when self.GWTrans = transitionFinder.findAllTransitionsGW(...) is called, set "Ttol": 1e-8 to lower value in
tunnelFromPhase_args = {...} to increase the precision in calculating Tnuc. This can drastically increase precision and runtime. The same holds for the parameter phitol = 1e-8, which
is set directly in ''def tunnelFromPhaseGW(...)''. Ttol sets the precision on the x-axis of the brentq minimization of the nucleation criterion while the resulting euclidean action's
precision is set by phitol. 

'''