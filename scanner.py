'''
The dp_analysis file functions as a control system for the tl_dark_photon_model module. It's possible to analyze single points or lattices of parameter points in the parameter space
of the model defined in model. When analyzing arrays of input parameters, parallel computation on multiple CPUs is optional. The tqdm package provides a progress bar
and an estimate of how long the calculation will take. In this case of the scan over a grid, several files are saved after the calculation: One file each for the GW
spectrum parameters (and those which we only want to see plotted, but are not acually relevant for calculating the GW signal), two files for the chosen input parameter grid
and one file each for every other input parameter. If errors occurr (e.g. no found phase transitions or unexpected outputs), the scanned parameter point gets shifted
minimally and the phase transition analysis of the model starts anew. In the case of consecutive errors, error will be given in the console output to understand and reproduce the error
later on, after the program has stopped. The produced output parameters are used to calculate the signal-to-noise ratio on the defined grid using the observability.py file,
which provides also a facility to plot all the output parameters and SNRs.

ERRORCODES:
0 Found a temperature which fulfills the nucleation criterion sufficiently well in brentq, but not good enough (|_tunnelFromPhaseAtT| < 1) to go on from here. I.e. most probably: a too-supercooled phase.
1 Found only one point that fulfills the nucleation criterion sufficiently well (|_tunnelFromPhaseAtT| < 1). This isn't suffient for a proper calculation of the beta parameter.
2 Due to some strange reason a negative beta was calculated. This shouldn't happen at all.
3 Adding more points of support to calculate the derivative of the euclidean action didn't yield a result with a relative error below 1 percent.
4 Found a second-order transition in TcTrans.
5 The concurring phase splitted unexpectedly in two minima. To avoid the calculation of the transitions from one minimum into the other, stop here.
6 Due to some strange reason a neither first, nor second-order transition has been documented. This shouldn't happen at all.
7 No temperature at which the nucleation criterion can be fulfilled was found.
8 There were at least two same points when calculating the linear regression for calculating beta. This yielded an infinte covariance matrix.
9 Didn't find a transition in TcTrans. Maybe the critical phase was too-supercooled.
10 Found that Tmin is bigger than Tmax: Tunneling cannot occur.
12 A second order transition was found after TcTrans.
13 A non-listed error occurred after calcGWTrans.
'''

import time
import numpy as np
from itertools_len import product
from tqdm import tqdm
from tqdm.auto import trange
from multiprocessing import Pool, set_start_method, freeze_support
from TransitionListener import observability

class Full_calculation():
    def __init__(self, DESCRIPTION, OVERVIEW_TITLE, INPUT_NAMES, INPUT_VALS, DAISY_TYPE, STABLE_VEV, N_TRY_MAX, ARGS_SHIFT, ARG_INDEX_TO_SHIFT, SINGLE_POINT, GRID_N, SCALE, GRID_PARAMS_NAMES, GRID_PARAMS_PLOT_NAMES, X_MINMAX, Y_MINMAX, SINGLE_PROCESS, VERBOSE, NUMBER_CPUS_USED, PLOT_POTENTIAL_AT_T, PLOT_T, OVERVIEW_DETECTOR_NAME):
        t0 = time.time()

        print("\n")
        print("\t------------------------------------------------------")
        print("\t\t TransitionListener starts analysis of")
        print("\t\t\t", DESCRIPTION)
        print("\t------------------------------------------------------")
        print("\n")

        self.DESCRIPTION            = DESCRIPTION
        self.OVERVIEW_TITLE         = OVERVIEW_TITLE
        self.OUTPUT_FOLDER_NAME     = "./" + str(time.time()) + "_" + DESCRIPTION + "/"
        self.DAISY_TYPE             = DAISY_TYPE
        self.STABLE_VEV             = STABLE_VEV
        self.INPUT_NAMES            = INPUT_NAMES + ("conversionFactor", "daisyType")
        self.INPUT_VALS             = [INPUT_VALS[0], INPUT_VALS[1], STABLE_VEV, INPUT_VALS[3], INPUT_VALS[4], INPUT_VALS[2] / STABLE_VEV, DAISY_TYPE]
        self.N_TRY_MAX              = N_TRY_MAX
        self.ARGS_SHIFT             = ARGS_SHIFT
        self.ARG_INDEX_TO_SHIFT     = ARG_INDEX_TO_SHIFT
        self.SINGLE_POINT           = SINGLE_POINT
        self.GRID_N                 = GRID_N
        self.SCALE                  = SCALE
        self.GRID_PARAMS_NAMES      = GRID_PARAMS_NAMES
        self.GRID_PARAMS_PLOT_NAMES = GRID_PARAMS_PLOT_NAMES
        self.X_MINMAX               = X_MINMAX
        self.Y_MINMAX               = Y_MINMAX
        self.SINGLE_PROCESS         = SINGLE_PROCESS
        self.VERBOSE                = VERBOSE
        self.NUMBER_CPUS_USED       = NUMBER_CPUS_USED
        self.ALL_PARAMS_NAMES       = (
            "alpha",
            "alpha_inf",
            "alpha_DSnorm",
            "betaH",
            "D_SM",
            "D",
            "Tnuc_SM_GeV",
            "Tnuc_DS_GeV",
            "g_eff_rho_SM_nuc",
            "g_eff_rho_DS_nuc",
            "g_eff_rho_tot_nuc",
            "g_eff_s_SM_nuc",
            "g_eff_s_DS_nuc",
            "g_eff_s_tot_nuc",
            "dof_ratio",
            "Tcrit_DS_GeV",
            "Tf_SM_GeV",
            "m_DP_GeV",
            "m_DH_GeV",
            "Delta_m_GeV",
            "xi_chem_dec",
            "T_SM_chem_dec",
            "xi_DH",
            "xi_DP",
            "g_eff_rho_tot_DH",
            "g_eff_rho_tot_DP",
            "lambda",
            "g",
            "v_GeV",
            "xi_nuc",
            "Gamma_GeV",
            "errors",
            "warnings",
            "can_warn")
        self.ALL_PARAMS_PLOT_NAMES  = (
            R"$\alpha$",
            R"$\alpha_\infty$",
            R"$\alpha_\mathrm{DS norm}$",
            R"$\beta/H$",
            R"$D_\mathrm{SM}$",
            R"$D$",
            R"$T_\mathrm{nuc}^\mathrm{SM}$/GeV",
            R"$T_\mathrm{nuc}^\mathrm{DS}$/GeV",
            R"$g_{\mathrm{eff}, \rho}^\mathrm{SM, nuc}$",
            R"$g_{\mathrm{eff}, \rho}^\mathrm{DS, nuc}$",
            R"$g_{\mathrm{eff}, \rho}^\mathrm{tot, nuc}$",
            R"$g_{\mathrm{eff}, s}^\mathrm{SM, nuc}$",
            R"$g_{\mathrm{eff}, s}^\mathrm{DS, nuc}$",
            R"$g_{\mathrm{eff}, s}^\mathrm{tot, nuc}$",
            R"$g_{\mathrm{eff}, \rho}^\mathrm{tot, nuc}$ / ${g_{\mathrm{eff}, s}^\mathrm{tot, nuc}}^{4/3}$",
            R"$T_\mathrm{crit}^\mathrm{DS}$/GeV",
            R"$T_\mathrm{f}$ / GeV",
            R"$m_\mathrm{DP}$ / GeV",
            R"$m_\mathrm{DH}$ / GeV",
            R"$\Delta m_\mathrm{DP-DH}$ / GeV",
            R"$\xi_\mathrm{chem. dec.}$",
            R"$T_\mathrm{SM}^\mathrm{chem. dec.}$ / GeV",
            R"$\xi_\mathrm{DH nonrel}$",
            R"$\xi_\mathrm{DP nonrel}$",
            R"$g_{\mathrm{eff}, \rho}^\mathrm{tot, DH nonrel}$",
            R"$g_{\mathrm{eff}, \rho}^\mathrm{tot, DP nonrel}$",
            R"$\lambda$",
            R"$g$",
            R"$v$ / GeV",
            R"$\xi_\mathrm{nuc}$",
            R"$\Gamma$ / GeV",
            "Errors",
            "Warnings",
            "Does 3-2 scattering stop before nr transition?")
        self.PLOT_POTENTIAL_AT_T    = PLOT_POTENTIAL_AT_T
        self.PLOT_T                 = PLOT_T
        self.OVERVIEW_DETECTOR_NAME = OVERVIEW_DETECTOR_NAME
        self.GW_PARAMS_NAMES        = ("alpha", "alpha_inf", "alpha_DSnorm", "beta/H", "D", "g_eff_s_tot_nuc", "g_eff_s_tot_nuc", "Tnuc_SM_GeV")
        self.NUM_GW_PARAMS          = len(self.GW_PARAMS_NAMES)
        self.DOF_UNTIL_NUC          = 4
        self.CONST_DOF_BEFORE_NUC   = True
        self.PLOT_DILUTION          = self.SINGLE_POINT

        self.single_point_analysis() if self.SINGLE_POINT else self.grid_analysis()
        
        t1 = time.time()
        print("\nFinished in", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))

    def single_point_analysis(self):
        for _ in trange(1):
            self.All_params = self.calc_GW_params(self.INPUT_VALS)
        self.GW_params = (self.All_params[0], self.All_params[1], self.All_params[2], self.All_params[3], self.All_params[5], self.All_params[13], self.All_params[10], self.All_params[6])
        print("Phase transition analysis done. Now, check if signal is observable and save plots.")

        from pathlib import Path
        Path(self.OUTPUT_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
        file = open(self.OUTPUT_FOLDER_NAME+"0_Input_params.txt", "x")
        for n, v in zip(self.INPUT_NAMES, self.INPUT_VALS):
                file.write("{:<22} {:e}\n".format(n, v))
        file.close()

        file = open(self.OUTPUT_FOLDER_NAME+"1_All_params.txt", "x")
        for n, v in zip(self.ALL_PARAMS_NAMES, self.All_params):
                file.write("{:<22} {:e}\n".format(n, v))
        file.close()
        
        obs = observability.GW_analysis_one_point(*self.GW_params, verbose=self.VERBOSE, foldername=self.OUTPUT_FOLDER_NAME, overview_title=self.OVERVIEW_TITLE) # output folder name, overview title
        obs.calc_SNR()
        obs.save_analysis()
        obs.plot_spectrum(showplot=False)
        del obs


    def define_grid(self):
        self.x_index = self.INPUT_NAMES.index(self.GRID_PARAMS_NAMES[0])
        self.y_index = self.INPUT_NAMES.index(self.GRID_PARAMS_NAMES[1])

        def linspace(minmax):
            return np.linspace(*minmax, num=self.GRID_N, endpoint=True).tolist()

        def logspace(minmax):
            return np.logspace(*minmax, num=self.GRID_N, endpoint=True).tolist()

        if self.SCALE == "lin":
            self.x = linspace(self.X_MINMAX)
            self.y = linspace(self.Y_MINMAX)
        elif self.SCALE == "loglin":
            self.x = logspace(self.X_MINMAX)
            self.y = linspace(self.Y_MINMAX)
        elif self.SCALE == "linlog":
            self.x = linspace(self.X_MINMAX)
            self.y = logspace(self.Y_MINMAX)
        elif self.SCALE == "loglog":
            self.x = logspace(self.X_MINMAX)
            self.y = logspace(self.Y_MINMAX)

        self.GW_params = np.zeros((self.GRID_N, self.GRID_N, self.NUM_GW_PARAMS))
        self.All_params = np.zeros((self.GRID_N, self.GRID_N, len(self.ALL_PARAMS_NAMES)))
        self.x_print = np.zeros((self.GRID_N, self.GRID_N))
        self.y_print = np.zeros((self.GRID_N, self.GRID_N))

        for i in np.arange(self.GRID_N):
            for j in np.arange(self.GRID_N):
                self.x_print[i, j] = self.x[j]
                self.y_print[i, j] = self.y[i]

    def grid_analysis(self):
        self.define_grid()

        if self.SINGLE_PROCESS:
            # ... using one CPU
            for i in trange(self.GRID_N, desc="Total"):
                for j in trange(self.GRID_N, desc="Current", leave=False):
                    input_params = self.INPUT_VALS
                    input_params[self.x_index] = self.x[i]
                    input_params[self.y_index] = self.y[j]
                    resultij = self.calc_GW_params(input_params)
                    self.All_params[i,j] = resultij
                    self.GW_params[i,j] = [resultij[0], resultij[1], resultij[2], resultij[3], resultij[5], resultij[13], resultij[10], resultij[6]]
        else:
            # ... using more CPUs
            params_list = self.INPUT_VALS.copy()
            for i in range(len(self.INPUT_VALS)):
                if i == self.x_index:
                    params_list[i] = self.x
                elif i == self.y_index:
                    params_list[i] = self.y
                else:
                    params_list[i] = [self.INPUT_VALS[i]]

            input_params = product(*params_list)
            total = self.GRID_N**2
            #freeze_support()
            #set_start_method('spawn')
            pool = Pool(processes=self.NUMBER_CPUS_USED)
            results = []
            for result in tqdm(pool.imap(func=self.calc_GW_params, iterable=input_params), total=total):
                results.append(result)
            self.All_params = np.array(results).reshape(self.GRID_N, self.GRID_N, len(self.ALL_PARAMS_NAMES))
            indices = [0, 1, 2, 3, 5, 13, 10, 6]
            for k, l in enumerate(indices):
                for j in range(self.GRID_N):
                    for i in range(self.GRID_N):
                        self.GW_params[i,j,k] = self.All_params[i,j,l]

            #self.GW_params = np.array((self.All_params[:,:,0], self.All_params[:,:,1], self.All_params[:,:,2], self.All_params[:,:,4], self.All_params[:,:,12], self.All_params[:,:,9], self.All_params[:,:,5]))
        
        # Save GW params and grid
        from pathlib import Path
        Path(self.OUTPUT_FOLDER_NAME).mkdir(parents=True, exist_ok=True)

        for name, param in zip(self.ALL_PARAMS_NAMES, self.All_params.T):
            np.savetxt(self.OUTPUT_FOLDER_NAME+"Output_param_"+name+".txt", param)

        for i in range(len(self.INPUT_VALS)):
            if i == self.x_index:
                np.savetxt(self.OUTPUT_FOLDER_NAME+"Input_grid_param_x_"+self.GRID_PARAMS_NAMES[0]+".txt", self.x_print)
            elif i == self.y_index:
                np.savetxt(self.OUTPUT_FOLDER_NAME+"Input_grid_param_y_"+self.GRID_PARAMS_NAMES[1]+".txt", self.y_print)
            else:
                np.savetxt(self.OUTPUT_FOLDER_NAME+"Input_single_param_"+self.INPUT_NAMES[i]+".txt", [self.INPUT_VALS[i]])

        print("\nPhase transition analysis done. Now, check if signal is observable.")
        obs = observability.GW_analysis_grid(GW_params_array=self.GW_params, All_params_array=self.All_params, All_params_names=self.ALL_PARAMS_NAMES, All_params_plot_names=self.ALL_PARAMS_PLOT_NAMES, x=self.x, y=self.y, scale=self.SCALE, xy_plot_names=self.GRID_PARAMS_PLOT_NAMES, foldername=self.OUTPUT_FOLDER_NAME, overview_title=self.OVERVIEW_TITLE)
        obs.plot_all_params(overview_detector_name=self.OVERVIEW_DETECTOR_NAME)
        obs.save_SNRs()
        obs.plot_SNRs()

    def calc_GW_params(self, args, n_try = 0):
        import tl_dark_photon_model as model

        dp = model.dp_eft(*args, verbose=self.VERBOSE, output_folder=self.OUTPUT_FOLDER_NAME, g_DS_until_nuc=self.DOF_UNTIL_NUC, is_DS_dof_const_before_PT=self.CONST_DOF_BEFORE_NUC, dilution_plots=self.PLOT_DILUTION)
        l = dp.l
        g = dp.g
        v_GeV = dp.v_GeV
        xi_nuc = dp.xi_nuc
        Gamma_GeV = dp.Gamma_GeV
        #dp.fancy_T_plot(args[2])
        if self.PLOT_POTENTIAL_AT_T:
            dp.plot_all_contributions(0, 150, T=self.PLOT_T, subtract=True, n=500)
            #dp.plot1d(0, 150, T=self.PLOT_T, subtract=True, n=1000)
            #dp.plotPhasesPhi()
            quit()

        if n_try == 0 and self.VERBOSE:
            tqdm.write("Current point: " + str(args))

        transition = dp.calcGWTrans()
        if transition:
            if 'error' in transition:
                # Error in calcGWTrans
                if self.VERBOSE:
                    tqdm.write("Found an error of type " + str(transition['error']) + " at " + str(args))
                if n_try == self.N_TRY_MAX:
                    # Tried already N_TRY_MAX times, ended still with errors
                    if self.VERBOSE:
                        tqdm.write(str(self.N_TRY_MAX) + " tries at slightly shifted input values didn't change the problem.")
                    tqdm.write("Found final error of type " + str(transition['error']) + " at " + str(args))
                    error = transition['error']
                    alpha, alpha_inf, alpha_DSnorm, betaH, DSM, D, Tnuc_SM_GeV, Tnuc_DS_GeV, g_eff_rho_SM_nuc, g_eff_rho_DS_nuc, g_eff_rho_tot_nuc, g_eff_s_SM_nuc, g_eff_s_DS_nuc, g_eff_s_tot_nuc, dof_ratio, Tcrit_DS_GeV, Tf_SM_GeV, m_DP_GeV, m_DH_GeV, Delta_m_GeV, xi_chem_dec, T_SM_chem_dec_GeV, xi_DH, xi_DP, g_eff_rho_tot_DH, g_eff_rho_tot_DP, warning, can_warn = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    all_params = (alpha, alpha_inf, alpha_DSnorm, betaH, DSM, D, Tnuc_SM_GeV, Tnuc_DS_GeV, g_eff_rho_SM_nuc, g_eff_rho_DS_nuc, g_eff_rho_tot_nuc, g_eff_s_SM_nuc, g_eff_s_DS_nuc, g_eff_s_tot_nuc, dof_ratio, Tcrit_DS_GeV, Tf_SM_GeV, m_DP_GeV, m_DH_GeV, Delta_m_GeV, xi_chem_dec, T_SM_chem_dec_GeV, xi_DH, xi_DP, g_eff_rho_tot_DH, g_eff_rho_tot_DP, l, g, v_GeV, xi_nuc, Gamma_GeV, error, warning, can_warn)
                    del dp
                    return  all_params
                else:
                    # Changing the input parameters slighly maybe prevents the error
                    if self.VERBOSE:
                        tqdm.write("Try to solve the problem by checking for alpha and beta at slightly shifted input parameters.")

                    args_list = list(args)
                    args_list[self.ARG_INDEX_TO_SHIFT] *= 1. + self.ARGS_SHIFT
                    del dp
                    return self.calc_GW_params(args_list, n_try+1)

            # Save results of calculation
            if np.squeeze(transition[0]["trantype"]) == 1:
                alpha = np.squeeze(transition[0]["alpha"])
                alpha_inf = np.squeeze(transition[0]["alpha_inf"])
                alpha_DSnorm = np.squeeze(transition[0]["alpha_DSnorm"])
                betaH = np.squeeze(transition[0]["betaH"])
                DSM = np.squeeze(transition[0]["D_SM"])
                D = np.squeeze(transition[0]["D"])
                Tnuc_SM_GeV = np.squeeze(transition[0]["Tnuc_SM_GeV"])
                Tnuc_DS_GeV = np.squeeze(transition[0]["Tnuc_DS_GeV"])
                g_eff_rho_SM_nuc = np.squeeze(transition[0]["g_eff_rho_SM_nuc"])
                g_eff_rho_DS_nuc = np.squeeze(transition[0]["g_eff_rho_DS_nuc"])
                g_eff_rho_tot_nuc = np.squeeze(transition[0]["g_eff_rho_tot_nuc"])
                g_eff_s_SM_nuc = np.squeeze(transition[0]["g_eff_s_SM_nuc"])
                g_eff_s_DS_nuc = np.squeeze(transition[0]["g_eff_s_DS_nuc"])
                g_eff_s_tot_nuc = np.squeeze(transition[0]["g_eff_s_tot_nuc"])
                dof_ratio = g_eff_rho_tot_nuc / g_eff_s_tot_nuc**(4/3)
                Tcrit_DS_GeV = np.squeeze(transition[0]["Tcrit_DS_GeV"])
                Tf_SM_GeV = np.squeeze(transition[0]["Tf_SM_GeV"])
                m_DP_GeV = dp.mDP_GeV
                m_DH_GeV = dp.mDH_GeV
                Delta_m_GeV = m_DP_GeV - m_DH_GeV
                xi_chem_dec = np.squeeze(transition[0]["xi_chem_dec"])
                T_SM_chem_dec_GeV = np.squeeze(transition[0]["T_SM_chem_dec_GeV"])
                xi_DH = np.squeeze(transition[0]["xi_DH"])
                xi_DP = np.squeeze(transition[0]["xi_DP"])
                g_eff_rho_tot_DH = np.squeeze(transition[0]["g_eff_rho_tot_DH"])
                g_eff_rho_tot_DP = np.squeeze(transition[0]["g_eff_rho_tot_DP"])
                error = np.nan
                warning = np.squeeze(transition[0]['warning'])
                can_warn = np.squeeze(transition[0]['can_warn'])
            else:
                if self.VERBOSE:
                    tqdm.write("The transition type is not 1. Maybe a second order transition has been found.")
                error = 12
                alpha, alpha_inf, alpha_DSnorm, betaH, DSM, D, Tnuc_SM_GeV, Tnuc_DS_GeV, g_eff_rho_SM_nuc, g_eff_rho_DS_nuc, g_eff_rho_tot_nuc, g_eff_s_SM_nuc, g_eff_s_DS_nuc, g_eff_s_tot_nuc, dof_ratio, Tcrit_DS_GeV, Tf_SM_GeV, m_DP_GeV, m_DH_GeV, Delta_m_GeV, xi_chem_dec, T_SM_chem_dec_GeV, xi_DH, xi_DP, g_eff_rho_tot_DH, g_eff_rho_tot_DP, warning, can_warn = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            if VERBOSE:
                tqdm.write("A non-listed error occurred.")
            error = 13
            alpha, alpha_inf, alpha_DSnorm, betaH, DSM, D, Tnuc_SM_GeV, Tnuc_DS_GeV, g_eff_rho_SM_nuc, g_eff_rho_DS_nuc, g_eff_rho_tot_nuc, g_eff_s_SM_nuc, g_eff_s_DS_nuc, g_eff_s_tot_nuc, dof_ratio, Tcrit_DS_GeV, Tf_SM_GeV, m_DP_GeV, m_DH_GeV, Delta_m_GeV, xi_chem_dec, T_SM_chem_dec_GeV, xi_DH, xi_DP, g_eff_rho_tot_DH, g_eff_rho_tot_DP, warning, can_warn = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if self.SINGLE_POINT:
            dp.fancy_output_calcGWTrans()

        del dp
        all_params = (alpha, alpha_inf, alpha_DSnorm, betaH, DSM, D, Tnuc_SM_GeV, Tnuc_DS_GeV, g_eff_rho_SM_nuc, g_eff_rho_DS_nuc, g_eff_rho_tot_nuc, g_eff_s_SM_nuc, g_eff_s_DS_nuc, g_eff_s_tot_nuc, dof_ratio, Tcrit_DS_GeV, Tf_SM_GeV, m_DP_GeV, m_DH_GeV, Delta_m_GeV, xi_chem_dec, T_SM_chem_dec_GeV, xi_DH, xi_DP, g_eff_rho_tot_DH, g_eff_rho_tot_DP, l, g, v_GeV, xi_nuc, Gamma_GeV, error, warning, can_warn)
        return all_params