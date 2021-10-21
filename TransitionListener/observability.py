'''
The observability.py module extends the original cosmoTransitions package by a utility for calculating the spectrum of a stochastic GW background produced by a FOPT.
The calculations works as presented in 1811.11175. It is possible to perform the calculation of signal-to-noise ratios for the detector sensitivities provided in
PLI-dataset.csv. This file includes 3 classes: GW_spectrum, GW_analysis_one_point and GW_analysis_grid. The latter two are children of GW_spectrum which is used to
import the detector data. GW_analysis_one_point can calculate the GW spectrum and its detectability given a set of input parameters ("alpha", "betaH", "Tnuc_SM",
"Tnuc_DS", "g_eff_SM", "g_eff_DS", "alpha_inf", "conversionFactor"), while GW_analysis_grid instances GW_analysis in its method SNR_on_grid() for every set of input
parameters. GW_analisys_grid can be called directly from the file controlling the TransitionListener output or by importing the output afterwards, which requires
defining the path of the necessary files in this file's main() method.
'''

import numpy as np
from scipy import integrate
from scipy import interpolate
from time import time
import matplotlib.pyplot as plt
import os

class GW_spectrum:
	'''
	Class used to import the sensitivity data curves of GW observaories.

	Attributes
	----------
	self.det_f : float array
		Frequencies from the detector data
	self.det_data : float array
		Detector sensitivity curves
	self.det_names : string tuple
		Deterctor names, ordered as in input file
	self.det_colors : string tuple
		Colors for the detector sensitivities in plots, ordered as in input file
	self.det : dictionary
		Dicitonary for the sensitivitie with the detector name as key
	self.det_thr : float tuple
		Expected detector threshold SNRs
	self.det_tobs : float tuple
		Expected detector observation times
	'''
	def __init__(self):
		data_path = os.path.dirname(__file__)
		if data_path == "":
			PLI_filename = "PLI-dataset.csv"
			noise_filename = "noise-dataset.csv"
		else:
			PLI_filename = data_path+"/PLI-dataset.csv"
			noise_filename = data_path+"/noise-dataset.csv"

		self.det_PLI_f, *self.det_PLI_data = np.genfromtxt(PLI_filename, delimiter = ",", unpack = True)
		self.det_noise_f, *self.det_noise_data = np.genfromtxt(noise_filename, delimiter = ",", unpack = True)
		self.det_names = ("SKA_5_yrs","SKA_10_yrs","SKA_20_yrs","EPTA","NANOGrav","LISA","B-DECIGO","DECIGO","BBO","ET")
		self.det_single_obs_names = ("LISA","B-DECIGO","ET")
		self.det_colors = ("purple","mediumorchid","plum","darkturquoise","limegreen","blue","saddlebrown","red","orange","gray")
		self.det = dict(zip(self.det_names, self.det_noise_data))
		self.det_thr = (4., 4., 4., 1.19, 0.697, 10., 8., 10., 10., 5.)
		yr = 365.25*24*3600
		self.det_tobs = (5.*yr, 10.*yr, 20.*yr, 8.*yr, 4.*yr, 4.*yr, 4.*yr, 4.*yr, 4.*yr, 5.*yr)

	def plot_sensitivities(self, plot_PLI=True):
		fig, ax = plt.subplots(figsize=(10, 5))
		transparency = 0.4

		if plot_PLI:
			for (name, data, color) in zip(self.det_names, self.det_PLI_data, self.det_colors):
				ax.plot(self.det_PLI_f, data, color = color, label = name)
				ax.fill_between(self.det_PLI_f, data,1, alpha = transparency, color = color)
		else:
			for (name, data, color) in zip(self.det_names, self.det_noise_data, self.det_colors):
				ax.plot(self.det_noise_f, data, color = color, label = name)
				ax.fill_between(self.det_noise_f, data,1, alpha = transparency, color = color)

		ax.set_title("PLI sensitivities")
		ax.set_xscale('log')
		ax.set_xlabel(R'$f$ / Hz')
		ax.set_ylabel(R'$h^2 \Omega$')
		ax.set_yscale('log')
		ax.set_ylim(10**(-18), 10**(-3))
		ax.set_xlim(10**(-9.5), 10**(3.5))
		ax.grid()
		ax.legend(bbox_to_anchor=(1.05, 1), ncol=1)
		plt.tight_layout()
		plt.show()

class GW_analysis_one_point(GW_spectrum):
	'''
	Class used to calculate the GW spectrum of a single FOPT

	Attributes
	----------
	self.alpha : float
		Strength of the FOPT
	self.betaH : float
		Inverse time scale of the FOPT, normalized to Hubble parameter at nucleation
	self.Tnuc_SM : float
		Photon bath temperature at the nucleation of the FOPT
	self.Tnuc_dS : float
		Temperature within the dark sector extending the SM at the nucleation of the FOPT
	self.g_eff_SM : float
		Effective, relativistiv dof in the SM at the nucleation of the FOPT
	self.g_eff_DS : float
		Effective, relativistic dof in dark sector at the nucleation of the FOPT
	self.alpha_inf : float
		Maximum value for alpha for which non-runaway bubbles are expected
	self.conversionFactor : float
		Factor between the energy unit used for the vev and temperatures in TransitionListener and 1 GeV. E.g. 1e-6 for keV.
	self.eps_turb : float
		Percentage of sound-wave energy that is transformed into MHD turbulance that produces GW contributions
	self.v_wall : float
		The bubble wall velocity in units of the speed of light in vacuum
	self.verbose : bool
		Switch for terminal outputs
	self.xi_nuc : float
		Ratio of self.Tnuc_DS and self.Tnuc_SM
	self.g_eff_tot : float
		Sum of self.g_eff_SM and self.g_eff_DS. Total effective, relativistic dof at the nucleation of the FOPT
	self.g_eff_tot_s : float
		Total entropic, effective, relativistic dof at the nucleation of the FOPT
	self.g_eff_tot_rho : float
		Total "energetic" (~T**4), effective, relativistic dof at the nucleation of the FOPT
	self.g_eff_tot_s_eq : float
		Total entropic, effective, relativistic dof at the matter-radiation equality
	self.rho_rad : float
		The radiative energy density in the GeV**4
	self.Hubble_Hz : float
		The Hubble parameter at nucleation in Hz
	self.beta_Hz : float
		The beta parameter in Hz
	self.is_runaway : bool
		Switch for bubble wall runaway case
	self.kappa_phi : float
		The ratio of scalar field gradient energy used to produce GW and the latent energy set free during the FOPT
	self.kappa_sw : float
		The ratio of sound wave energy used to produce GW and the latent energy set free during the FOPT
	self.kappa_turb : float
		The ratio of MHD turbulance energy used to produce GW and the latent energy set free during the FOPT
	self.sources : string tuple
		A list of the GW production sources: 'scalar field', 'sound waves', 'turbulence'
	self.GW_import_params_names: string tuple
		A list of the GW parameters that have to be importet: "alpha", "betaH", "Tnuc_SM", "Tnuc_DS", "g_eff_SM",
		"g_eff_DS", "alpha_inf", "conversionFactor"
	self.GW_params_vals : tuple
		A list of the respective variables whose names are listed in self.GW_params_names.
	self.GW_params_names : string tuple
		A list of all the parameters that are used to calculate the GW spectrum: "alpha", "betaH", "Tnuc_SM",
		"Tnuc_DS", "xi_nuc", "g_eff_SM", "g_eff_DS", "g_eff_tot", "g_eff_tot_s", "g_eff_tot_s_eq", "g_eff_tot_rho",
		"conversionFactor", "eps_turb", "v_wall", "rho_rad", "Hubble_Hz", "beta_Hz", "alpha_inf", "is_runaway",
		"kappa_phi", "kappa_sw", "kappa_turb"
	self.GW_params : dictionary
		Dictionary to make an element from self.GW_param_vals callable by its name set in self.GW_params_names
	'''
	def __init__(self, alpha=1, alpha_inf=.3, alpha_DSnorm=1, betaH=100, D=1, g_eff_s_tot_nuc=100, g_eff_rho_tot_nuc=100, Tnuc_SM_GeV=100, g_eff_s_SM_0=3.9309363, eps_turb=.1, v_wall=1, verbose=False, foldername="", overview_title=""):
		if verbose:
			print("\n")
			print("   ----------------------------")
			print("       GW Spectrum Analysis    ")
			print("   ----------------------------")
			print("\n")

		super(GW_analysis_one_point, self).__init__()
		self.T0_SM_GeV = 2.35253655e-13 # 2.73 * 8.61735e-5 * 1e9
		self.m_Planck_GeV = 1.220910e19 # m_planck = .... GeV. (Note: not the 8 pi G convention!)
		self.GeV_Hz = 1./6.582119e-25 # 1 GeV = ... Hz
		self.sources = ('scalar field', 'sound waves', 'turbulence')

		self.alpha = alpha
		self.alpha_inf = alpha_inf
		self.alpha_DSnorm = alpha_DSnorm
		self.betaH = betaH
		self.D = D
		self.g_eff_s_tot_nuc = g_eff_s_tot_nuc
		self.g_eff_rho_tot_nuc = g_eff_rho_tot_nuc
		self.Tnuc_SM_GeV = Tnuc_SM_GeV
		self.g_eff_s_SM_0 = g_eff_s_SM_0
		self.eps_turb = eps_turb
		self.v_wall = v_wall
		self.verbose = verbose

		self.rho_rad_GeV4 = np.pi**2. / 30. * self.g_eff_rho_tot_nuc * self.Tnuc_SM_GeV**4.
		self.Hubble_Hz = self.calc_Hubble_Hz() # Hubble parameter at nucleation in Hz
		self.beta_Hz = self.betaH * self.Hubble_Hz # Inverse timescale of PT in Hz
		self.is_runaway = self.is_runaway()
		self.kappa_phi = self.calc_kappa_phi()
		self.kappa_sw = self.calc_kappa_sw()
		self.kappa_turb = self.calc_kappa_turb()

		self.overview_title = overview_title
		self.foldername = foldername

		if verbose:
			self.print_params()

	def print_params(self):
		print("\n\nFULL GW PARAMETERS:\n")
		vals = (self.alpha, self.alpha_inf, self.alpha_DSnorm, self.betaH, self.D, self.g_eff_s_tot_nuc, self.g_eff_rho_tot_nuc, self.Tnuc_SM_GeV, self.g_eff_s_SM_0, self.eps_turb, self.v_wall)
		names = ("alpha", "alpha_inf", "alpha_DSnorm", "beta/H", "D", "g_eff_s_tot_nuc", "g_eff_s_tot_nuc", "Tnuc_SM_GeV", "g_eff_s_SM_0", "eps_turb", "v_wall")
		dic = dict(zip(names, vals))
		for n, v in dic.items():
			print("{:<22} {:<25}".format(n, v))

	def calc_Hubble_Hz(self):
		# Hubble constant in units of Hz
		np.seterr(invalid="ignore")
		H_Hz = np.sqrt(self.rho_rad_GeV4)
		np.seterr(invalid="warn")
		H_Hz *= np.sqrt(8. * np.pi / 3.) / self.m_Planck_GeV
		H_Hz *= self.GeV_Hz
		return H_Hz

	def is_runaway(self):
		if self.alpha_DSnorm > self.alpha_inf:
			return True
		else:
			return False

	def calc_kappa_phi(self):
		if self.is_runaway:
			return 1 - self.alpha_inf/self.alpha_DSnorm
		else:
			return 0.

	def calc_kappa_sw(self):
		def kappa(a):
			np.seterr(invalid="ignore")
			kappa = a/(0.73 + 0.083*np.sqrt(a) + a)
			np.seterr(invalid="warn")
			return kappa 
		if self.is_runaway:
			return self.alpha_inf / self.alpha_DSnorm * kappa(self.alpha_inf)
		else:
			return kappa(self.alpha_DSnorm)

	def calc_kappa_turb(self):
		return self.eps_turb*self.kappa_sw

	def normalization(self, source):
		norm = {'scalar field': 1., 'sound waves': 1.59e-1, 'turbulence': 2.01e1}
		return norm[source]

	def efficiency_kappa(self, source):
		kappa = {'scalar field': self.kappa_phi, 'sound waves': self.kappa_sw, 'turbulence': self.kappa_turb}
		return kappa[source]

	def exponent_p(self, source):
		p = {'scalar field': 2. ,'sound waves': 2., 'turbulence': 3./2.}
		return p[source]

	def exponent_q(self, source):
		q = {'scalar field': 2. ,'sound waves': 1., 'turbulence': 1.}
		return q[source]

	def vel_factor_Delta(self, source):
		v = self.v_wall
		Delta = {'scalar field': 0.11 * v**3. / (0.42 + v**2), 'sound waves': v, 'turbulence': v}
		return Delta[source]

	def peak_frequency(self, source):
		v = self.v_wall
		b_Hz = self.beta_Hz
		fp_Hz = {'scalar field': 0.62 * b_Hz / (1.8 - 0.1 * v + v**2.), 'sound waves': 2. * b_Hz / np.sqrt(3.) / v, 'turbulence': 3.5 * b_Hz / 2. / v}
		return fp_Hz[source]

	def spectral_shape(self, source, frequency_Hz):
		f_Hz = frequency_Hz
		H_Hz = self.Hubble_Hz
		fp_Hz = self.peak_frequency(source)
		f_fp = f_Hz / fp_Hz
		sf = {'scalar field': 3.8 * f_fp**2.8 / (1 + 2.8 * f_fp**3.8) , 'sound waves': f_fp**3. * (7. / (4. + 3. * f_fp**2))**3.5, 'turbulence': f_fp**3. / (1. + f_fp)**(11./3.) / (1. + 8. * np.pi * f_Hz / H_Hz)}
		return sf[source]

	def Omega(self, source, frequency_Hz):
		f_Hz = frequency_Hz
		a = self.alpha
		Hb = 1. / self.betaH

		norm = self.normalization(source)
		vel_Delta = self.vel_factor_Delta(source)
		efficiency = (self.efficiency_kappa(source) * a / (1. + a))**self.exponent_p(source)
		timescale = Hb**self.exponent_q(source)     
		shape = self.spectral_shape(source, f_Hz)

		Omega = norm * vel_Delta * efficiency * timescale * shape
		return Omega

	def h2Omega_0(self, source, frequency_0_Hz):
		g_eff_rho_tot_nuc = self.g_eff_rho_tot_nuc
		g_eff_s_tot_nuc = self.g_eff_s_tot_nuc
		g_eff_s_SM_0 = self.g_eff_s_SM_0
		D = self.D
		Tnuc_SM_GeV = self.Tnuc_SM_GeV
		T0_SM_GeV = self.T0_SM_GeV

		np.seterr(invalid="ignore")
		inv_redshift = D**(1/3) * (g_eff_s_tot_nuc / g_eff_s_SM_0)**(1./3.) * Tnuc_SM_GeV / T0_SM_GeV
		Rh2 = 2.473e-5 / D**(4./3.) * (g_eff_s_SM_0 / g_eff_s_tot_nuc)**(4./3.) * g_eff_rho_tot_nuc / 2.
		np.seterr(invalid="warn")

		frequenciy_nuc_Hz = inv_redshift * frequency_0_Hz
		h2Omega_0 = Rh2 * self.Omega(source, frequenciy_nuc_Hz)
		return h2Omega_0


	def h2Omega_0_sum(self, frequency_Hz):
		h2Omega_0_sum = 0
		for source in self.sources:
			h2Omega_0_sum += self.h2Omega_0(source, frequency_Hz)
		return h2Omega_0_sum


	def plot_spectrum(self, showplot=True):
		fig, ax = plt.subplots(figsize=(10, 5))
		transparency = 0.4

		frequencies_Hz = np.logspace(-9.5, 3.5, 1000) # Hz
		for source in self.sources:
			ax.plot(frequencies_Hz, self.h2Omega_0(source, frequencies_Hz), label = source)
		ax.plot(frequencies_Hz, self.h2Omega_0_sum(frequencies_Hz), label = "Sum", color = "black")    

		for (name, data, color) in zip(self.det_names, self.det_PLI_data, self.det_colors):
			ax.plot(self.det_PLI_f, data, color = color, label = name)
			ax.fill_between(self.det_PLI_f, data,1, alpha = transparency, color = color)

		if self.overview_title == "":
			ax.set_title("GW spectrum and PLI sensitivities")
		else:
			ax.set_title(self.overview_title)
		ax.set_xscale('log')
		ax.set_xlabel(R'$f$ / Hz')
		ax.set_ylabel(R'$h^2 \Omega_\mathrm{GW} (f)$')
		ax.set_yscale('log')
		ax.set_ylim(10**(-25), 10**(-3))
		ax.set_xlim(10**(-9.5), 10**(3.5))
		ax.grid()
		ax.legend(bbox_to_anchor=(1.05, 1), ncol=1)
		plt.tight_layout()
		if showplot:
			plt.show()
		else:
			from pathlib import Path
			Path(self.foldername).mkdir(parents=True, exist_ok=True)
			plt.savefig(self.foldername+"GW_spectrum.pdf")
		plt.close(fig)

	def plot_spectrum_minimal(self):
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		fig, ax = plt.subplots(figsize=(8, 7))
		transparency = 0.4

		frequencies_Hz = np.logspace(-9.5, 3.5, 1000) # Hz
		source_names = ["Bubble collisions", "Sound waves", "MHD turbulence"]
		for i, source in enumerate(self.sources):
			ax.plot(frequencies_Hz, self.h2Omega_0(source, frequencies_Hz), label = source_names[i])
		ax.plot(frequencies_Hz, self.h2Omega_0_sum(frequencies_Hz), label = "Total signal", color = "black")    

		for (name, data, color) in zip(self.det_names, self.det_PLI_data, self.det_colors):
			ax.plot(self.det_PLI_f, data, color = color)
			ax.fill_between(self.det_PLI_f, data,1, alpha = transparency, color = color)

		ax.text(1e-7, 1e-9, "SKA", size = 16, color = "purple")
		ax.text(1e-7, 4e-6, "NanoGrav", size = 16, color = "limegreen")
		ax.text(1.5e-7, 2e-4, "EPTA", size = 16, color = "darkturquoise")
		ax.text(1e-4, 1e-4, "LISA", size = 16, color = "blue")
		ax.text(3e-2, 1e-11, "B-DECIGO", size = 16, color = "saddlebrown")
		ax.text(3e-2, 1e-14, "DECIGO", size = 16, color = "red")
		ax.text(1e0, 1e-16, "BBO", size = 16, color = "orange")
		ax.text(1e2, 1e-10, "ET", size = 16, color = "gray")
		ax.set_xscale('log')
		ax.set_xlabel(R'Frequency / Hz', size = 16)
		ax.set_ylabel(R'Sensitivity, signal $h^2 \Omega(f)$', size = 16)
		ax.set_yscale('log')
		ax.tick_params(axis = "x", labelsize = 16)
		ax.tick_params(axis = "y", labelsize = 16)
		ax.set_ylim(10**(-18), 10**(-3))
		ax.set_xlim(10**(-9.5), 10**(3.5))
		ax.legend(loc = 1, fontsize = 16)
		plt.tight_layout()
		plt.savefig("observability.pdf")
		plt.close(fig)

	def calc_SNR(self):
		def calc_SNR_for_det(detector_name):
			detector_index = self.det_names.index(detector_name)
			sensitivity = self.det_noise_data[detector_index]
			frac = np.array((self.h2Omega_0_sum(self.det_noise_f)/sensitivity)**2)
			integral = integrate.simps(frac[1:], self.det_noise_f[1:])
			tobs = self.det_tobs[detector_index]
			snr2 = 2. * tobs * integral
			if detector_name in self.det_single_obs_names:
				snr2 /= 2.
			return np.sqrt(snr2)

		SNR = np.zeros(len(self.det_names))
		detectable = np.zeros(len(self.det_names))
		for i in range(len(self.det_names)):
			SNR[i] = calc_SNR_for_det(self.det_names[i])
			log_thr = np.log10(self.det_thr[i])
			detectable[i] = True if np.log10(SNR[i]) > log_thr else False

		if self.verbose:
			print("\n\nSNR VALUES AND OBSERVABILITY:\n")
			for n, v, d in zip(self.det_names, SNR, detectable):
				print ("{:<22} {:<25} {}".format(n, "{:e}".format(v), d))

		self.SNR = SNR
		self.detectable = detectable
		return SNR

	def save_analysis(self):
		from pathlib import Path
		Path(self.foldername).mkdir(parents=True, exist_ok=True)
		file = open(self.foldername+"2_GW_params.txt", "x")
		vals = (self.alpha, self.alpha_inf, self.alpha_DSnorm, self.betaH, self.D, self.g_eff_s_tot_nuc, self.g_eff_rho_tot_nuc, self.Tnuc_SM_GeV, self.g_eff_s_SM_0, self.eps_turb, self.v_wall)
		names = ("alpha", "alpha_inf", "alpha_DSnorm", "beta/H", "D", "g_eff_s_tot_nuc", "g_eff_s_tot_nuc", "Tnuc_SM_GeV", "g_eff_s_SM_0", "eps_turb", "v_wall")
		dic = dict(zip(names, vals))
		for n, v in dic.items():
			file.write("{:<22} {:<25} \n".format(n, v))
		file.close()
		file = open(self.foldername+"3_SNRs.txt", "x")
		for n, v, d in zip(self.det_names, self.SNR, self.detectable):
			file.write("{:<22} {:<25} {} \n".format(n, "{:e}".format(v), d))
		file.close()

class GW_analysis_grid(GW_spectrum):
	'''
	Class to loop thorugh an array of input parameters for GW_analysis_single_point to calculate and plot SNR on the given grid. Includes also a method to print all
	input parameters on the the imported grid.

	Attributes
	----------
	self.foldername : string
		The folder where the resulting plots should be saved to
	self.GW_params_array : 3d numpy array of size (N,N,M)
		Array in which all GW_params are saved. The grid has a side length of N and overall M parameters are imported.
	self.GW_params_names : string tuple
		List of length M of the names of the imported GW parameters
	self.GW_params_plot_names : string tuple
		Similar to self.GW_params_names, but can contain matplotlib-LaTeX code for fancier descriptions in plots
	self.scale : string
		The scale of the x and y axis: "lin" for both linear, "linlog" for a log-scaled y-axis, "loglin" for a log-scaled
		x-axis, "loglog" for both axes log-scaled
	self.xy_plot_names : string tuple
		The variable names on the x- and y-axes. Can contain matplotlib-LaTeX.
	self.gridsize : int
		The side length N of the grid on which the SNRs should be calculated
	self.SNR : 3d numpy array of size (N,N,L)
		SNR values for each point in the grid (N,N) and for each of the L detectors.
	'''
	def __init__(self, GW_params_array, All_params_array, All_params_names, All_params_plot_names, x, y, scale, xy_plot_names, foldername, overview_title):
		super(GW_analysis_grid, self).__init__()
		self.foldername = foldername
		self.GW_params_array = GW_params_array
		self.All_params_array = All_params_array
		self.All_params_names = All_params_names
		self.All_params_plot_names = All_params_plot_names
		self.scale = scale
		self.xy_plot_names = xy_plot_names
		self.overview_title = overview_title
		if self.scale == "lin":
			self.x = np.append(x, 2*x[-1] - x[-2])
			self.y = np.append(y, 2*y[-1] - y[-2])
		elif self.scale == "linlog":
			self.x = np.append(x, 2*x[-1] - x[-2])
			self.y = np.append(y, y[-1] + 10**(np.log10(y[-1]) - np.log10(y[0])) * (y[1] - y[0]))
		elif self.scale == "loglin":
			self.x = np.append(x, x[-1] + 10**(np.log10(x[-1]) - np.log10(x[0])) * (x[1] - x[0]))
			self.y = np.append(y, 2*y[-1] - y[-2])
		elif self.scale == "loglog":
			self.x = np.append(x, x[-1] + 10**(np.log10(x[-1]) - np.log10(x[0])) * (x[1] - x[0]))
			self.y = np.append(y, y[-1] + 10**(np.log10(y[-1]) - np.log10(y[0])) * (y[1] - y[0]))
		self.gridsize = len(x)
		self.SNR = self.SNR_on_grid()

	def SNR_on_grid(self):
		n = self.gridsize
		SNR = np.zeros((n,n,len(self.det_names)))

		for i in range(n):
			for j in range(n):
				one_point_analysis = GW_analysis_one_point(*self.GW_params_array[i,j])
				SNR[i,j] = one_point_analysis.calc_SNR()
		return SNR

	def plot_SNR(self, detector_name, unvis_is_dark=True):
		detector_index = self.det_names.index(detector_name)
		detector_sensitivity = self.det_noise_data[detector_index]
		logSNR = np.log10(self.SNR[:,:,detector_index])
		blackdots = np.full_like(logSNR,1)

		if unvis_is_dark:
			log_thr = np.log10(self.det_thr[detector_index])
			blackdots[logSNR<log_thr] = 0
			logSNR[logSNR<log_thr] = np.nan

		fig, ax = plt.subplots()
		if unvis_is_dark:
			ax.pcolormesh(self.x, self.y, blackdots.T, cmap="gist_gray")
		im = ax.pcolormesh(self.x, self.y, logSNR.T)

		fig.colorbar(im, ax=ax)
		ax.set_title(R"$\mathrm{log}_{10}$ SNR for "+detector_name)
		ax.set_xlabel(self.xy_plot_names[0])
		ax.set_ylabel(self.xy_plot_names[1])
		if self.scale == 'linlog':
			ax.set_yscale('log')
		if self.scale == 'loglin':
			ax.set_xscale('log')
		if self.scale == 'loglog':
			ax.set_xscale('log')
			ax.set_yscale('log')
		plt.savefig(self.foldername+"SNR_"+detector_name+".pdf")
		plt.close(fig)

	def plot_SNRs(self):
		for dn in self.det_names:
			self.plot_SNR(dn)

	def save_SNRs(self):
		for dn in self.det_names:
			detector_index = self.det_names.index(dn)
			np.seterr(invalid="ignore")
			logSNR = np.log10(self.SNR[:,:,detector_index])
			np.seterr(invalid="warn")
			np.savetxt(self.foldername+"log_SNR_"+dn+".txt", logSNR)

	def plot_all_params(self, overview_detector_name="LISA"):
		log_params = dict()
		lin_params = dict()

		for name, param in zip(self.All_params_names, self.All_params_array.T):
			np.seterr(invalid="ignore", divide="ignore")
			log_params[name] = np.log10(param)
			lin_params[name] = param
			np.seterr(invalid="warn", divide="warn")

		# Define parameters to plot (3 GW params + 1 SNR)
		param_names = ("alpha", "Tnuc_SM_GeV", "D")
		param_plot_names = (R"$\mathrm{log}_{10}$ $\alpha$", R"$\mathrm{log}_{10}$ $T_\mathrm{nuc}^\mathrm{SM}$ / GeV", R"$\mathrm{log}_{10}$ $D$")
		SNR_name = (overview_detector_name)
		SNR_plot_name = (R"$\mathrm{log}_{10}$ $\mathrm{SNR}$ for " + overview_detector_name)

		# prepare 3 params
		log_overview_params = dict()
		for name in param_names:
			log_overview_params[name] = log_params[name]

		# prepare SNR
		detector_index = self.det_names.index(SNR_name)
		detector_sensitivity = self.det_noise_data[detector_index]
		logSNR = np.log10(self.SNR[:,:,detector_index])
		log_thr = np.log10(self.det_thr[detector_index])
		blackdots = np.full_like(logSNR,1)
		greydots = np.full_like(logSNR,1)
		blackdots[logSNR<log_thr] = 0
		for i in range(len(lin_params['can_warn'])):
			for j in range(len(lin_params['can_warn'])):
				if lin_params['can_warn'][j,i] == False:
					greydots[j,i] = np.nan
		logSNR[logSNR<log_thr] = np.nan

		# Prepare axes
		fig, axs = plt.subplots(2,2)
		ax1 = axs[0,0]
		ax2 = axs[0,1]
		ax3 = axs[1,0]
		ax4 = axs[1,1]
		for ax in (ax1, ax2, ax3, ax4):
			ax.set_xlabel(self.xy_plot_names[0])
			ax.set_ylabel(self.xy_plot_names[1])
			if self.scale == 'linlog':
				ax.set_yscale('log')
			if self.scale == 'loglin':
				ax.set_xscale('log')
			if self.scale == 'loglog':
				ax.set_xscale('log')
				ax.set_yscale('log')

		# Plot 3 params
		GW_axes = (ax1, ax2, ax3)
		for i, ax in enumerate(GW_axes):
			im = GW_axes[i].pcolormesh(self.x, self.y, log_overview_params[param_names[i]])
			fig.colorbar(im, ax=GW_axes[i])
			GW_axes[i].set_title(param_plot_names[i])

		# Plot SNR
		ax4.set_title(SNR_plot_name)
		ax4.pcolormesh(self.x, self.y, blackdots.T, cmap="gist_gray")
		im4 = ax4.pcolormesh(self.x, self.y, logSNR.T)
		ax4.pcolormesh(self.x, self.y, greydots, cmap="binary", alpha=0.5)
		fig.colorbar(im4, ax=ax4)
		fig.suptitle(self.overview_title) 
		plt.tight_layout()

		# Savefig
		plt.savefig(self.foldername+"overview_plot.pdf")
		plt.close(fig)

		# Log plots
		for name, plotname in zip(self.All_params_names, self.All_params_plot_names):
			fig, ax = plt.subplots()
			im = ax.pcolormesh(self.x, self.y, log_params[name])
			fig.colorbar(im, ax=ax)
			ax.set_title(R"$\mathrm{log}_{10}$ "+plotname)
			ax.set_xlabel(self.xy_plot_names[0])
			ax.set_ylabel(self.xy_plot_names[1])
			if self.scale == 'linlog':
				ax.set_yscale('log')
			if self.scale == 'loglin':
				ax.set_xscale('log')
			if self.scale == 'loglog':
				ax.set_xscale('log')
				ax.set_yscale('log')
			plt.savefig(self.foldername+"log_plot_"+name+".pdf")
			plt.close(fig)

		# Lin plots
		for name, plotname in zip(self.All_params_names, self.All_params_plot_names):
			fig, ax = plt.subplots()
			im = ax.pcolormesh(self.x, self.y, lin_params[name])
			fig.colorbar(im, ax=ax)
			ax.set_title(plotname)
			ax.set_xlabel(self.xy_plot_names[0])
			ax.set_ylabel(self.xy_plot_names[1])
			if self.scale == 'linlog':
				ax.set_yscale('log')
			if self.scale == 'loglin':
				ax.set_xscale('log')
			if self.scale == 'loglog':
				ax.set_xscale('log')
				ax.set_yscale('log')
			plt.savefig(self.foldername+"lin_plot_"+name+".pdf")
			plt.close(fig)


class GW_spectra_comparison(GW_spectrum):
	def plot_N_spectra(self, GW_params, labels, linestyles, colors, alphas, title, outputname, N=1, annotations=False, second_annotation=False, xlim=(10**(-5), 2e-1), ylim=(10**(-15), 10**(-10))):
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		transparency = 0.2

		fig, ax = plt.subplots(figsize=(7.4, 4.8))
		frequencies_Hz = np.logspace(-9.5, 5, 1000) # Hz

		for (name, data, color) in zip(self.det_names, self.det_PLI_data, self.det_colors):
			if name != "LISA":
			#if name != "ET":
				continue
			else:
				color = "gray"
			ax.plot(self.det_PLI_f, data, color = color, alpha = 0.3)
			ax.fill_between(self.det_PLI_f, data, 1, alpha = transparency, color = color)

		N = len(labels)

		for i in range(N):
			print(i, colors[i], labels[i], alphas[i], linestyles[i])
			analysis = GW_analysis_one_point(*GW_params[i,:])
			signal = analysis.h2Omega_0_sum(frequencies_Hz)
			ax.plot(frequencies_Hz, signal, color=colors[i], label=labels[i], alpha=alphas[i], linestyle=linestyles[i])

		ax.set_title(title, size=16)
		ax.set_xscale('log')
		ax.set_xlabel(R'Signal frequency $f$ / Hz', size = 16)
		ax.set_ylabel(R'Signal strength $h^2 \Omega_\mathrm{GW}(f)$', size = 16)
		ax.set_yscale('log')
		ax.tick_params(axis = "x", labelsize = 14)
		ax.tick_params(axis = "y", labelsize = 14)
		ax.set_ylim(ylim)
		ax.set_xlim(xlim)

		if annotations:
			ax.annotate("", xy=(3e-3, 1e-11), xycoords='data', xytext=(2.4e-2, 2e-14), textcoords='data', arrowprops=dict(arrowstyle="simple,tail_width=0.1,head_width=1,head_length=1", connectionstyle="arc3,rad=.3", color="darkred"))
			bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
			ax.text(1e-2, 2.2e-12, "Temperature\nratio", ha="center", va="center", size=12, bbox=bbox_props, color="darkred")
		if second_annotation:
			ax.annotate("", xy=(1.5e-5, 2e-14), xycoords='data', xytext=(1.5e-4, 1e-11), textcoords='data', arrowprops=dict(arrowstyle="simple,tail_width=0.1,head_width=1,head_length=1", connectionstyle="arc3", color="lightcoral"))
			ax.text(6e-5, 1e-12, "Mediator\nlifetime", ha="center", va="center", size=12, bbox=bbox_props, color="lightcoral")
		ax.legend(fontsize = 10)
		plt.tight_layout()
		plt.savefig(str(time()) + "_" + outputname + ".pdf")
		plt.show()
		plt.close(fig)



def main():
	pass

if __name__== "__main__":
	main()