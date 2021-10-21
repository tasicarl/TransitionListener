'''
Script to calculate the effective energetic (rho) and entropic (s) degrees of freedom of the Standard Model (SM) and an additional dark sector (DS)
at a given temperature T in GeV * conversionFactor. Between the DS and the SM bath is a temperature ratio of xi.

The calculation is taken from 1609.04979, while the SM g_eff data is taken from https://member.ipmu.jp/satoshi.shirai/EOS2018 (1803.01038)
'''

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.integrate as integrate

class effective_dof():
	def __init__(self):
		# Imports SM data. Note: T_SM_sup corresponds in any case to GeV
		try:
			T, grho, grho_err, gs, gs_err = np.genfromtxt("TransitionListener/geff_SM2.dat", unpack=True)
		except:
			T, grho, grho_err, gs, gs_err = np.genfromtxt("geff_SM2.dat", unpack=True)
		self.T_SM_sup = T
		self.g_eff_rho_SM_sup = grho
		self.g_eff_s_SM_sup = gs
		self.g_eff_rho_SM_interpolation = interpolate.interp1d(self.T_SM_sup, self.g_eff_rho_SM_sup, bounds_error = False, fill_value = (self.g_eff_rho_SM_sup[0], self.g_eff_rho_SM_sup[-1]))
		self.g_eff_s_SM_interpolation = interpolate.interp1d(self.T_SM_sup, self.g_eff_s_SM_sup, bounds_error = False, fill_value = (self.g_eff_s_SM_sup[0], self.g_eff_s_SM_sup[-1]))

	def calc_g_eff_rho_tot(self, bosons, fermions, T_SM, conversionFactor, xi):
		return self.calc_g_eff_rho_SM(T_SM, conversionFactor) + self.calc_g_eff_rho_DS(bosons, fermions, T_SM * xi) * xi**4

	def calc_g_eff_rho_SM(self, T_SM, conversionFactor):
		return self.g_eff_rho_SM_interpolation(T_SM * conversionFactor)

	def calc_g_eff_rho_DS(self, bosons, fermions, T_DS):
		def boson_integration(m2, T_DS, is_physical):
			nbosons = np.squeeze(m2.shape)
			b = np.zeros(nbosons)
			for i in np.arange(nbosons):
				if is_physical[i]:
					z2 = m2[i] / (T_DS+1e-100)**2.
					if z2 > 0:
						z = np.sqrt(z2) + 1e-100
						b[i] = integrate.quad(lambda u: u**2 * (u**2 - z2)**(1/2) / (np.exp(u) - 1), z, np.inf)[0]
					else:
						#print("Found case of imaginary z = m / T_DS")
						#print("m2 = ", m2[i])
						#print("T_DS = ", T_DS)
						b[i] = np.nan
			return b

		def fermion_integration(m2, T_DS):
			nfermions = np.squeeze(m2.shape)
			f = np.zeros(nfermions)
			for i in np.arange(nfermions):
				z2 = m2[i] / (T_DS+1e-100)**2.
				z2 = 0 if z2 < 0 else z2
				f[i] = integrate.quad(lambda u: u**2 * (u**2 - z2)**(1/2) / (np.exp(u) + 1), np.sqrt(z2), np.inf)[0]
			return f

		m2, g, c, is_physical = bosons
		np.seterr(over = "ignore")
		g_eff_rho = np.sum(g * boson_integration(m2, T_DS, is_physical), axis=-1)
		np.seterr(over = "warn")

		m2, g = fermions
		if m2 != 0:
			np.seterr(over = "ignore")
			g_eff_rho += np.sum(g * fermion_integration(m2, T_DS), axis=-1)
			np.seterr(over = "warn")
		return 15 / np.pi**4. * g_eff_rho

	def calc_g_eff_s_tot(self, bosons, fermions, T_SM, conversionFactor, xi):
		return self.calc_g_eff_s_SM(T_SM, conversionFactor) + self.calc_g_eff_s_DS(bosons, fermions, T_SM * xi) * xi**3

	def calc_g_eff_s_SM(self, T_SM, conversionFactor):
		return self.g_eff_s_SM_interpolation(T_SM*conversionFactor)

	def calc_g_eff_s_DS(self, bosons, fermions, T_DS):
		return (3 * self.calc_g_eff_rho_DS(bosons, fermions, T_DS) + self.calc_g_eff_P_DS(bosons, fermions, T_DS))/4

	def calc_g_eff_P_DS(self, bosons, fermions, T_DS):
		def boson_integration(m2, T_DS, is_physical):
			nbosons = np.squeeze(m2.shape)
			b = np.zeros(nbosons)
			for i in np.arange(nbosons):
				if is_physical[i]:
					z2 = m2[i] / (T_DS+1e-100)**2.
					b[i] = integrate.quad(lambda u: (u**2 - z2)**(3/2) / (np.exp(u) - 1), np.sqrt(z2), np.inf)[0]
			return b

		def fermion_integration(m2, T_DS):
			nfermions = np.squeeze(m2.shape)
			f = np.zeros(nfermions)
			for i in np.arange(nfermions):
				z2 = m2[i] / (T_DS+1e-100)**2.
				z2 = 0 if z2 < 0 else z2
				f[i] = integrate.quad(lambda u: (u**2 - z2)**(3/2) / (np.exp(u) + 1), np.sqrt(z2), np.inf)[0]
			return f

		m2, g, c, is_physical = bosons
		np.seterr(over = "ignore")
		g_eff_P = np.sum(g * boson_integration(m2, T_DS, is_physical), axis=-1)
		np.seterr(over = "warn")

		m2, g = fermions
		if m2 != 0:
			np.seterr(over = "ignore")
			g_eff_P += np.sum(g * fermion_integration(m2, T_DS), axis=-1)
			np.seterr(over = "warn")
		return 15 / np.pi**4. * g_eff_P
		
	r'''
	def plot_SMDS_g_eff_rho_and_s(self, Tmin_GeV=self.T_SM_sup[0], Tmax_GeV=self.T_SM_sup[-1]):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10,3])
		Trange = np.logspace(np.log10(Tmin_GeV), np.log10(Tmax_GeV), num = 100, endpoint = True)
		plt.plot(self.T_SM_sup, self.g_eff_rho_SM_sup, "x", label = R"$g_{\mathrm{eff}, \rho}^\mathrm{SM, sup}$")
		plt.plot(Trange, self.g_eff_rho_SM_interpolation(Trange), label = R"$g_{\mathrm{eff}, \rho}^\mathrm{SM}$")
		plt.plot(self.T_SM_sup, self.g_eff_s_SM_sup, "x", label = R"$g_{\mathrm{eff}, s}^\mathrm{SM, sup}$")
		plt.plot(Trange, self.g_eff_s_SM_interpolation(Trange), label = R"$g_{\mathrm{eff}, s}^\mathrm{SM}$")
		plt.xlabel("T / GeV")
		plt.xscale("log")
		plt.yscale("log")
		plt.legend()
		plt.show()


	def plot_DS_g_eff_rho_and_s(self, bosons, fermions, conversionFactor):
		num = 100
		Trange = np.logspace(np.log10(1e-7 / conversionFactor), np.log10(1e-3 / conversionFactor), num=num, endpoint=True)
		add_geff_rho = np.zeros(num)
		add_geff_s = np.zeros(num)
		for n,T in zip(range(num), Trange):
			add_geff_rho[n] = self.calc_g_eff_rho_DS(bosons, fermions, Trange[n])
			add_geff_s[n] = self.calc_g_eff_s_DS(bosons, fermions, Trange[n])
		plt.plot(Trange*conversionFactor, add_geff_rho, label=R"$g_{\mathrm{eff},\rho}^\mathrm{DS}$")
		plt.plot(Trange*conversionFactor, add_geff_s, label=R"$g_{\mathrm{eff},s}^\mathrm{DS}$")
		plt.xlabel("T / GeV")
		plt.xscale("log")
		plt.legend()
		plt.show()
	'''
