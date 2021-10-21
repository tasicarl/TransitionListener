'''
This file includes the class `dilution_factor_calculation`, which can be used to calculate the dilution factor D as defined in arxiv:1811.03608v3
("Homeopathic Dark Matter, or how ..."). The input parameters are the initial photon bath temperature Ti, the mediator mass m, the mediator decay width Gamma, and the
temperature ratio between the dark sector and the SM bath at the chemical decoupling of the two sectors xi. Additional parameters are the ratio fDM of initial DM energy density to initial DS energy
density and the number of initial entropic DS dofs g_med, as well as the efective 3 -> 2 coupling of the dark mediator.

First, the dark sector and amount of Boltzmann suppression are calculated, then the SM energy and entropy dofs, then the initial energy densities in the
DS and the SM using the full integral formula. The age of the universe at T = Ti is calculated by integrating over the evolution of the Hubble parameter
infering from the initial energy densities to the ones before. If the mediator mass is bigger than its initial temperature, it is counted as matter whereas
in the other case ot is counted as radiation, resulting in an effective factor 2/3 or 1/2.

To solve the set of coupled ODEs (in a numerically convenient log-fashion), the time evolution of the mediator energy density is required. To not solve
the complete Boltzmann equation in every time step, an approximation is used that cuts the function in two pieces, before and after the relativistic-non-
relativistic transition of the mediator at t = tau * x_nr where x_nr is the dimensionless nr-transition time approximated as 7 * x_i / nu**2 with x_i =
t_i / tau (age of the universe in units of mediator lifetime at T = T_i) and nu = m / T_d,i where T_d,i = xi_i * T_i where xi_i is the initial temperature
between DS and SM.

Additionally, there are functions to plot the solution and a small facility that can be used for scans over the input parameters.
'''
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
try:
	import geff
except:
	from TransitionListener import geff

from time import perf_counter
from time import time

class dilution_factor_calculation():
	def __init__(self, Ti=800000, m=550000, Gamma=1e-9, xi_i=1, fDM=0, g_med=1, alpha32=0.00451176, verbose=True, foldername="", plot_results=False):
		self.inputparams =  Ti, m, Gamma, xi_i, fDM, g_med, alpha32

		# Planck mass in GeV, final x until ODE should be solved, nu = m_med / T_dark,i
		Mpl = 2.4e18
		xf = 10
		Tdi = Ti * xi_i
		nu = m / Tdi # < 1: rel, > 1: nr

		# initial relativistic dofs
		geff_instance = geff.effective_dof()
		grhoi = geff_instance.calc_g_eff_rho_SM(Ti, 1)
		gsi = geff_instance.calc_g_eff_s_SM(Ti, 1)
		del geff_instance
		gamma_i = grhoi / gsi

		# Initial energy densities
		def rho_initial(_T, _g, _nu):
			def kernel(u):
				np.seterr(over="ignore")
				k = u**2 * np.sqrt(u**2 + _nu**2) / (np.exp(np.sqrt(u**2 + _nu**2)) - 1)
				np.seterr(over="warn")
				return k
			return _g * _T**4 / (2 * np.pi**2) * quad(kernel, 0, np.inf)[0]

		def rho_initial_NR(_m, _g, _nu):
			def kernel(u):
				np.seterr(over="ignore")
				k = u**2 * np.sqrt(1 + u**2) / (np.exp(_nu * np.sqrt(1 + u**2)) - 1)
				np.seterr(over="warn")
				return k
			return _g * _m**4 / (2 * np.pi**2) * quad(kernel, 0, np.inf)[0]

		rho_medi = rho_initial(Tdi, g_med, nu)

		if rho_medi == 0:
			if verbose:
				print("\nFirst try of calculating rho_medi failed, try other method.")
			rho_medi = rho_initial_NR(m, g_med, nu)
			if rho_medi == 0:
				if verbose:
					print("Couldn't calculate rho_medi, because nu = ", nu, " is too large. Go on with rho_medi = 1e-100!\n")
				rho_medi = 1e-100
			elif verbose:
				print("Other method worked.\n")

		rho_radi = rho_initial(Ti, grhoi, 0)
		frad = rho_radi / rho_medi
		if np.isinf(frad):
			frad = 1e100
		rho_DMi = rho_medi * fDM
		rho_toti = rho_radi + rho_medi + rho_DMi

		# Initial time t_i / tau where T = Ti
		def hubbletime_RD_MD_factor(_nu, _rho_SM_rad, _rho_med, _rho_DM):
			rho_tot = _rho_SM_rad + _rho_med + _rho_DM
			if _nu < 1: # mediator initially relativistic
			#if True:
				rho_R = _rho_SM_rad + _rho_med
				rho_M = _rho_DM
			else:
				rho_R = _rho_SM_rad
				rho_M = _rho_DM + _rho_med
			Omega_R = rho_R / rho_tot
			Omega_M = rho_M / rho_tot
			def kernel(x):
				return 1/np.sqrt(Omega_M / x  + Omega_R / x**2)
			return quad(kernel, 0, 1)[0]
		xi = np.sqrt(3 * Mpl**2 * Gamma**2 / rho_toti)
		xi*= hubbletime_RD_MD_factor(nu, rho_radi, rho_medi, rho_DMi) # factor between 1/2 and 2/3 depending on RD or MD at initial point in time.

		# Relativistic - non-relativistic transition
		xnr = 7 * nu**(-2) * xi # 7 is approximate, could be further improved
		if xnr < xi:
			xnr = xi
	
		# Time-scale x_H
		xH = np.sqrt(3 * Gamma**2 * Mpl**2 / rho_medi)

		if xi < 1: # Condition for non-relativistic transitions, alternatively xnr < 1
			# computational input parameters
			init_cond = [0, 0, 0, 0, np.log(gamma_i)] 			# initial scalefactor, initial entropy ratio**3/4, initial mediator energy density ratio, initial entropic dof to initial entropic dof ratio, initial energetic/entropic dof ratio 
			N = 100 						# number of points of support for plot
			interval = [xi, xf]				# dimensionless time interval in which ODE is solved
			x = np.logspace(np.log10(xi), np.log10(xf), N) # points of support for plot

			# zeta
			try:
				logrho_supp, zeta_supp = np.genfromtxt("TransitionListener/dlogs-dlogrho.dat", unpack=True)
			except:
				logrho_supp, zeta_supp = np.genfromtxt("dlogs-dlogrho.dat", unpack=True)
			from scipy import interpolate
			self.zeta = interpolate.interp1d(logrho_supp, zeta_supp, bounds_error = False, fill_value = (zeta_supp[0], zeta_supp[-1]))
			self.Gamma = Gamma
			self.alpha32 = alpha32
			self.m = m
			self.rho_medi = rho_medi
			self.g_med = g_med
			self.can_warn = False
			self.nu = nu

			# Solve Friedmann equation
			args = interval, init_cond, x, xi, xH, xnr, fDM, frad, gamma_i, Ti
			logz, logr, logs, logG, loggamma = self.solve_friedmann(*args)
			z = np.exp(logz)
			r = np.exp(logr)
			s = np.exp(logs)
			G = np.exp(logG)
			gamma = np.exp(loggamma)
			friedmann_solution = z, r, s, G, gamma 

			# Other functions to plot
			gs = G * gsi
			grho = gamma * gs
			S = s**(3/4)
			rho_med = r * rho_medi
			rho_DM = rho_medi * fDM / z**3
			rho_rad = rho_medi * frad / z**4  * gamma / gamma_i * s / G**(1/3)
			rho_rad_new = rho_medi * frad / z**4  * gamma / gamma_i * (s-1) / G**(1/3)
			rho_rad_old = rho_medi * frad / z**4  * gamma / gamma_i * G**(1/3)
			T = Ti / z * s**(1/4) / G**(1/3)
			T_always_RD = (45/2/np.pi**2 * Mpl**2 * Gamma**2 / grho / x**2)**(1/4)

			self.grhoi = grhoi
			self.gsi = gsi
			self.g_med = g_med
			self.xi_i = xi_i
			self.x = x
			self.friedmann_solution = friedmann_solution
			self.other_plot_functions = xi, S, T, T_always_RD, grho, gs, rho_DM, rho_med, rho_rad_new, rho_rad_old
			self.finalT = T[-1]
			self.DSM = s[-1]**(3/4)
			self.xnr = xnr
			self.foldername = foldername
			
			#### Save plots and produce output
			if verbose and self.can_warn:
				print("An unexpected error occured in the cannibalism calculation: Gamma_32 < H for a relativistic mediator.")

			if plot_results:
				if verbose:
					print("Save dilution factor calculation + cannibalism plots...")
				self.plot_solution()

		elif verbose:
			print("\nError! xi > 1, non-physical input parameters!", self.inputparams, "\n")

	def plot_solution(self, showplot=False, savedata=True):
		if False:
			plt.rc('text', usetex=True)
			plt.rc('font', family='serif')

		x = self.x
		z, r, s, G, gamma = self.friedmann_solution
		xi, S, T, T_always_RD, grho, gs, rho_DM, rho_med, rho_rad_new, rho_rad_old = self.other_plot_functions
		xpivot = x[-1] / z[-1]**2

		f, axes = plt.subplots(2, 3, sharex=True, figsize=[12,5])
		ax1 = axes[0,0]
		ax2 = axes[0,1]
		ax3 = axes[1,0]
		ax4 = axes[1,1]
		ax5 = axes[0,2]
		ax6 = axes[1,2]

		ax1.set_title(R"Normalized scalefactor $\bar{a} = a / a_\mathrm{cd}$", size=16)
		ax1.loglog(x, (x/xi)**(1/2), alpha = 0.5, label='initial RD')
		ax1.loglog(x, (x/xpivot)**(1/2), alpha = 0.5, label='final RD')
		ax1.loglog(x, z, label=R'$\bar{a}(x)$')
		#ax1.set_ylim(10**(-4), 10**10)
		ax1.legend(fontsize=12)

		ax2.set_title("Photon bath temperature", size=16)
		ax2.loglog(x, T_always_RD, label=R"$T_\mathrm{SM}^\mathrm{LCDM}$ / GeV")
		ax2.loglog(x, T, label=R"$T_\mathrm{SM}$ / GeV")
		ax2.legend(fontsize=12)
		#ax2.set_ylim(10**(-4), 10**10)

		ax3.set_title(R"Entropy ratio $S_\mathrm{SM} / S_\mathrm{SM,cd}$", size=16)
		ax3.loglog(x, S, label='S ratio')
		ax3.set_xlabel(R'Time $x = t / \tau$', size=14)
		#ax3.set_ylim(10**(-4), 10**10)

		ax4.set_title(R"Energy densities $\rho a^3$ / GeV$^4$", size=16)
		ax4.loglog(x, (rho_rad_new + rho_rad_old) * z**3, label='Rad')
		ax4.loglog(x, rho_med * z**3, label='Med')
		ax4.loglog(x, rho_DM * z**3, label='DM')
		ax4.legend(fontsize=12)
		ax4.set_xlabel(R'Time $x = t / \tau$', size=14)
		#ax4.set_ylim(10**(-20), 10**15)

		ax5.set_title(R"Dofs $g_{*s}^\mathrm{SM}$ and $g_{*\rho}^\mathrm{SM}$", size=16)
		ax5.plot(x, grho, label=R'$g_\mathrm{*\rho}$')
		ax5.plot(x, gs, "--", label=R'$g_\mathrm{*s}$')
		ax5.legend(fontsize=12)
		
		ax6.set_title(R"Dof ratios $\gamma$ and $G$", size=16)
		ax6.plot(x, gamma, label=R'$\gamma$')
		ax6.plot(x, G, label=R'$G$')
		ax6.legend(fontsize=12)
		ax6.set_xlabel(R'Time $x = t / \tau$', size=14)

		plt.tight_layout()
		time = perf_counter()

		#if showplot:
		if False:
			plt.show() 
		else:	
			from pathlib import Path
			Path(self.foldername).mkdir(parents=True, exist_ok=True)
			plt.savefig(self.foldername+"Dilution.pdf")
		plt.close(f)

		if savedata:
			file = open(self.foldername+"Dilution_data.txt", "x") 
			file.write("inputparams: " + str(self.inputparams))
			file.write("\n\n")
			file.write("x\n" + str(x))
			file.write("\n\n")
			file.write("z\n" + str(z))
			file.write("\n\n")
			file.write("T_LCDM\n" + str(T_always_RD))
			file.write("\n\n")
			file.write("T\n" + str(T))
			file.write("\n\n")
			file.write("S\n" + str(S))
			file.write("\n\n")
			file.write("rho_rad\n" + str(rho_rad_new + rho_rad_old))
			file.write("\n\n")
			file.write("rho_med\n" + str(rho_med))
			file.write("\n\n")
			file.write("rho_DM\n" + str(rho_DM))
			file.write("\n\n")
			file.write("gs\n" + str(gs))
			file.write("\n\n")
			file.write("grho\n" + str(grho))
			file.write("\n\n")
			file.write("gamma\n" + str(gamma))
			file.write("\n\n")
			file.write("G\n" + str(G))
			file.close()

	def derivatives(self, logz, logr, logs, logG, loggamma, x, xi, xH, xnr, fDM, frad, gamma_i, Ti, d_G_spl, d_gamma_spl):
		# non-log expressions:
		z = np.exp(logz)
		r = np.exp(logr)
		s = np.exp(logs)
		G = np.exp(logG)
		gamma = np.exp(loggamma)

		# derivatives of G and Gamma w.r.t. T at T(x)
		dGdT = d_G_spl(Ti / z * s**(1/4) / G**(1/3))
		dgammadT = d_gamma_spl(Ti / z * s**(1/4) / G**(1/3))

		# log derivatives of z, r, s w.r.t. x
		dlogzdx = 1 / xH * np.sqrt(r + fDM/z**3 + frad/z**4 * gamma / gamma_i * s / G**(1/3))

		H = dlogzdx * self.Gamma # H = a-dot / a = z' / z / tau = (log z)' * Gamma
		c = 25 * np.sqrt(5) * np.pi**2 / 5185 * self.alpha32**2
		n_med = r * self.rho_medi / self.m # assumes nr mediator, could also be calculated precisely.
		Gamma32 = c / self.m**5 * n_med**2 # 3->2 rate, mediator cannibalism

		if Gamma32 > H:
			rhobar = r * self.rho_medi / (self.g_med * self.m**4 / (2 * np.pi**2)) # the rhobar was used to interpolate. Now, fix dimensions first.
			prefactor = 3 * self.zeta(np.log(rhobar))
		else:
			if x <= xnr:
				prefactor = 4
				self.can_warn = True
			else:
				prefactor = 3
		#print(Gamma32>H, Gamma32, H, prefactor)
		dlogrdx = - 1 - prefactor * dlogzdx
		dlogsdx = - G**(1/3) / s * gamma_i / frad * z**4 * r * (dlogrdx + prefactor * dlogzdx)

		# non-log derivatives of z and s (necessary for G' and gamma')
		dzdx = dlogzdx * z
		dsdx = dlogsdx * s
		
		# G'
		dlogGdx = - 3 / 4 * Ti * dGdT / s**(3/4) / z
		dlogGdx *= 4 * s * dzdx - dsdx * z
		dlogGdx /= Ti * dGdT * s**(1/4) + 3 * G**(4/3) * z

		dGdx = dlogGdx * G

		# gamma'
		dloggammadx = Ti * dgammadT / gamma
		dloggammadx *= 3 * G * z * dsdx - 4 * dGdx * z * s - 12 * G * dzdx * s
		dloggammadx /= 12 * G**(4/3) * s**(3/4) * z**2

		#dlogGdx, dloggammadx = 0, 0
		return dlogzdx, dlogrdx, dlogsdx, dlogGdx, dloggammadx

	def friedmann(self, x, logzrsGgamma, *args):
		#Friedann equation caller
		logz, logr, logs, logG, loggamma = logzrsGgamma

		xi, xH, xnr, fDM, frad, gamma_i, Ti, d_G_spl, d_gamma_spl = args
		return self.derivatives(logz, logr, logs, logG, loggamma, x, xi, xH, xnr, fDM, frad, gamma_i, Ti, d_G_spl, d_gamma_spl)

	def solve_friedmann(self, interval, init_cond, *args):
		x, xi, xH, xnr, fDM, frad, gamma_i, Ti = args
		T_fill = np.logspace(np.log10(1.9952623e-06), np.log10(9.9738985e+16), 100)

		dof = geff.effective_dof()
		grho_fill = dof.calc_g_eff_rho_SM(T_fill, 1)
		gs_fill = dof.calc_g_eff_s_SM(T_fill, 1)

		G_fill = gs_fill / dof.calc_g_eff_s_SM(Ti, 1)
		del dof
		gamma_fill = grho_fill / gs_fill

		G_deriv = np.gradient(G_fill) / np.gradient(T_fill)
		gamma_deriv = np.gradient(gamma_fill) / np.gradient(T_fill)

		d_G_spl = interp1d(T_fill, G_deriv, bounds_error = False, fill_value = (G_deriv[0], G_deriv[-1]))
		d_gamma_spl = interp1d(T_fill, gamma_deriv, bounds_error = False, fill_value = (gamma_deriv[0], gamma_deriv[-1]))

		if False:
			plt.plot(T_fill, grho_fill, label = "g_rho")
			plt.plot(T_fill, gs_fill, label = "g_s")
			plt.plot(T_fill, G_fill, label = "G = gs / gsi")
			plt.plot(T_fill, gamma_fill, label = "gamma = grho / gs")
			plt.plot(T_fill, G_deriv, label = "dGdT")
			plt.plot(T_fill, gamma_deriv, label = "dgammaT")
			plt.plot(T_fill, -G_deriv, label = "-dGdT")
			plt.plot(T_fill, -gamma_deriv, label = "-dgammaT")
			T_2 = np.logspace(-6, 16, 100000)
			#plt.plot(T_2, d_G_spl(T_2))
			#plt.plot(T_2, d_gamma_spl(T_2))
			plt.xscale("log")
			plt.yscale("log")
			#plt.ylim(1e0, 1e3)
			#plt.xlim(1e-5, 1e4)
			plt.legend()
			plt.show()

		sol = solve_ivp(self.friedmann, interval, init_cond, args = (xi, xH, xnr, fDM, frad, gamma_i, Ti, d_G_spl, d_gamma_spl), dense_output=True, method="DOP853", rtol=1e-10)
		return sol.sol(x)