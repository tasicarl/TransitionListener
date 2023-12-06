# TransitionListener
TransitionListener is tool for calculating spectra of stochastic gravitational wave backgrounds emitted in first-order phase transitions of dark sectors beyond the standard model of particle physics. The code was used to obtain the results presented in `arXiv:2109.06208` "Turn up the volume: Listening to phase transitions in hot dark sectors" by Fatih Ertas, Felix Kahlhoefer and Carlo Tasillo. Therein, a special focus was set on the effects of increasign the gravitational wave spectrum's amplitude by increasing the temperature ratio between the dark setor and the standard model bath as well as the dilution effect by an intermediate period of early matter domination.

The underlying code CosmoTransitions (v2.0.2) by Carroll L. Wainwright (see `arXiv:1109.4189v1`) has been extended by several methods and modules for the calculation of
- effective relativistic degrees of freedom in the standard model (see `arXiv:1803.01038`) and the dark sector bath
- the nucleation temperature, transition strength, threshold transition strength for runaway bubbles and inverse transition timescale
- the dilution factor of a stochastic GW spectrum from entropy injection
- GW spectra as described in `arXiv:1512.06239`
- signal-to-noise ratios for current and future observatories as described in `arXiv:1811.11175v2`

To start the program with an example model analysis use

	python My_point.py

Using this procedure, a point in the model parameter space of a U(1) extension to the SM gauge group can be analyzed: First, the effective potential is calculated; then, the possibility of a first-order phase transition is considered. Using the nucleation criterion that the bubble nucleation rate reaches $H^-4$ (with $H$ being the Hubble parameter at the nucleation temeprature), the nucleation temeprature of bubbles is obtained. The calculation of the transition strength $/alpha$ and the inverse timescale $/beta \ H$ follows. As the dark sector is assumed to be unstable, further the possibility of decays to the SM is investigated. If these decays happen sufficiently late, a considerable entropy injection can be obtained that dilutes the GW signal. This dilution is quantified using the quantity D as defined in `arXiv:1811.03608v3`. After having obtained all necessary parameters, the GW spectrum as it would be observed by LISA or the Einstein Telescope is computed. Eventually, the expected signal-to-noise ratios for a list of future observatories are computed. All intermediate parameters of the analysis are saved together with some informative plots that might facilitate interpreting the physical results of the model analysis. Comments on all input parameters can be found in the file `my_point.py`.

For the analysis of a given part of a model parameter space use

	python My_scan.py

In doing so, the defined range of parameters will be analyzed on a two-dimensional grid. Eventually the results and all intermediate parameters will be plotted on that predefined grid.

Additionally, there is also the possibility to compare the resulting GW spectra referring to a list of parameter space configurations. This analysis can be started using

	python My_comparison.py

To check if the effective potential of the given model defined in `tl_dark_photon_model_mb.py` is calculated correctly, a cross-check can be obtained by executing

	python My_potential_plot_point.py

The code has been tested with `python v3.8.8`, `scipy v1.5.2`, `numpy v1.20.1`, `matplotlib v3.3.4`, `itertools-len v1.0`, and `tqmd v4.59.0`. Please feel free to write an email to carlo.tasillo@desy.de in case you identify any bug in the code or still need some further documentation. Enjoy!

**Note** (December 6, 2023): The CosmoTransitions backend we are using runs into errors of type 7 (nucleation criterion cannot be fulfilled) when the latest version of `scipy`. We can confirm that the code still runs using `scipy v1.10.1`. The issue is due to a the brentq method throwing a ValueError. We added the file `TL.yml` to set up a conda environment in which TransitionListener should run smoothly.