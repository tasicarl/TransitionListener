"""The local significance of the preference for the phase transition against the binary lifetime tau_f (figure 6),
read from data/preference_tau_profile.npz (binaries/preference_tau_profile.py): per prior set the minimum over tau_f
and where it lies, the value at tau_f = 5 and 7 Gyr, and the lifetime above which it exceeds 3 and 4 sigma.
    python preference_profile_numbers.py"""
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
d = np.load(os.path.join(DATA, "preference_tau_profile.npz"))
tau = d["tau"]
for prior, name in (("nanograv", "NANOGrav"), ("recent", "survey-averaged"), ("literature", "Tomczak-Kormendy")):
    s = d[f"sigma_cp_v2_{prior}"]
    i = int(np.nanargmin(s))
    at = {t: float(np.interp(t, tau, s)) for t in (5.0, 7.0)}
    above = {k: (f"{tau[np.argmax(s > k)]:.2f} Gyr" if np.any(s > k) else "never") for k in (3.0, 4.0)}
    print(f"{name}: minimum {s[i]:.2f} sigma at tau_f = {tau[i]:.2f} Gyr; {at[5.0]:.2f} sigma at 5 Gyr, "
          f"{at[7.0]:.2f} sigma at 7 Gyr; above 3 sigma from {above[3.0]}, above 4 sigma from {above[4.0]}")
