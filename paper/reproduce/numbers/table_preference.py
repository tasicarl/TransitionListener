"""Writes output/table_preference_main.tex (CURN: NANOGrav and IPTA) and output/table_preference_hd.tex (NANOGrav:
CURN and Hellings--Downs) from the preference runs of binaries/preference.py (paper/data), for the default
lower edge of the hardening timescale (preference_tau0.1.json) and for tau_f >= 4 Gyr (preference_tau4.json).
The local significance is binaryfog.significance.sigma_from_gain for two free parameters (g, v), as in preference.py.
For IPTA the maximum of ln L with the phase transition is lowered by 0.15, the scatter of the IPTA likelihood between
neighbouring grid points around its maximum (caption of table tab:preference).
    python table_preference.py [--check]      (--check: compare with the files instead of writing them)"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUTPUT = os.path.join(HERE, "output")
from binaryfog.significance import sigma_from_gain  # noqa: E402  (pip install binaryfog==1.0.0)

IPTA_SCATTER = 0.15
ROWS = (("nanograv", r"NANOGrav~\cite{NANOGrav:2023hfp}"),
        ("literature", r"Tomczak--Kormendy~\cite{Tomczak:2013bxa, Kormendy:2013dxa}"),
        ("recent", r"Galaxy survey-averaged~\cite{Bringmann:2026uxo}"))
TAUS = ("0.1", "4")


def sigma(run, key, prior):
    r = run[key][prior]
    gain = r["gain"] - (IPTA_SCATTER if key == "ipta2" else 0.0)
    return f"${float(sigma_from_gain(gain, 2)):.1f}\\sigma$"


def table(columns, header):
    runs = {t: json.load(open(os.path.join(DATA, f"preference_tau{t}.json"))) for t in TAUS}
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{$\tau_{\rm f} \geq 0.1$\,Gyr} & \multicolumn{2}{c}{$\tau_{\rm f} \geq 4$\,Gyr} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             "Astrophysical priors & " + " & ".join(header * 2) + r" \\", r"\midrule"]
    for prior, name in ROWS:
        lines.append(name + " & " + " & ".join(sigma(runs[t], k, prior) for t in TAUS for k in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


OUT = {"table_preference_main.tex": table(("cp_v2", "ipta2"), ("NANOGrav", "IPTA")),
       "table_preference_hd.tex": table(("cp_v2", "hd_v2"), ("CURN", "Hellings--Downs"))}
if __name__ == "__main__":
    for name, text in OUT.items():
        path = os.path.join(OUTPUT, name)
        if "--check" in sys.argv:
            old = open(path).read()
            print(name, "identical" if old == text else "DIFFERS")
            if old != text:
                print(text)
        else:
            open(path, "w").write(text)
            print("wrote", os.path.basename(path))
