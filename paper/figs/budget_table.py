"""The theory-budget tables from the exact percolation along the line scan (data/line_budget.json, written by
reproduce/rate/line_budget.py):
  ../budget_table.tex                               the full table of appendix C, \\input by main.tex
  ../reproduce/numbers/output/budget_table_main.tex the rows quoted in section 2.3
Run by make_figures.py (target table_budget) or on its own: python budget_table.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "..", "data", "line_budget.json")))
OUTPUT = os.path.join(HERE, "..", "reproduce", "numbers", "output")
os.makedirs(OUTPUT, exist_ok=True)


def tex_sci(x, digits=2):
    m, e = f"{x:.{digits}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def kev(x):
    """T_p in keV: three significant figures, groups above a thousand separated by {,} as in the text."""
    k = x * 1e6
    if k >= 1000:
        return f"{k:,.0f}".replace(",", "{,}")
    return f"{k:.3g}"


def sgn(x, d=0):
    """A signed number; zero at the printed precision carries no sign."""
    s = f"{x:+.{d}f}"
    return s[1:] if float(s) == 0 else s


def pm(x, d=1):
    return f"\\pm{x:.{d}f}"


# table 1: every cell is the change of the NLO value under one alternative, X_alt/X_NLO - 1 in
# per cent: LO instead of NLO and the dimension-six operator signed; the two scale bands, the
# next-order estimate and the prefactor as plus-minus half the difference between the two directions
main = []
for o in res:
    n = o["nlo"]
    if not n or o["g"] not in (0.44, 0.45, 0.46, 0.475, 0.5, 0.55, 0.6, 0.7, 0.8):
        continue
    cells = [f"${o['g']:g}$", kev(n["T_p"]), f"{n['betaH_R']:.1f}", f"${o['S_nlo_minus_lo_at_Tp']:+.1f}$"]
    for q in ("T_p", "betaH_R"):
        e = o["entries_pct"][q]
        d = 0 if q == "T_p" else 1
        cells += [f"${sgn(e['lo_signed'], d)}$", f"${pm(e['mu4'])}$", f"${pm(e['mu3'])}$",
                  f"${sgn(e['dim6_signed'], 1)}$", f"${pm(e['hord'], d)}$", f"${pm(e['pref'], d)}$"]
    main.append(" & ".join(cells) + " \\\\")
# the closing rule is part of the file: a \\bottomrule after \\input inside a tabular fails
open(os.path.join(OUTPUT, "budget_table_main.tex"), "w").write("\n".join(main) + "\n\\bottomrule\n")

full = ["\\begin{tabular}{@{}lrrrrrrrrrr@{}}", "\\toprule",
        " & & & & & & & \\multicolumn{2}{c}{Theory band [\\%]} & \\multicolumn{2}{c}{Num.~accuracy [\\%]} \\\\",
        "\\cmidrule(lr){8-9}\\cmidrule(l){10-11}",
        "$g$ & $\\Tp$ [keV] & $\\bHR$ & $\\bHS$ & $\\bHG$ & $\\frac{1}{H}\\frac{\\dd\\ln a^3(1-P_{\\rm t})}{\\dd t}$ & "
        "$\\Delta S$ & $\\delta\\Tp$ & $\\delta\\bHR$ & $\\delta\\Tp$ & $\\delta\\bHR$ \\\\", "\\midrule"]
for o in res:
    n = o["nlo"]
    if not n or o.get("figure_only"):          # the extra couplings of figure 3 are not tabulated
        continue
    e, b = o["entries_pct"]["T_p"], o["entries_pct"]["betaH_R"]
    tl = o.get("TL")
    pc = lambda a, b_: f"${sgn(100 * (a / b_ - 1), 2)}$"
    tlt = pc(tl['T_p'], n['T_p']) if tl else "--"
    tlb = pc(tl['betaH_R'], n['betaH_R']) if (tl and tl.get("betaH_R")) else "--"
    full.append(f"${o['g']:g}$ & {kev(n['T_p'])} & {n['betaH_R']:.1f} & {n['betaH_S']:.1f} & "
                f"{n['beta_Gamma_H']:.1f} & ${n['dlnV_dt_over_H']:+.2f}$ & ${o['delta_S_at_Tp']:+.2f}$ & "
                f"$\\pm{e['total']:.0f}$ & $\\pm{b['total']:.1f}$ & {tlt} & {tlb} \\\\")
full += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "..", "budget_table.tex"), "w").write("\n".join(full) + "\n")
print("wrote budget_table_main.tex and budget_table.tex")
