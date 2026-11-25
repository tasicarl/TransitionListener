"""The counterterms of the 2HDM without symbolic work: the tensor entries against SymPy, the system matrix against its
derivation, the counterterms against recorded numbers."""
import json
import os

import numpy as np
import pytest
import sympy as sp

from transitionlistener.counterterms import tensors, twohdm_solver
from transitionlistener.helper_functions import load_potential

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDED = {
    "lambda3=5.5, type 1": ({"lambda3": 5.5, "yukawa_type": 1}, (
        -0.8566852718723972, -0.31467329737888106, -0.2404162087987007, 0.0, 0.0021019131098849457, 8022.872935163546,
        7838.869136400959, 15.187287625803704)),
    "lambda3=5.8096, type 2": ({"lambda3": 5.80958300958301, "yukawa_type": 2}, (
        14.206049829867212, -0.34720429092243593, -0.2655453459266102, 0.0, -0.020645970722002895, 6781.212000674641,
        8863.756139972578, -75.31736372763402)),
}
BENCHMARK = {"lambda1": 0.28894, "lambda2": 0.26237, "lambda4": -2.17494, "lambda5": -2.2417, "m12_sq_GeV2": 1573.171,
             "tan_beta": 21.3949, "v_GeV": 246.22}


def model(values):
    return load_potential(os.path.join(ROOT, "models", "TL_2HDM_BSMPT.py"), "R2HDM")({**BENCHMARK, **values}, verbose=False)


@pytest.mark.parametrize("name", RECORDED)
def test_the_counterterms_are_the_recorded_ones(name):
    values, recorded = RECORDED[name]
    pot = model(values)
    ours = [getattr(pot, key) for key in ("dlambda1", "dlambda2", "dlambda3", "dlambda4", "dlambda5", "dm11_sq", "dm22_sq",
                                          "dm12_sq")]
    assert ours == pytest.approx(list(recorded), rel=1e-13)


def entries(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from entries(value)
    elif isinstance(tree, str):
        yield tree


def test_every_tensor_entry_is_the_sympy_expression():
    """Only names that SymPy reads the same way can reach the compiled entries, and each of them has this value."""
    path = twohdm_solver.GENERATED_TENSOR_FILE
    raw = json.load(path.open())
    metadata = raw["metadata"]
    rng = np.random.default_rng(3)
    numbers = {name: float(rng.uniform(0.3, 3.0)) for name in metadata["parameters"]}
    for name in ("Vud", "Vus", "Vub", "Vcd", "Vcs", "Vcb", "Vtd", "Vts", "Vtb"):
        numbers[name] = complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
    names = tensors.build_substitutions(metadata, numbers)
    aliases = metadata.get("parameter_symbol_names", {})
    values = {str(aliases.get(key, key)): value for key, value in numbers.items()}
    expressions = sorted(set(entries(raw["tensors"])))
    assert len(expressions) > 40
    for text in expressions:
        expression = sp.sympify(text, evaluate=False)
        by_sympy = complex(sp.N(expression.subs({symbol: values[symbol.name] for symbol in expression.free_symbols}), 30))
        ours = complex(eval(compile(text, "<test>", "eval"), names))
        assert ours == pytest.approx(by_sympy, rel=1e-13, abs=1e-13), text


def test_a_tensor_entry_without_a_number_is_refused():
    names = tensors.build_substitutions({}, {"a": 1.0})
    assert tensors.evaluate_expression(compile("Symbol('a', real=True)", "<test>", "eval"), names) == 1.0
    with pytest.raises(ValueError, match="Unresolved"):
        tensors.evaluate_expression(compile("Symbol('b', real=True)", "<test>", "eval"), names)


def test_tensors_values_are_trimmed_as_real_or_imaginary():
    names = tensors.build_substitutions({}, {"a": 2.0})
    code = lambda text: compile(text, "<test>", "eval")     # noqa: E731
    assert tensors.evaluate_expression(code("Mul(Symbol('a'), Rational(1, 4))"), names) == 0.5
    assert type(tensors.evaluate_expression(code("Add(Symbol('a'), Mul(Integer(0), I))"), names)) is float
    assert tensors.evaluate_expression(code("Mul(I, Symbol('a'))"), names) == 2j
    assert tensors.evaluate_expression(code("conjugate(Add(Integer(1), I))"), names) == 1 - 1j
    assert tensors.evaluate_expression(code("Pow(Symbol('a'), Integer(-1))"), names) == 0.5
    assert tensors.evaluate_expression(None, names) == 0.0


def test_the_counterterm_matrix_is_the_derivative_of_the_counterterm_potential():
    """Rows: the tadpoles and curvatures of CONDITION_SPECS of the potential in the docstring, at the vacuum, per counterterm."""
    deltas = sp.symbols("dm11 dm22 dm12 dl1 dl2 dl3 dl5", real=True)
    dm11, dm22, dm12, dl1, dl2, dl3, dl5 = deltas
    fields = sp.symbols("phi1p_R phi1p_I phi2p_R phi2p_I phi1_R phi1_I phi2_R phi2_I", real=True)
    f = fields
    phi1 = sp.Matrix([(f[0] + sp.I * f[1]) / sp.sqrt(2), (f[4] + sp.I * f[5]) / sp.sqrt(2)])
    phi2 = sp.Matrix([(f[2] + sp.I * f[3]) / sp.sqrt(2), (f[6] + sp.I * f[7]) / sp.sqrt(2)])
    a = (phi1.H * phi1)[0]
    b = (phi2.H * phi2)[0]
    c = (phi1.H * phi2)[0]
    d = (phi2.H * phi1)[0]
    V = (dm11 * a + dm22 * b - dm12 * (c + d) + dl1 / 2 * a**2 + dl2 / 2 * b**2 + dl3 * a * b
         + dl5 / 2 * (c**2 + d**2))
    v1, v2 = sp.symbols("v1 v2", real=True)
    vacuum = {f[0]: 0, f[1]: 0, f[2]: 0, f[3]: 0, f[5]: 0, f[7]: 0, f[4]: v1, f[6]: v2}
    rows = []
    for kind, index in twohdm_solver.CONDITION_SPECS:
        condition = sp.diff(V, f[index]) if kind == "grad" else sp.diff(V, f[index[0]], f[index[1]])
        condition = sp.expand(condition.subs(vacuum))
        rows.append([sp.diff(condition, delta) for delta in deltas])
    derived = sp.lambdify((v1, v2), sp.Matrix(rows), "numpy")
    for point in ((1.7, 3.1), (246.0, 11.0), (0.4, 230.0)):
        assert twohdm_solver._counterterm_matrix(*point) == pytest.approx(np.asarray(derived(*point), dtype=float), rel=1e-14)
