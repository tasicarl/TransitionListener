"""
Low-level helpers for symbolic curvature tensors.

The utilities provided here centralise the JSON I/O that we use to ship SymPy
expressions from the generator stage to the runtime evaluation.  They also
offer a thin numerical wrapper around each expression so that models can keep
their own code short and declarative.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import json
from itertools import permutations
import numpy as np

from transitionlistener import errors

if TYPE_CHECKING:
    import sympy as sp


def _require_sympy():
    """Import SymPy on demand and raise a targeted error when it is unavailable."""
    try:
        import sympy as sp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SymPy is required for symbolic counterterm tensor evaluation. "
            "Install `sympy` to use SymPy-based counterterm models."
        ) from exc
    return sp


def _is_index_key(key: str) -> bool:
    """Return whether ``key`` encodes a tensor index tuple such as ``"(0, 1)"``."""
    key = key.strip()
    return key.startswith("(") and key.endswith(")")


def _parse_index_key(key: str) -> tuple[int, ...]:
    """Parse a stringified tensor-index tuple back into integers."""
    stripped = key.strip()[1:-1]
    if not stripped:
        return tuple()
    return tuple(int(part.strip()) for part in stripped.split(","))


def _convert_tensor_dict(obj: Any) -> Any:
    """Recursively turn JSON-loaded tensor data back into SymPy expressions."""
    sp = _require_sympy()
    if isinstance(obj, dict):
        converted = {}
        for key, value in obj.items():
            if _is_index_key(key):
                converted[_parse_index_key(key)] = _convert_tensor_dict(value)
            else:
                converted[key] = _convert_tensor_dict(value)
        return converted
    if isinstance(obj, list):
        return [_convert_tensor_dict(item) for item in obj]
    if isinstance(obj, str):
        return sp.sympify(obj, evaluate=False)
    return obj


def _initialise_parameter_symbols(
    metadata: dict[str, Any],
    *,
    complex_parameters: tuple[str, ...] = (),
) -> dict[str, sp.Symbol]:
    """Construct the SymPy symbols corresponding to the stored metadata labels."""
    sp = _require_sympy()
    parameter_aliases = metadata.get("parameter_symbol_names", {})
    parameter_symbols: dict[str, sp.Symbol] = {}
    for name in metadata.get("parameters", []):
        alias = parameter_aliases.get(name, name)
        alias_expr = sp.sympify(alias)
        if isinstance(alias_expr, sp.Symbol):
            base_name = alias_expr.name
        else:
            base_name = str(alias)
            alias_expr = sp.Symbol(base_name)
        is_complex = base_name in complex_parameters
        symbol = sp.Symbol(base_name) if is_complex else sp.Symbol(base_name, real=True)
        for tag in {name, base_name, str(alias)}:
            parameter_symbols[tag] = symbol
        parameter_symbols[alias_expr] = symbol
    metadata["parameter_symbols"] = parameter_symbols
    return parameter_symbols


@lru_cache(maxsize=None)
def load_tensor_dataset(
    tensor_file: Path,
    *,
    complex_parameters: tuple[str, ...] = (
        "Vud",
        "Vus",
        "Vub",
        "Vcd",
        "Vcs",
        "Vcb",
        "Vtd",
        "Vts",
        "Vtb",
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load a JSON tensor dataset and return SymPy expressions ready for evaluation.

    Parameters
    ----------
    tensor_file:
        Absolute path to the JSON document produced by the counterterm generator.
    complex_parameters:
        Iterable with the subset of parameters that need complex-valued symbols.

    Returns
    -------
    (metadata, tensors)
        Metadata dictionary (augmented with ``parameter_symbols``) and a nested
        dictionary of SymPy expressions keyed by index tuples.
    """

    if not tensor_file.exists():
        raise errors.InitPotentialError(
            f"Generated tensor file not found: {tensor_file}. "
            "Run the counterterm generator to create it."
        )

    with tensor_file.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    metadata = raw.get("metadata", {})
    _initialise_parameter_symbols(metadata, complex_parameters=complex_parameters)
    tensors = _convert_tensor_dict(raw.get("tensors", {}))
    return metadata, tensors


def build_substitutions(
    metadata: Mapping[str, Any],
    parameter_values: Mapping[str, complex],
) -> dict[sp.Symbol, complex]:
    """
    Construct the substitution dictionary used during tensor evaluation.

    Parameters
    ----------
    metadata:
        Metadata dictionary returned by :func:`load_tensor_dataset`.
    parameter_values:
        Mapping from parameter identifiers (as stored in the metadata) to the
        numerical values provided by the model.

    Returns
    -------
    dict
        Mapping ``sympy.Symbol -> value`` ready to be passed to
        :func:`evaluate_expression`.
    """

    sp = _require_sympy()
    subs: dict[sp.Symbol, complex] = {}
    param_symbols: dict[str, sp.Symbol] = dict(metadata.get("parameter_symbols", {}))
    aliases = metadata.get("parameter_symbol_names", {})

    def assign(label: str, value: complex) -> None:
        symbol = param_symbols.get(label)
        if symbol is None:
            symbol = sp.Symbol(label)
            param_symbols[label] = symbol
        subs[symbol] = value
        # Also populate the alias to avoid round trips.
        alias_label = aliases.get(label)
        if alias_label is not None:
            alias_symbol = param_symbols.get(alias_label)
            if alias_symbol is None:
                alias_symbol = sp.Symbol(alias_label)
                param_symbols[alias_label] = alias_symbol
            subs[alias_symbol] = value

    for key, value in parameter_values.items():
        assign(str(key), complex(value))

    metadata["parameter_symbols"] = param_symbols  # type: ignore[index]
    return subs


def evaluate_expression(
    expr: sp.Expr,
    substitutions: Mapping[sp.Symbol, complex],
    *,
    extra_substitutions: Mapping[str | sp.Symbol, complex] | None = None,
) -> complex | float:
    """
    Evaluate a tensor entry to floating-point precision.

    Parameters
    ----------
    expr:
        SymPy expression representing the tensor component.
    substitutions:
        Base substitution map generated by :func:`build_substitutions`.
    extra_substitutions:
        Optional field-dependent substitutions that should override the base map.

    Returns
    -------
    complex
        Numerical value of the expression.  Purely real results are returned as
        ``float`` for convenience.
    """

    sp = _require_sympy()
    if expr is None:
        return 0.0

    subs = dict(substitutions)
    if extra_substitutions:
        for key, value in extra_substitutions.items():
            symbol = key if isinstance(key, sp.Symbol) else sp.Symbol(str(key))
            subs[symbol] = complex(value)

    evaluated = sp.N(expr.subs(subs), 30)
    free_symbols = getattr(evaluated, "free_symbols", set())
    if free_symbols:
        missing = ", ".join(str(sym) for sym in sorted(free_symbols, key=str))
        raise ValueError(f"Unresolved symbols in tensor evaluation: {missing} (expression: {expr})")

    try:
        value = complex(evaluated)
    except TypeError:
        real_part = float(sp.N(sp.re(evaluated), 30))
        imag_part = float(sp.N(sp.im(evaluated), 30))
        value = complex(real_part, imag_part)

    if abs(value.imag) < 1e-12:
        return float(value.real)
    if abs(value.real) < 1e-12:
        return complex(0.0, value.imag)
    return value


def ensure_symmetric(matrix):
    """
    Symmetrise a square numerical matrix.

    This helper is used to enforce the hermiticity of Coleman–Weinberg Hessians
    after evaluating them numerically.
    """

    return 0.5 * (matrix + matrix.T)


def _determine_dim(keys, positions):
    """Infer an array dimension from the largest index stored in ``keys``."""
    max_index = 0
    for key in keys:
        for pos in positions:
            max_index = max(max_index, key[pos])
    return max_index + 1


def evaluate_curvature_arrays(
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    parameter_values: Mapping[str, complex],
    *,
    yukawa_key: str,
) -> dict[str, np.ndarray]:
    """
    Evaluate the curvature tensors stored in ``tensors`` using ``parameter_values``.
    """

    subs = build_substitutions(metadata, parameter_values)
    n_h = len(metadata.get("field_basis", []))
    if n_h == 0:
        n_h = _determine_dim(tensors.get("Curvature_Higgs_L2", {}).keys(), (0, 1))

    h2 = np.zeros((n_h, n_h), dtype=float)
    for (i, j), expr in tensors.get("Curvature_Higgs_L2", {}).items():
        value = float(evaluate_expression(expr, subs))
        h2[i, j] = value
        h2[j, i] = value

    h3 = np.zeros((n_h, n_h, n_h), dtype=float)
    for key, expr in tensors.get("Curvature_Higgs_L3", {}).items():
        value = float(evaluate_expression(expr, subs))
        for perm in set(permutations(key)):
            h3[perm] = value

    h4 = np.zeros((n_h, n_h, n_h, n_h), dtype=float)
    for key, expr in tensors.get("Curvature_Higgs_L4", {}).items():
        value = float(evaluate_expression(expr, subs))
        for perm in set(permutations(key)):
            h4[perm] = value

    gauge_data = tensors.get("Curvature_Gauge_G2H2", {})
    if gauge_data:
        n_g = _determine_dim(gauge_data.keys(), (0, 1))
    else:
        n_g = 0
    gauge = np.zeros((n_g, n_g, n_h, n_h), dtype=float)
    for (a, b, i, j), expr in gauge_data.items():
        value = float(evaluate_expression(expr, subs))
        gauge[a, b, i, j] = value
        gauge[a, b, j, i] = value
        gauge[b, a, j, i] = value
        gauge[b, a, i, j] = value

    quark_data = tensors.get("Curvature_Quark_F2H1", {}).get(yukawa_key, {})
    lepton_data = tensors.get("Curvature_Lepton_F2H1", {}).get(yukawa_key, {})

    n_q = _determine_dim(quark_data.keys(), (0, 1)) if quark_data else 0
    n_l = _determine_dim(lepton_data.keys(), (0, 1)) if lepton_data else 0

    quark = np.zeros((n_q, n_q, n_h), dtype=complex)
    for (i, j, k), expr in quark_data.items():
        quark[i, j, k] = evaluate_expression(expr, subs)

    lepton = np.zeros((n_l, n_l, n_h), dtype=complex)
    for (i, j, k), expr in lepton_data.items():
        lepton[i, j, k] = evaluate_expression(expr, subs)

    return {
        "H2": h2,
        "H3": h3,
        "H4": h4,
        "Gauge": gauge,
        "Quark": quark,
        "Lepton": lepton,
    }
