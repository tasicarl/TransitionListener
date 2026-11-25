"""
Low-level helpers for symbolic curvature tensors.

The generator stage writes each tensor entry as the ``sympy.srepr`` of an expression in a JSON file. Here the entries are
compiled to Python code, which reads the numbers of a model without SymPy: the file holds only Symbol, Integer, Rational,
Add, Mul, Pow, I and conjugate, and :func:`build_substitutions` gives those names their numerical meaning.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import json
import math
from itertools import permutations
import numpy as np

from transitionlistener import errors


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
    """Recursively turn JSON-loaded tensor data into compiled expressions, keyed by index tuples."""
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
        return compile(obj, "<curvature tensor>", "eval")
    return obj


@lru_cache(maxsize=None)
def load_tensor_dataset(tensor_file: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load a JSON tensor dataset.

    Parameters
    ----------
    tensor_file:
        Absolute path to the JSON document produced by the counterterm generator.

    Returns
    -------
    (metadata, tensors)
        Metadata dictionary and a nested dictionary of compiled expressions keyed by index tuples.
    """

    if not tensor_file.exists():
        raise errors.InitPotentialError(
            f"Generated tensor file not found: {tensor_file}. "
            "Run the counterterm generator to create it."
        )

    with tensor_file.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    return raw.get("metadata", {}), _convert_tensor_dict(raw.get("tensors", {}))


def build_substitutions(
    metadata: Mapping[str, Any],
    parameter_values: Mapping[str, complex],
) -> dict[str, Any]:
    """
    Construct the names under which the compiled expressions run.

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
        The globals for :func:`evaluate_expression`.
    """

    aliases = metadata.get("parameter_symbol_names", {})
    numbers: dict[str, complex] = {}
    for key, value in parameter_values.items():
        numbers[str(key)] = value
        alias = aliases.get(str(key))
        if alias is not None:
            numbers[str(alias)] = value

    def symbol(name: str, **assumptions: Any) -> complex:
        try:
            return numbers[name]
        except KeyError:
            raise ValueError(f"Unresolved symbol in tensor evaluation: {name}") from None

    return {
        "__builtins__": {},
        "Symbol": symbol,
        "Integer": int,
        "Rational": lambda numerator, denominator: numerator / denominator,
        "Add": lambda *terms: sum(terms),
        "Mul": lambda *factors: math.prod(factors),
        "Pow": pow,
        "I": 1j,
        "conjugate": lambda value: complex(value).conjugate(),
    }


def evaluate_expression(expr: Any, names: Mapping[str, Any]) -> complex | float:
    """
    Evaluate a compiled tensor entry.

    Parameters
    ----------
    expr:
        Compiled expression from :func:`load_tensor_dataset`.
    names:
        The globals from :func:`build_substitutions`.

    Returns
    -------
    complex
        Numerical value of the expression.  Purely real results are returned as
        ``float`` for convenience.
    """

    if expr is None:
        return 0.0
    value = complex(eval(expr, names))
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
