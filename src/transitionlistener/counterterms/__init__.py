"""
Counterterm and quantum-correction utilities.

This package bundles the Coleman–Weinberg tensor helpers with the registry and
solver infrastructure used to generate and apply model-specific counterterms.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Type

import importlib
import json
import pkgutil

from . import cw, tensors

__all__ = [
    "cw",
    "tensors",
    "CounterTerms",
    "CountertermGenerator",
    "register_generator",
    "get_generator",
    "generate_counterterms",
]


@dataclass(slots=True)
class CounterTerms:
    """
    Generic container for renormalisation counterterms.

    Instead of hard-coding a particular model layout, the container stores an
    arbitrary mapping from identifiers to floating-point values and exposes it
    via attribute, item, and dictionary-style access.  This keeps the 2HDM
    interface backwards compatible while allowing other models to choose their
    own naming convention.
    """

    values: Dict[str, float]

    def __post_init__(self) -> None:
        """Copy the supplied mapping so later mutations do not leak back to the caller."""
        self.values = dict(self.values)

    def __getattr__(self, name: str) -> float:
        """Expose stored counterterms through attribute access."""
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(f"Unknown counterterm '{name}'.") from exc

    def __getitem__(self, key: str) -> float:
        """Expose stored counterterms through dictionary-style indexing."""
        return self.values[key]

    def get(self, key: str, default: float | None = None) -> float | None:
        """Return ``values[key]`` when present and ``default`` otherwise."""
        return self.values.get(key, default)

    def as_dict(self) -> Dict[str, float]:
        """Return a shallow copy of the stored counterterms."""
        return dict(self.values)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, float]) -> "CounterTerms":
        """Build a :class:`CounterTerms` instance from a dictionary."""
        return cls(dict(mapping))

    @classmethod
    def from_kwargs(cls, **kwargs: float) -> "CounterTerms":
        """Build a :class:`CounterTerms` instance from keyword arguments."""
        return cls(dict(kwargs))


class CountertermGenerator:
    """
    Base class for SymPy-based counterterm dataset generators.

    Subclasses must implement :meth:`build_dataset`, returning the metadata and
    tensor dictionary that will be serialised to JSON.
    """

    model_key: str = "BaseModel"
    output_filename: str = "counterterms.json"

    def output_path(self) -> Path:
        """Return the canonical location of the generated tensor JSON file."""
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "models" / "generated" / self.output_filename

    def build_dataset(self) -> Tuple[dict, dict]:
        """Return ``(metadata, tensors)`` describing the generated counterterm dataset."""
        raise NotImplementedError

    def _convert_sympy(self, data: Any) -> Any:
        """Recursively turn SymPy objects into JSON-serialisable structures."""
        import sympy as sp

        if isinstance(data, dict):
            converted = {}
            for key, value in data.items():
                if isinstance(key, tuple):
                    key_str = str(tuple(key))
                else:
                    key_str = str(key)
                converted[key_str] = self._convert_sympy(value)
            return converted
        if isinstance(data, list):
            return [self._convert_sympy(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self._convert_sympy(item) for item in data)
        if isinstance(data, sp.Basic):
            return sp.srepr(data)
        return data

    def to_json(self) -> dict:
        """Build the JSON payload with metadata and tensor expressions."""
        metadata, tensors = self.build_dataset()
        return {"metadata": metadata, "tensors": self._convert_sympy(tensors)}


_GENERATOR_REGISTRY: Dict[str, Type[CountertermGenerator]] = {}
_GENERATORS_LOADED = False


def _ensure_generators_loaded() -> None:
    """Import all counterterm plug-ins so they can register themselves."""

    global _GENERATORS_LOADED
    if _GENERATORS_LOADED:
        return

    package_path = Path(__file__).resolve().parent
    package_name = __name__
    for module_info in pkgutil.iter_modules([str(package_path)]):
        name = module_info.name
        if name.startswith("_") or name == "__pycache__":
            continue
        importlib.import_module(f"{package_name}.{name}")

    _GENERATORS_LOADED = True


def register_generator(generator_cls: Type[CountertermGenerator]) -> None:
    """
    Register a :class:`CountertermGenerator` subclass under its model key (and alias).

    Parameters
    ----------
    generator_cls:
        Concrete generator class providing ``model_key`` and optional ``alias``.
    """

    for key in (generator_cls.model_key, getattr(generator_cls, "alias", None)):
        if key:
            _GENERATOR_REGISTRY[key] = generator_cls


def get_generator(model_key: str) -> CountertermGenerator:
    """
    Retrieve the generator instance associated with ``model_key``.

    Parameters
    ----------
    model_key:
        Identifier supplied either as a fully-qualified module path or alias.

    Raises
    ------
    KeyError
        If no generator was registered for the provided key.
    """

    _ensure_generators_loaded()

    try:
        generator_cls = _GENERATOR_REGISTRY[model_key]
    except KeyError as exc:
        raise KeyError(
            f"No counterterm generator registered for '{model_key}'."
        ) from exc
    return generator_cls()


def generate_counterterms(
    model_key: str,
    *,
    force: bool = False,
    output: Path | None = None,
) -> Path:
    """
    Generate (or reuse) the curvature tensor dataset for the given model.

    Parameters
    ----------
    model_key:
        Identifier resolving to a registered generator class.
    force:
        If ``True`` the JSON file is recreated even when a previous version exists.
    output:
        Optional override for the output path. When omitted, the generator default
        (``models/generated/<name>.json``) is used.

    Returns
    -------
    Path
        Filesystem path of the (re)generated JSON file.
    """

    generator = get_generator(model_key)
    out_path = output or generator.output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return out_path

    payload = generator.to_json()
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path
