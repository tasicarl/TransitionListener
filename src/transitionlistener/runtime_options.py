"""Optional runtime overrides for scripts and benchmark runners.

The core configuration classes live in :mod:`transitionlistener.config`.  This
module keeps input-dictionary parsing and named precision bundles out of that
defaults-only file.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import copy
from collections.abc import Mapping

from transitionlistener.config import TracingConf
from transitionlistener import errors


ALGORITHM_MODES = ("adaptive_step_size", "fixed_step_size")


PRECISION_OVERRIDE_KEYS = (
    "precision_trace_field_accuracy",
    "precision_trace_temp_accuracy",
    "precision_trace_dtstart",
    "precision_trace_tjump",
    "precision_trace_dtmin",
    "precision_trace_local_n",
    "precision_trace_local_edge",
    "precision_deform_converge_0",
    "precision_deform_fRatioConv",
)

PERCOLATION_OVERRIDE_KEYS = (
    "percolation_integral_method",
    "percolation_time_temperature_mode",
    "percolation_n_action",
    "percolation_n_action_min",
    "percolation_n_action_increment",
    "percolation_n_action_max",
    "percolation_maxit",
    "percolation_max_action_temperatures",
    "percolation_acc_tperc",
    "percolation_acc_tfinal",
    "percolation_acc_rh",
    "percolation_weak_threshold",
    "percolation_jitter_GH4_threshold",
    "percolation_action_jitter_tunneltight_rescue",
    "percolation_n_jitter_save",
)

GW_OVERRIDE_KEYS = (
    "gw_wall_velocity",
)

PRECISION_MODE_NAMES = ("default", "robust", "xtrace", "tunneltight", "benchmark")


def normalize_precision_mode(precision_mode: str | None) -> str:
    """Return a validated precision-mode label, defaulting ``None`` to ``"default"``."""
    if precision_mode is None:
        return "default"
    mode = str(precision_mode)
    if mode not in PRECISION_MODE_NAMES:
        raise ValueError(f"Unknown precision_mode {precision_mode!r}.")
    return mode


def precision_mode_profile(mode: str) -> dict:
    """Materialise the named tracing/tunneling override bundle for ``mode``."""
    default_conf = TracingConf()
    default_trace = default_conf.tracing_args
    default_tunnel = default_conf.tunneling_params["deformation_deform_params"]
    profile = {
        "precision_trace_field_accuracy": default_conf.tracing_field_accuracy,
        "precision_trace_temp_accuracy": default_conf.tracing_temp_accuracy,
        "precision_trace_dtstart": default_trace["dtstart"],
        "precision_trace_tjump": default_trace["tjump"],
        "precision_trace_dtmin": default_trace["dtmin"],
        "precision_trace_local_n": default_trace["local_min_args"]["n"],
        "precision_trace_local_edge": default_trace["local_min_args"]["edge"],
        "precision_deform_converge_0": default_tunnel["converge_0"],
        "precision_deform_fRatioConv": default_tunnel["fRatioConv"],
    }

    if mode == "robust":
        profile.update(
            precision_trace_field_accuracy=3e-4,
            precision_trace_temp_accuracy=3e-4,
            precision_trace_dtstart=5e-5,
            precision_trace_tjump=5e-6,
            precision_trace_dtmin=3e-7,
            precision_trace_local_n=200,
            precision_trace_local_edge=0.03,
        )
    if mode in {"xtrace", "benchmark"}:
        profile.update(
            precision_trace_field_accuracy=1e-4,
            precision_trace_temp_accuracy=1e-4,
            precision_trace_dtstart=2e-5,
            precision_trace_tjump=2e-6,
            precision_trace_dtmin=1e-7,
            precision_trace_local_n=300,
            precision_trace_local_edge=0.02,
        )
    if mode in {"tunneltight", "benchmark"}:
        profile.update(
            precision_deform_converge_0=1.0,
            precision_deform_fRatioConv=5e-3,
        )
    return profile


def apply_precision_overrides(tracing_conf, overrides: dict[str, float | int | None]) -> None:
    """Apply individual tracing and path-deformation overrides in place."""
    if overrides["precision_trace_field_accuracy"] is not None:
        tracing_conf.tracing_field_accuracy = float(overrides["precision_trace_field_accuracy"])
    if overrides["precision_trace_temp_accuracy"] is not None:
        tracing_conf.tracing_temp_accuracy = float(overrides["precision_trace_temp_accuracy"])

    tracing = copy.deepcopy(tracing_conf.tracing_args)
    for key, trace_key in (
        ("precision_trace_dtstart", "dtstart"),
        ("precision_trace_tjump", "tjump"),
        ("precision_trace_dtmin", "dtmin"),
    ):
        if overrides[key] is not None:
            tracing[trace_key] = float(overrides[key])

    local = dict(tracing.get("local_min_args", {}))
    if overrides["precision_trace_local_n"] is not None:
        local["n"] = int(round(float(overrides["precision_trace_local_n"])))
    if overrides["precision_trace_local_edge"] is not None:
        local["edge"] = float(overrides["precision_trace_local_edge"])
    tracing["local_min_args"] = local
    tracing_conf.tracing_args = tracing

    tunneling = copy.deepcopy(tracing_conf.tunneling_params)
    deform = dict(tunneling.get("deformation_deform_params", {}))
    if overrides["precision_deform_converge_0"] is not None:
        deform["converge_0"] = float(overrides["precision_deform_converge_0"])
    if overrides["precision_deform_fRatioConv"] is not None:
        deform["fRatioConv"] = float(overrides["precision_deform_fRatioConv"])
    tunneling["deformation_deform_params"] = deform
    tracing_conf.tunneling_params = tunneling


def apply_precision_mode(tracing_conf, precision_mode: str | None) -> str:
    """Resolve ``precision_mode`` and apply its predefined override profile."""
    mode = normalize_precision_mode(precision_mode)
    if mode != "default":
        apply_precision_overrides(tracing_conf, precision_mode_profile(mode))
    return mode


def _as_int(value, minimum: int) -> int:
    """Cast ``value`` to ``int`` while enforcing a lower bound."""
    return max(int(round(float(value))), int(minimum))


def _as_float(value, minimum: float | None = None) -> float:
    """Cast ``value`` to ``float`` while optionally enforcing a lower bound."""
    number = float(value)
    if minimum is not None:
        number = max(number, float(minimum))
    return number


def apply_percolation_overrides(percolation_conf, overrides: dict[str, object]) -> None:
    """Apply runtime percolation controls parsed from model or scan inputs."""
    overrides = {key: overrides.get(key) for key in PERCOLATION_OVERRIDE_KEYS}

    if overrides["percolation_integral_method"] is not None:
        method = str(overrides["percolation_integral_method"])
        if method not in {"ode", "double_integral"}:
            raise ValueError("percolation_integral_method must be 'ode' or 'double_integral'.")
        percolation_conf.integral_method = method
    if overrides["percolation_time_temperature_mode"] is not None:
        mode = str(overrides["percolation_time_temperature_mode"])
        if mode not in {"sound_speed", "bag"}:
            raise ValueError("percolation_time_temperature_mode must be 'sound_speed' or 'bag'.")
        percolation_conf.time_temperature_mode = mode

    int_minimums = {
        "percolation_n_action": ("n_action", 2),
        "percolation_n_action_min": ("n_action_min", 2),
        "percolation_n_action_increment": ("n_action_increment", 1),
        "percolation_n_action_max": ("n_action_max", 2),
        "percolation_maxit": ("maxit", 1),
        "percolation_max_action_temperatures": ("max_action_temperatures", 1),
        "percolation_n_jitter_save": ("n_jitter_save", 0),
    }
    for key, (attr, minimum) in int_minimums.items():
        if overrides[key] is not None:
            setattr(percolation_conf, attr, _as_int(overrides[key], minimum))

    float_minimums = {
        "percolation_acc_tperc": ("acc_tperc", 0.0),
        "percolation_acc_tfinal": ("acc_tfinal", 0.0),
        "percolation_acc_rh": ("acc_rh", 0.0),
        "percolation_weak_threshold": ("weak_threshold", 0.0),
        "percolation_jitter_GH4_threshold": ("jitter_GH4_threshold", 0.0),
    }
    for key, (attr, minimum) in float_minimums.items():
        if overrides[key] is not None:
            setattr(percolation_conf, attr, _as_float(overrides[key], minimum=minimum))

    bool_overrides = {
        "percolation_action_jitter_tunneltight_rescue": "action_jitter_tunneltight_rescue",
    }
    for key, attr in bool_overrides.items():
        if overrides[key] is not None:
            setattr(percolation_conf, attr, bool(overrides[key]))

    percolation_conf.n_action_max = max(
        int(percolation_conf.n_action_max),
        int(percolation_conf.n_action_min),
    )


def apply_gw_overrides(gw_conf, overrides: dict[str, object]) -> None:
    """Apply runtime GW-observable overrides consumed from scan inputs."""
    if overrides.get("gw_wall_velocity") is not None:
        gw_conf.wall_velocity = overrides["gw_wall_velocity"]


def apply_input_overrides(potential, args: tuple, dargs: dict) -> tuple[tuple, dict]:
    """Pop shared runtime-override keys from the model input dictionary.

    Consumes ``precision_mode``, ``percolation_algorithm_mode`` and the keyed
    override dicts; leaves the remaining entries untouched for model-specific
    parameter validation.
    """
    args = list(args)
    dargs = dict(dargs)
    inputparam_dict = None

    if args and isinstance(args[0], Mapping):
        inputparam_dict = dict(args[0])
        args[0] = inputparam_dict
    else:
        for key in ("inputparam_dict", "input_dict"):
            value = dargs.get(key)
            if isinstance(value, Mapping):
                inputparam_dict = dict(value)
                dargs[key] = inputparam_dict
                break

    if inputparam_dict is None:
        return tuple(args), dargs

    precision_mode = inputparam_dict.pop("precision_mode", None)
    algorithm_mode = inputparam_dict.pop("percolation_algorithm_mode", None)
    precision_overrides = {key: inputparam_dict.pop(key, None) for key in PRECISION_OVERRIDE_KEYS}
    percolation_overrides = {key: inputparam_dict.pop(key, None) for key in PERCOLATION_OVERRIDE_KEYS}
    gw_overrides = {key: inputparam_dict.pop(key, None) for key in GW_OVERRIDE_KEYS}

    try:
        apply_precision_mode(potential.config.tracingConf, precision_mode)
    except ValueError as exc:
        raise errors.InitPotentialError(str(exc)) from exc

    if algorithm_mode is not None:
        algorithm_mode = str(algorithm_mode)
        if algorithm_mode not in ALGORITHM_MODES:
            raise errors.InitPotentialError(
                f"percolation_algorithm_mode must be one of {ALGORITHM_MODES}, "
                f"got {algorithm_mode!r}."
            )
        potential.config.percolationConf.algorithm_mode = algorithm_mode

    apply_precision_overrides(potential.config.tracingConf, precision_overrides)
    apply_percolation_overrides(potential.config.percolationConf, percolation_overrides)
    apply_gw_overrides(potential.config.gwConf, gw_overrides)
    return tuple(args), dargs
