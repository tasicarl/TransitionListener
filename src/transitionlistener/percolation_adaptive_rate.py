"""Rate-peak and hot-onset helpers for adaptive_step_size percolation.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import math

import numpy as np
from scipy import interpolate
from scipy import optimize

from transitionlistener.bubbledynamics import _temperature_grid

HOT_RATE_REFINE_LOG = 1.0
HOT_RATE_ACCEPT_LOG = 0.0
HOT_RATE_ANCHORS = (3.0, 2.0, 1.0, 0.0)
HOT_PROBABILITY_ANCHORS = (1.0e-3, 3.0e-3, 1.0e-2)
HOT_SHOULDER_LOG10_RATE_THRESHOLD = 1.0
HOT_SHOULDER_MAX_LOG_STEP = 0.3
HOT_SHOULDER_ONSET_P_ZERO = 1.0e-8
HOT_SHOULDER_ONSET_MAX_LOG_STEP = 0.05

RATE_PEAK_REFINE_FLOOR_LOG10 = -10.0
RATE_PEAK_ONRAMP_FLOOR_LOG10 = -18.0
RATE_PEAK_ONRAMP_MIN_GAIN_LOG10 = 8.0
RATE_PEAK_ONRAMP_MIN_LOG_WIDTH = 0.12
SUBCRITICAL_RATE_PEAK_MIN_PROMINENCE = 1.0
RATE_PEAK_INERT_MIN_PEAK_LOG10 = -3.0
RATE_PEAK_INERT_LOG10 = -10.0
RATE_PEAK_INERT_MAX_INTERVAL_DI = 1.0e-4
RATE_PEAK_INERT_MIN_MONOTONE_POINTS = 2
RATE_PEAK_INERT_MONOTONE_TOLERANCE_LOG10 = 0.25


def _temperature_midpoint(high: float, low: float) -> float | None:
    """Return a geometric or arithmetic midpoint suited to logarithmic temperature scans."""
    high = float(high)
    low = float(low)
    if not (np.isfinite(high) and np.isfinite(low)) or high <= low:
        return None
    if high > 0.0 and low > 0.0 and high / low >= 2.0:
        return float(np.sqrt(high * low))
    return 0.5 * (high + low)


def _temperature_tolerance(
    values: np.ndarray | list[float] | tuple[float, ...] | None,
    candidate: float,
) -> float:
    """Return the absolute tolerance used when deduplicating temperature samples."""
    candidate = float(candidate)
    values_arr = np.asarray([] if values is None else values, dtype=float)
    scale = max(
        abs(candidate),
        np.max(np.abs(values_arr)) if values_arr.size else 0.0,
        1.0,
    )
    return 1e-6 * scale


def _temperature_present(
    values: np.ndarray | list[float] | tuple[float, ...] | None,
    candidate: float,
) -> bool:
    """Return whether ``candidate`` is already represented inside ``values``."""
    values_arr = np.asarray([] if values is None else values, dtype=float)
    if values_arr.size == 0:
        return False
    return bool(np.any(np.isclose(values_arr, float(candidate), atol=_temperature_tolerance(values_arr, candidate), rtol=0.0)))


def _find_value_crossing(
    temperatures: np.ndarray,
    values: np.ndarray,
    target: float,
) -> tuple[float | None, float | None, int | None]:
    """Locate the first linearised crossing of ``values`` through ``target``."""
    temps = np.asarray(temperatures, dtype=float)
    sampled = np.asarray(values, dtype=float)
    if temps.size != sampled.size or temps.size < 2:
        return None, None, None

    bracket_index = None
    for index in range(sampled.size - 1):
        value_high = float(sampled[index])
        value_low = float(sampled[index + 1])
        if not np.isfinite(value_high) or not np.isfinite(value_low):
            continue
        if value_high == target or value_low == target or (
            value_high < target < value_low or value_low < target < value_high
        ):
            bracket_index = index
            break
    if bracket_index is None:
        return None, None, None

    i_hi = int(bracket_index)
    i_lo = min(i_hi + 1, len(temps) - 1)
    T_hi = float(temps[i_hi])
    T_lo = float(temps[i_lo])
    V_hi = float(sampled[i_hi])
    V_lo = float(sampled[i_lo])
    if not (np.isfinite(T_hi) and np.isfinite(T_lo) and np.isfinite(V_hi) and np.isfinite(V_lo)):
        return None, None, None
    if abs(V_lo - V_hi) < 1e-15:
        return 0.5 * (T_hi + T_lo), abs(T_hi - T_lo), i_hi
    crossing = T_hi + (float(target) - V_hi) / (V_lo - V_hi) * (T_lo - T_hi)
    return float(crossing), abs(T_hi - T_lo), i_hi


def _probability_at(temperature: float, temperatures: np.ndarray, probabilities: np.ndarray) -> float | None:
    """Interpolate the false-vacuum probability profile at one temperature."""
    temps = np.asarray(temperatures, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    finite = np.isfinite(temps) & np.isfinite(probs)
    temps = temps[finite]
    probs = probs[finite]
    if temps.size < 2:
        return None
    temperature = float(temperature)
    scale = max(np.max(np.abs(temps)), abs(temperature), 1.0)
    matches = np.flatnonzero(np.isclose(temps, temperature, atol=1e-10 * scale, rtol=0.0))
    if matches.size:
        value = float(probs[int(matches[0])])
        return value if np.isfinite(value) else None
    order = np.argsort(temps)
    temps = temps[order]
    probs = probs[order]
    if np.any(np.diff(temps) <= 0.0):
        return None
    value = float(np.interp(temperature, temps, probs, left=np.nan, right=np.nan))
    if not np.isfinite(value):
        return None
    return float(min(max(value, 0.0), 1.0))


def _remesh_dp_limit(probability_high: float, probability_low: float, settings) -> float:
    """Choose the allowed probability jump for a remeshed interval based on its regime."""
    generic = max(float(settings.large_delta_p_refine_threshold), 0.0)
    strict = 0.05
    p_min = min(float(probability_high), float(probability_low))
    p_max = max(float(probability_high), float(probability_low))
    if p_max <= 0.2 or p_min < 0.2:
        return strict
    if p_min >= 0.8 or p_max > 0.8:
        return min(generic, strict) if generic > 0.0 else strict
    return generic


def _log_gamma_h4_array(
    temperatures: np.ndarray,
    actions: np.ndarray,
    hubble: np.ndarray,
) -> np.ndarray:
    """Vectorized ``log(Gamma/H^4)`` for support arrays."""

    from transitionlistener import bubbledynamics as bd

    temps = np.asarray(temperatures, dtype=float)
    acts = np.asarray(actions, dtype=float)
    hub = np.asarray(hubble, dtype=float)
    result = np.full(temps.shape, np.nan, dtype=float)
    valid = np.isfinite(temps) & np.isfinite(acts) & np.isfinite(hub) & (temps > 0.0) & (hub > 0.0)
    if np.any(valid):
        result[valid] = np.asarray(bd.logGamma(temps[valid], acts[valid]), dtype=float) - 4.0 * np.log(hub[valid])
    return result


def _log10_gamma_h4_array(
    temperatures: np.ndarray,
    actions: np.ndarray,
    hubble: np.ndarray,
) -> np.ndarray:
    """Vectorized ``log10(Gamma/H^4)`` for support arrays."""

    return _log_gamma_h4_array(temperatures, actions, hubble) / math.log(10.0)


def _refine_step1_rate_peak(
    temperatures: np.ndarray,
    log_gamma_h4_sym: np.ndarray,
) -> tuple[float | None, float | None, int | None]:
    """Return one interior peak of log(Gamma/H^4_sym), refined from cached scout points."""

    temps = np.asarray(temperatures, dtype=float)
    log_rate = np.asarray(log_gamma_h4_sym, dtype=float)
    if temps.size != log_rate.size or temps.size < 3:
        return None, None, None

    peak_index = int(np.nanargmax(log_rate))
    if peak_index <= 0 or peak_index >= temps.size - 1:
        return None, None, None

    peak_value = float(log_rate[peak_index])
    left_neighbor = float(log_rate[peak_index - 1])
    right_neighbor = float(log_rate[peak_index + 1])
    boundary_prominence = peak_value - max(float(log_rate[0]), float(log_rate[-1]))
    if not (peak_value > left_neighbor and peak_value > right_neighbor and boundary_prominence > 1.0):
        return None, None, None

    x = np.asarray(temps[::-1], dtype=float)
    y = np.asarray(log_rate[::-1], dtype=float)
    interpolant = interpolate.PchipInterpolator(x, y, extrapolate=False)
    lower = float(temps[peak_index + 1])
    upper = float(temps[peak_index - 1])
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        return float(temps[peak_index]), peak_value, peak_index

    try:
        optimum = optimize.minimize_scalar(
            lambda value: -float(interpolant(float(value))),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": max(1e-10 * max(abs(lower), abs(upper), 1.0), 1e-14)},
        )
    except Exception:
        optimum = None

    if optimum is None or not bool(getattr(optimum, "success", False)):
        return float(temps[peak_index]), peak_value, peak_index

    peak_temperature = float(optimum.x)
    peak_log_rate = float(interpolant(peak_temperature))
    if not (np.isfinite(peak_temperature) and np.isfinite(peak_log_rate)):
        return float(temps[peak_index]), peak_value, peak_index
    return peak_temperature, peak_log_rate, peak_index


def _build_peak_centered_step1_window(
    temperatures: np.ndarray,
    log_gamma_h4_sym: np.ndarray,
    settings,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Build a startup window around the interior rate peak of a U-shaped action."""

    temps = np.asarray(temperatures, dtype=float)
    log_rate = np.asarray(log_gamma_h4_sym, dtype=float)
    peak_temperature, peak_log_rate, peak_index = _refine_step1_rate_peak(temps, log_rate)
    if peak_temperature is None or peak_log_rate is None or peak_index is None:
        return None, None, None, None

    drop = max(4.0, -math.log(max(float(settings.f_start), 1e-12)))
    threshold = float(peak_log_rate) - float(drop)

    start_index = int(peak_index)
    while start_index > 0 and float(log_rate[start_index - 1]) >= threshold:
        start_index -= 1
    end_index = int(peak_index)
    while end_index < len(log_rate) - 1 and float(log_rate[end_index + 1]) >= threshold:
        end_index += 1

    window_high = float(temps[start_index])
    window_low = float(temps[end_index])

    if start_index > 0:
        T_hi = float(temps[start_index - 1])
        T_lo = float(temps[start_index])
        y_hi = float(log_rate[start_index - 1])
        y_lo = float(log_rate[start_index])
        if np.isfinite(y_hi) and np.isfinite(y_lo) and abs(y_lo - y_hi) > 1e-15:
            frac = (threshold - y_hi) / (y_lo - y_hi)
            if 0.0 <= frac <= 1.0:
                window_high = float(T_hi + frac * (T_lo - T_hi))

    if end_index < len(log_rate) - 1:
        T_hi = float(temps[end_index])
        T_lo = float(temps[end_index + 1])
        y_hi = float(log_rate[end_index])
        y_lo = float(log_rate[end_index + 1])
        if np.isfinite(y_hi) and np.isfinite(y_lo) and abs(y_lo - y_hi) > 1e-15:
            frac = (threshold - y_hi) / (y_lo - y_hi)
            if 0.0 <= frac <= 1.0:
                window_low = float(T_hi + frac * (T_lo - T_hi))

    if not (np.isfinite(window_high) and np.isfinite(window_low) and window_high > window_low):
        return None, None, None, None
    return float(peak_temperature), float(peak_log_rate), float(window_high), float(window_low)


def _build_log_temperature_bridge_grid(
    hot_temperature: float,
    cold_temperature: float,
    *,
    max_log_step: float,
    max_new_points: int,
) -> np.ndarray | None:
    """Return geometric bridge points so the interval obeys one ``Delta ln T`` limit."""

    hot_temperature = float(hot_temperature)
    cold_temperature = float(cold_temperature)
    if (
        max_new_points <= 0
        or not np.isfinite(hot_temperature)
        or not np.isfinite(cold_temperature)
        or hot_temperature <= cold_temperature
        or cold_temperature <= 0.0
        or hot_temperature <= 0.0
    ):
        return None

    dln_t = float(math.log(hot_temperature / cold_temperature))
    if not np.isfinite(dln_t) or dln_t <= float(max_log_step):
        return None

    n_segments = max(int(math.ceil(dln_t / max(float(max_log_step), 1.0e-6))), 2)
    n_points = min(max(int(n_segments - 1), 1), int(max_new_points))
    bridge = np.geomspace(hot_temperature, cold_temperature, n_points + 2, dtype=float)[1:-1]
    bridge = _temperature_grid(bridge)
    return bridge if bridge.size > 0 else None


def _build_dynamiczoomwindow_hot_shoulder_grid(
    temperatures: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    hubble: np.ndarray,
    settings,
    *,
    max_new_points: int,
) -> tuple[np.ndarray | None, str | None]:
    """Resolve low-P hot-shoulder gaps where the local rate is already active."""

    if max_new_points <= 0:
        return None, None

    temps = np.asarray(temperatures, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    acts = np.asarray(actions, dtype=float)
    hub = np.asarray(hubble, dtype=float)
    if not (temps.size >= 2 and temps.size == probs.size == acts.size == hub.size):
        return None, None
    order = np.argsort(-temps)
    temps = temps[order]
    probs = probs[order]
    acts = acts[order]
    hub = hub[order]

    p_upper = max(max(HOT_PROBABILITY_ANCHORS), float(settings.f_start))
    rate_threshold = HOT_SHOULDER_LOG10_RATE_THRESHOLD
    max_log_step = HOT_SHOULDER_MAX_LOG_STEP
    onset_max_log_step = HOT_SHOULDER_ONSET_MAX_LOG_STEP
    onset_p_zero = HOT_SHOULDER_ONSET_P_ZERO
    best_interval: tuple[float, float, float, float, bool] | None = None
    log10_rate = _log10_gamma_h4_array(temps, acts, hub)

    for i in range(temps.size - 1):
        hot_temperature = float(temps[i])
        cold_temperature = float(temps[i + 1])
        hot_probability = float(probs[i])
        cold_probability = float(probs[i + 1])
        if max(hot_probability, cold_probability) > p_upper + 1.0e-12:
            continue
        if hot_temperature <= 0.0 or cold_temperature <= 0.0:
            continue
        dln_t = float(math.log(hot_temperature / cold_temperature))
        if not np.isfinite(dln_t):
            continue
        is_onset_crossing = hot_probability < onset_p_zero and cold_probability > onset_p_zero
        effective_threshold = onset_max_log_step if is_onset_crossing else max_log_step
        if dln_t <= effective_threshold:
            continue
        endpoint_rates = log10_rate[[i, i + 1]]
        finite_endpoint_rates = endpoint_rates[np.isfinite(endpoint_rates)]
        rate_level = float(np.max(finite_endpoint_rates)) if finite_endpoint_rates.size else -np.inf
        rate_active = np.isfinite(rate_level) and rate_level > rate_threshold
        onset_buried = is_onset_crossing or (
            hot_probability < onset_p_zero
            and cold_probability < onset_p_zero
            and (i + 2) < temps.size
            and float(probs[i + 2]) > onset_p_zero
        )
        if not (rate_active or onset_buried):
            continue
        if best_interval is None or dln_t > best_interval[3]:
            best_interval = (hot_temperature, cold_temperature, rate_level, dln_t, is_onset_crossing)

    if best_interval is None:
        return None, None

    hot_temperature, cold_temperature, rate_level, dln_t, is_onset_crossing = best_interval
    bridge_max_log_step = onset_max_log_step if is_onset_crossing else max_log_step
    candidate = _build_log_temperature_bridge_grid(
        hot_temperature,
        cold_temperature,
        max_log_step=bridge_max_log_step,
        max_new_points=int(max_new_points),
    )
    if candidate is None or candidate.size == 0:
        return None, None
    mode_tag = "onset_crossing" if is_onset_crossing else "low-P"
    reason = (
        f"hot_shoulder_refine from {mode_tag} interval "
        f"T={hot_temperature:2.5g}->{cold_temperature:2.5g}, "
        f"dlnT={dln_t:2.3g}, max log10(Gamma/H^4)={rate_level:2.3g}"
    )
    return candidate, reason


def _build_dynamiczoomwindow_rate_peak_grid(
    temperatures: np.ndarray,
    actions: np.ndarray,
    hubble: np.ndarray,
    settings,
    *,
    max_new_points: int,
) -> tuple[np.ndarray | None, str | None]:
    """Refine around a sampled local maximum or unresolved on-ramp of ``Gamma/H^4``."""

    if max_new_points <= 0:
        return None, None

    temps = np.asarray(temperatures, dtype=float)
    acts = np.asarray(actions, dtype=float)
    hub = np.asarray(hubble, dtype=float)
    if not (temps.size >= 3 and temps.size == acts.size == hub.size):
        return None, None

    order = np.argsort(-temps)
    temps = temps[order]
    acts = acts[order]
    hub = hub[order]
    log10_rate = _log10_gamma_h4_array(temps, acts, hub)
    finite = np.isfinite(temps) & np.isfinite(log10_rate) & (temps > 0.0)
    if np.count_nonzero(finite) < 2:
        return None, None

    candidate = np.asarray([], dtype=float)
    occupied = _temperature_grid(temps)

    def add_candidate(value: float | None) -> bool:
        nonlocal candidate
        if value is None:
            return False
        try:
            numeric = float(value)
        except Exception:
            return False
        if not np.isfinite(numeric):
            return False
        if _temperature_present(occupied, numeric) or _temperature_present(candidate, numeric):
            return False
        candidate = _temperature_grid(np.append(candidate, numeric))
        return True

    def add_interval_points(high: float, low: float, budget: int) -> None:
        if budget <= 0:
            return
        high = float(high)
        low = float(low)
        if not (np.isfinite(high) and np.isfinite(low) and high > low and low > 0.0):
            return
        if high / low >= 1.25:
            points = np.geomspace(high, low, int(budget) + 2, dtype=float)[1:-1]
        else:
            points = np.linspace(high, low, int(budget) + 2, dtype=float)[1:-1]
        for point in points:
            add_candidate(float(point))

    def build_rate_onramp_candidate() -> tuple[np.ndarray | None, str | None]:
        nonlocal candidate
        candidate = np.asarray([], dtype=float)
        rate_floor = RATE_PEAK_REFINE_FLOOR_LOG10
        onramp_floor = RATE_PEAK_ONRAMP_FLOOR_LOG10
        min_rate_gain = RATE_PEAK_ONRAMP_MIN_GAIN_LOG10
        min_log_width = RATE_PEAK_ONRAMP_MIN_LOG_WIDTH

        best_interval: tuple[float, int, float, float, float, float] | None = None
        for index in range(temps.size - 1):
            if not (finite[index] and finite[index + 1]):
                continue
            hot_temperature = float(temps[index])
            cold_temperature = float(temps[index + 1])
            if not (hot_temperature > cold_temperature > 0.0):
                continue
            hot_log_rate = float(log10_rate[index])
            cold_log_rate = float(log10_rate[index + 1])
            rate_gain = cold_log_rate - hot_log_rate
            log_width = float(math.log(hot_temperature / cold_temperature))
            if not (np.isfinite(rate_gain) and np.isfinite(log_width)):
                continue
            if rate_gain < min_rate_gain or log_width < min_log_width:
                continue
            if cold_log_rate < onramp_floor and max(hot_log_rate, cold_log_rate) < rate_floor:
                continue

            score = log_width * max(rate_gain, 0.0) * (1.0 + max(cold_log_rate - onramp_floor, 0.0))
            if best_interval is None or score > best_interval[0]:
                best_interval = (score, index, hot_temperature, cold_temperature, hot_log_rate, cold_log_rate)

        if best_interval is None:
            return None, None

        _, _, hot_temperature, cold_temperature, hot_log_rate, cold_log_rate = best_interval
        for target in (rate_floor, -5.0, 0.0):
            if candidate.size >= int(max_new_points):
                break
            if min(hot_log_rate, cold_log_rate) <= float(target) <= max(hot_log_rate, cold_log_rate):
                if abs(cold_log_rate - hot_log_rate) > 1.0e-15:
                    frac = (float(target) - hot_log_rate) / (cold_log_rate - hot_log_rate)
                    log_hot = math.log(hot_temperature)
                    log_cold = math.log(cold_temperature)
                    add_candidate(float(math.exp(log_hot + frac * (log_cold - log_hot))))

        if candidate.size < int(max_new_points):
            add_interval_points(
                hot_temperature,
                cold_temperature,
                int(max_new_points) - int(candidate.size),
            )

        if candidate.size == 0:
            return None, None
        reason = (
            "rate_peak_refine on unresolved on-ramp "
            f"T={hot_temperature:2.5g}->{cold_temperature:2.5g}, "
            f"log10(Gamma/H^4)={hot_log_rate:2.3g}->{cold_log_rate:2.3g}"
        )
        return candidate[: int(max_new_points)], reason

    def add_rate_crossings(target: float) -> None:
        for index in range(temps.size - 1):
            left = float(log10_rate[index])
            right = float(log10_rate[index + 1])
            if not (np.isfinite(left) and np.isfinite(right)):
                continue
            if (left - target) * (right - target) > 0.0:
                continue
            if abs(right - left) < 1.0e-15:
                add_candidate(0.5 * (float(temps[index]) + float(temps[index + 1])))
                continue
            frac = (float(target) - left) / (right - left)
            if 0.0 <= frac <= 1.0:
                add_candidate(float(temps[index]) + frac * (float(temps[index + 1]) - float(temps[index])))

    def add_largest_rate_band_gaps(max_points: int) -> None:
        if max_points <= 0:
            return
        rate_floor = RATE_PEAK_REFINE_FLOOR_LOG10
        band_points: list[float] = []
        for temp, rate in zip(temps, log10_rate):
            if np.isfinite(rate) and float(rate) >= rate_floor:
                band_points.append(float(temp))
        band_points.extend(float(value) for value in candidate.tolist())
        band_points = [
            float(value)
            for value in _temperature_grid(band_points)
            if np.isfinite(float(value)) and float(value) > 0.0
        ]
        if len(band_points) < 2:
            return
        for _ in range(int(max_points)):
            points = _temperature_grid(band_points)
            gaps: list[tuple[float, float, float]] = []
            for high, low in zip(points[:-1], points[1:]):
                if float(high) <= float(low) or float(low) <= 0.0:
                    continue
                gaps.append((float(math.log(float(high) / float(low))), float(high), float(low)))
            if not gaps:
                return
            _, high, low = max(gaps, key=lambda item: item[0])
            new_point = _temperature_midpoint(high, low)
            if not add_candidate(new_point):
                return
            band_points.append(float(new_point))

    budget = max(int(max_new_points), 1)
    rate_floor = RATE_PEAK_REFINE_FLOOR_LOG10
    onramp_grid, onramp_reason = build_rate_onramp_candidate()
    if onramp_grid is not None and onramp_grid.size > 0:
        return onramp_grid, onramp_reason

    if np.count_nonzero(finite) < 3:
        return None, None
    finite_indices = np.flatnonzero(finite)
    peak_index = int(finite_indices[int(np.nanargmax(log10_rate[finite]))])
    if peak_index <= 0 or peak_index >= temps.size - 1:
        return None, None
    if not (finite[peak_index - 1] and finite[peak_index + 1]):
        return None, None

    peak_rate = float(log10_rate[peak_index])
    hot_rate = float(log10_rate[peak_index - 1])
    cold_rate = float(log10_rate[peak_index + 1])
    if not (peak_rate > hot_rate and peak_rate > cold_rate):
        return None, None
    prominence = peak_rate - max(hot_rate, cold_rate)
    min_prominence = SUBCRITICAL_RATE_PEAK_MIN_PROMINENCE
    if prominence < min_prominence:
        return None, None

    candidate = np.asarray([], dtype=float)
    if peak_rate >= rate_floor:
        peak_temperature, _, _ = _refine_step1_rate_peak(temps, log10_rate)
        add_candidate(peak_temperature)
        for target in (rate_floor, -5.0, 0.0):
            if candidate.size >= budget:
                break
            if float(target) <= peak_rate + 1.0e-12:
                add_rate_crossings(float(target))
        if candidate.size < budget:
            add_largest_rate_band_gaps(budget - int(candidate.size))
    else:
        hot_budget = max(1, budget // 2)
        cold_budget = max(1, budget - hot_budget)
        if budget == 1:
            hot_gap = math.log(float(temps[peak_index - 1]) / float(temps[peak_index]))
            cold_gap = math.log(float(temps[peak_index]) / float(temps[peak_index + 1]))
            if hot_gap >= cold_gap:
                add_interval_points(float(temps[peak_index - 1]), float(temps[peak_index]), 1)
            else:
                add_interval_points(float(temps[peak_index]), float(temps[peak_index + 1]), 1)
        else:
            add_interval_points(float(temps[peak_index - 1]), float(temps[peak_index]), hot_budget)
            add_interval_points(float(temps[peak_index]), float(temps[peak_index + 1]), cold_budget)

    if candidate.size == 0:
        return None, None
    candidate = candidate[: int(max_new_points)]
    reason = (
        "rate_peak_refine around "
        f"T={float(temps[peak_index]):2.5g}, "
        f"max log10(Gamma/H^4)={peak_rate:2.3g}, "
        f"prominence={prominence:2.3g}"
    )
    return candidate, reason


def _hot_head_underresolved(
    temperatures: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    hubble: np.ndarray,
    settings,
    *,
    for_acceptance: bool,
) -> tuple[bool, float | None, float | None]:
    """Return whether the hottest retained support point still starts too late."""

    temps = np.asarray(temperatures, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    acts = np.asarray(actions, dtype=float)
    hub = np.asarray(hubble, dtype=float)
    if temps.size == 0 or probs.size == 0 or acts.size == 0 or hub.size == 0:
        return False, None, None
    if not (temps.size == probs.size == acts.size == hub.size):
        return False, None, None

    probability_hot = float(probs[0]) if np.isfinite(probs[0]) else None
    log_rate = _log_gamma_h4_array(temps, acts, hub)
    log_gamma_h4_hot = float(log_rate[0]) if log_rate.size and np.isfinite(log_rate[0]) else None
    if probability_hot is None or log_gamma_h4_hot is None:
        return False, probability_hot, log_gamma_h4_hot

    if probability_hot > float(settings.f_start):
        return False, probability_hot, log_gamma_h4_hot

    threshold = HOT_RATE_ACCEPT_LOG if for_acceptance else HOT_RATE_REFINE_LOG
    if log_gamma_h4_hot <= threshold:
        return False, probability_hot, log_gamma_h4_hot

    hot_band_upper = float(min(HOT_PROBABILITY_ANCHORS))
    hot_band_upper = max(float(hot_band_upper), 1.0e-12)
    hot_band_mask = np.isfinite(probs) & (probs <= hot_band_upper + 1.0e-12)
    if np.count_nonzero(hot_band_mask) >= 3:
        hot_band_indices = np.flatnonzero(hot_band_mask)
        hot_band_probs = probs[hot_band_mask]
        band_reaches_anchor = bool(np.nanmax(hot_band_probs) >= 0.5 * hot_band_upper)
        if band_reaches_anchor:
            resolved_index = int(hot_band_indices[int(np.nanargmax(hot_band_probs))])
            resolved_log_rate = float(log_rate[resolved_index])
            if np.isfinite(resolved_log_rate) and resolved_log_rate <= threshold:
                return False, probability_hot, log_gamma_h4_hot

    return True, probability_hot, log_gamma_h4_hot


def _probability_anchor_presence_tolerance(
    target: float,
    settings,
) -> float:
    """Return the probability-space tolerance used to mark one anchor as covered."""

    target = float(target)
    threshold = _remesh_dp_limit(target, target, settings)
    tolerance = 0.35 * float(threshold) if threshold > 0.0 else float("inf")

    if any(
        math.isclose(
            target,
            level,
            rel_tol=0.0,
            abs_tol=max(1.0e-12, 1.0e-9 * max(abs(level), abs(target), 1.0)),
        )
        for level in HOT_PROBABILITY_ANCHORS
    ):
        tolerance = min(tolerance, max(1.0e-4, 0.25 * abs(target)))
    return float(tolerance)


def _build_step1_hot_rate_anchor_grid(
    profile,
    settings,
    current_grid: np.ndarray,
) -> np.ndarray | None:
    """Carry a few hotter ``log(Gamma/H^4)`` anchors from the scout into the active grid."""

    temps = np.asarray(profile.temperatures, dtype=float)
    probs = np.asarray(profile.probabilities, dtype=float)
    log_rate = np.asarray(profile.log_gamma_h4_sym, dtype=float)
    active = _temperature_grid(current_grid)
    if temps.size < 2 or temps.size != probs.size or temps.size != log_rate.size or active.size == 0:
        return None

    candidate = np.asarray([], dtype=float)
    current_hot = float(active[0])
    scale = max(np.max(np.abs(active)), np.max(np.abs(temps)), 1.0)
    atol = 1e-10 * scale
    hot_probability = float(probs[0]) if np.isfinite(probs[0]) else None
    hot_log_rate = float(log_rate[0]) if np.isfinite(log_rate[0]) else None
    if hot_probability is None or hot_log_rate is None:
        return None
    if hot_probability > float(settings.f_start):
        return None

    levels = sorted(
        {
            float(value)
            for value in (
                *HOT_RATE_ANCHORS,
                HOT_RATE_REFINE_LOG,
                HOT_RATE_ACCEPT_LOG,
            )
            if np.isfinite(float(value))
        },
        reverse=True,
    )

    def add_candidate(value: float | None) -> None:
        nonlocal candidate
        if value is None or not np.isfinite(float(value)):
            return
        if float(value) <= current_hot + atol:
            return
        if _temperature_present(active, float(value)) or _temperature_present(candidate, float(value)):
            return
        candidate = _temperature_grid(np.append(candidate, float(value)))

    if hot_log_rate > HOT_RATE_ACCEPT_LOG:
        add_candidate(float(temps[0]))

    for level in levels:
        if hot_log_rate < float(level):
            continue
        crossing, _, _ = _find_value_crossing(temps, log_rate, float(level))
        add_candidate(crossing)

    return candidate if candidate.size > 0 else None


def _build_step1_hot_probability_anchor_grid(
    profile,
    settings,
    current_grid: np.ndarray,
) -> np.ndarray | None:
    """Carry explicit hot-side probability anchors from the scout into the active grid."""

    temps = np.asarray(profile.temperatures, dtype=float)
    probs = np.asarray(profile.probabilities, dtype=float)
    active = _temperature_grid(current_grid)
    if temps.size < 2 or temps.size != probs.size or active.size == 0:
        return None

    candidate = np.asarray([], dtype=float)

    def occupied_probabilities(temperatures_to_sample: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for value in np.asarray(temperatures_to_sample, dtype=float):
            probability = _probability_at(value, temps, probs)
            values.append(float("nan") if probability is None else float(probability))
        return np.asarray(values, dtype=float)

    occupied_probs = occupied_probabilities(active)

    def add_candidate(value: float | None) -> None:
        nonlocal candidate, active, occupied_probs
        if value is None or not np.isfinite(float(value)):
            return
        numeric = float(value)
        if _temperature_present(active, numeric) or _temperature_present(candidate, numeric):
            return
        candidate = _temperature_grid(np.append(candidate, numeric))
        active = _temperature_grid(np.append(active, numeric))
        occupied_probs = occupied_probabilities(active)

    p_min = float(np.nanmin(probs))
    p_max = float(np.nanmax(probs))
    for target in HOT_PROBABILITY_ANCHORS:
        if not (p_min + 1.0e-12 < target < p_max - 1.0e-12):
            continue
        tolerance = _probability_anchor_presence_tolerance(target, settings)
        if np.any(np.isfinite(occupied_probs) & (np.abs(occupied_probs - target) <= tolerance)):
            continue
        crossing, _, crossing_index = _find_value_crossing(temps, probs, target)
        if crossing_index is None:
            continue
        add_candidate(crossing)

    return candidate if candidate.size > 0 else None


def _build_dynamiczoomwindow_hot_head_grid(
    temperatures: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    hubble: np.ndarray,
    settings,
    *,
    new_high: float,
    max_new_points: int,
) -> tuple[np.ndarray | None, str | None]:
    """Return a focused hot-side refinement batch for unresolved onset support."""

    if max_new_points <= 0:
        return None, None

    temps = np.asarray(temperatures, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    acts = np.asarray(actions, dtype=float)
    hub = np.asarray(hubble, dtype=float)
    if not (temps.size == probs.size == acts.size == hub.size) or temps.size < 2:
        return None, None

    finite_mask = np.isfinite(temps) & np.isfinite(probs)
    temps = temps[finite_mask]
    probs = probs[finite_mask]
    acts = acts[finite_mask]
    hub = hub[finite_mask]
    if temps.size < 2:
        return None, None

    current = _temperature_grid(temps)
    if current.size < 2:
        return None, None

    candidate = np.asarray([], dtype=float)
    occupied = current.copy()
    added_labels: list[str] = []

    def occupied_probabilities(temperatures_to_sample: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for value in np.asarray(temperatures_to_sample, dtype=float):
            probability = _probability_at(value, temps, probs)
            values.append(float("nan") if probability is None else float(probability))
        return np.asarray(values, dtype=float)

    occupied_probs = occupied_probabilities(occupied)

    def add_candidate(value: float | None, label: str) -> bool:
        nonlocal candidate, occupied, occupied_probs
        if value is None or not np.isfinite(float(value)):
            return False
        numeric = float(value)
        if _temperature_present(occupied, numeric) or _temperature_present(candidate, numeric):
            return False
        candidate = _temperature_grid(np.append(candidate, numeric))
        occupied = _temperature_grid(np.append(occupied, numeric))
        occupied_probs = occupied_probabilities(occupied)
        added_labels.append(label)
        return True

    p_min = float(np.nanmin(probs))
    p_max = float(np.nanmax(probs))
    for target in HOT_PROBABILITY_ANCHORS:
        if candidate.size >= int(max_new_points):
            break
        if not (p_min + 1.0e-12 < target < p_max - 1.0e-12):
            continue
        tolerance = _probability_anchor_presence_tolerance(target, settings)
        if np.any(np.isfinite(occupied_probs) & (np.abs(occupied_probs - target) <= tolerance)):
            continue
        crossing, _, crossing_index = _find_value_crossing(temps, probs, target)
        if crossing_index is None:
            continue
        add_candidate(crossing, f"P={target:2.3g}")

    log_rate = _log_gamma_h4_array(temps, acts, hub)
    hot_log_rate = float(log_rate[0]) if log_rate.size and np.isfinite(log_rate[0]) else None
    if hot_log_rate is not None:
        levels = sorted(
            {
                float(value)
                for value in (
                    *HOT_RATE_ANCHORS,
                    HOT_RATE_REFINE_LOG,
                    HOT_RATE_ACCEPT_LOG,
                )
                if np.isfinite(float(value))
            },
            reverse=True,
        )
        for level in levels:
            if candidate.size >= int(max_new_points):
                break
            if hot_log_rate < float(level):
                continue
            crossing, _, crossing_index = _find_value_crossing(temps, log_rate, float(level))
            if crossing_index is None:
                continue
            add_candidate(crossing, f"logR={level:2.3g}")

    remaining = int(max_new_points) - int(candidate.size)
    current_high = float(current[0])
    target_high = float(max(new_high, current_high))
    if remaining > 0 and target_high > current_high:
        n_high = min(remaining, 2)
        if current_high > 0.0 and target_high / current_high >= 2.0:
            extra_hot = np.geomspace(target_high, current_high, n_high + 1, dtype=float)[:-1]
        else:
            extra_hot = np.linspace(target_high, current_high, n_high + 1, dtype=float)[:-1]
        for value in extra_hot:
            if candidate.size >= int(max_new_points):
                break
            add_candidate(value, f"hot@{float(value):2.5g}")

    if candidate.size == 0:
        return None, None
    return candidate, "hot_head_refine added " + ", ".join(added_labels)


def _rate_interval_is_post_peak_inert(
    temperatures: np.ndarray,
    actions: np.ndarray | None,
    hubble: np.ndarray | None,
    interval_high: float,
    interval_low: float,
    settings,
    *,
    vw: float = 1.0,
) -> bool:
    """Return whether one interval is safely beyond a resolved rate peak."""

    if actions is None or hubble is None:
        return False
    temps = np.asarray(temperatures, dtype=float)
    acts = np.asarray(actions, dtype=float)
    hub = np.asarray(hubble, dtype=float)
    if temps.size < 3 or not (temps.size == acts.size == hub.size):
        return False

    order = np.argsort(-temps)
    temps = temps[order]
    acts = acts[order]
    hub = hub[order]
    log10_rate = _log10_gamma_h4_array(temps, acts, hub)
    finite = np.isfinite(temps) & np.isfinite(log10_rate) & (temps > 0.0)
    if np.count_nonzero(finite) < 3:
        return False

    finite_indices = np.flatnonzero(finite)
    peak_index = int(finite_indices[int(np.nanargmax(log10_rate[finite]))])
    if peak_index <= 0 or peak_index >= temps.size - 1:
        return False

    peak_rate = float(log10_rate[peak_index])
    if peak_rate < RATE_PEAK_INERT_MIN_PEAK_LOG10:
        return False

    high = float(interval_high)
    low = float(interval_low)
    if not (np.isfinite(high) and np.isfinite(low) and high > low):
        return False

    scale = max(np.max(np.abs(temps)), abs(high), abs(low), 1.0)
    atol = _temperature_tolerance(temps, high) + _temperature_tolerance(temps, low) + 1.0e-12 * scale
    high_matches = np.flatnonzero(np.isclose(temps, high, rtol=0.0, atol=atol))
    low_matches = np.flatnonzero(np.isclose(temps, low, rtol=0.0, atol=atol))
    if high_matches.size == 0 or low_matches.size == 0:
        return False
    high_index = int(high_matches[0])
    low_index = int(low_matches[0])
    if high_index >= low_index:
        return False

    endpoint_rates = log10_rate[[high_index, low_index]]
    if not np.all(np.isfinite(endpoint_rates)):
        return False
    if float(np.max(endpoint_rates)) > RATE_PEAK_INERT_LOG10:
        return False

    max_interval_dI = RATE_PEAK_INERT_MAX_INTERVAL_DI
    if max_interval_dI > 0.0:
        log_a = np.full_like(log10_rate, -np.inf, dtype=float)
        valid_a = np.isfinite(log10_rate) & np.isfinite(hub) & (hub > 0.0) & np.isfinite(temps) & (temps > 0.0)
        log_a[valid_a] = log10_rate[valid_a] * math.log(10.0) + 3.0 * np.log(hub[valid_a]) - 4.0 * np.log(temps[valid_a])
        inv_h = np.where(np.isfinite(hub) & (hub > 0.0), 1.0 / hub, 0.0)
        volume_to_cold = np.zeros_like(temps, dtype=float)
        for index in range(max(temps.size - 2, -1), -1, -1):
            dt = abs(float(temps[index]) - float(temps[index + 1]))
            volume_to_cold[index] = volume_to_cold[index + 1] + 0.5 * dt * (
                abs(float(inv_h[index])) + abs(float(inv_h[index + 1]))
            )

        interval_log_a = log_a[[high_index, low_index]]
        if not np.all(np.isfinite(interval_log_a)):
            return False
        max_log_a = float(np.max(interval_log_a))
        max_volume = float(np.max(volume_to_cold[[high_index, low_index]]))
        interval_width = abs(float(high) - float(low))
        wall_velocity = max(float(vw), 0.0)
        if max_volume <= 0.0 or interval_width <= 0.0 or wall_velocity <= 0.0:
            interval_dI_bound = 0.0
        else:
            log_bound = (
                math.log(4.0 * math.pi / 3.0)
                + 3.0 * math.log(wall_velocity)
                + max_log_a
                + 3.0 * math.log(max_volume)
                + math.log(interval_width)
            )
            if log_bound > 700.0:
                interval_dI_bound = float("inf")
            elif log_bound < -745.0:
                interval_dI_bound = 0.0
            else:
                interval_dI_bound = float(math.exp(log_bound))
        if not np.isfinite(interval_dI_bound) or interval_dI_bound > max_interval_dI:
            return False

    min_monotone_points = RATE_PEAK_INERT_MIN_MONOTONE_POINTS
    monotone_tolerance = RATE_PEAK_INERT_MONOTONE_TOLERANCE_LOG10

    if high_index > peak_index and low_index > peak_index:
        shoulder_rates = log10_rate[peak_index : low_index + 1]
        if np.count_nonzero(np.isfinite(shoulder_rates)) < min_monotone_points + 1:
            return False
        diffs = np.diff(shoulder_rates[np.isfinite(shoulder_rates)])
        return bool(np.all(diffs <= monotone_tolerance))
    if high_index < peak_index and low_index < peak_index:
        shoulder_rates = log10_rate[high_index : peak_index + 1]
        if np.count_nonzero(np.isfinite(shoulder_rates)) < min_monotone_points + 1:
            return False
        diffs = np.diff(shoulder_rates[np.isfinite(shoulder_rates)])
        return bool(np.all(diffs >= -monotone_tolerance))
    return False
