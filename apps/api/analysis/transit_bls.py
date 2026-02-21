from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares


def detrend_flux(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float = 0.75,
) -> np.ndarray:
    """Detrend a light curve with a robust running-median baseline."""
    if window_days <= 0:
        raise ValueError("window_days must be > 0")

    time_arr = np.asarray(time, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    if time_arr.size != flux_arr.size:
        raise ValueError("time and flux must have equal lengths")

    diffs = np.diff(time_arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    cadence_days = float(np.median(diffs)) if diffs.size > 0 else np.nan

    if np.isfinite(cadence_days) and cadence_days > 0:
        window_points = max(5, int(round(window_days / cadence_days)))
    else:
        window_points = 101
    if window_points % 2 == 0:
        window_points += 1

    series = pd.Series(flux_arr)
    trend = series.rolling(
        window=window_points,
        center=True,
        min_periods=max(3, window_points // 5),
    ).median()
    trend = trend.interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)

    detrended = np.full_like(flux_arr, np.nan, dtype=float)
    valid_trend = np.isfinite(trend) & (trend != 0)
    detrended[valid_trend] = flux_arr[valid_trend] / trend[valid_trend]

    med = np.nanmedian(detrended)
    if np.isfinite(med) and med != 0:
        detrended = detrended / med

    return detrended


def _count_observed_transits(
    t_min: float,
    t_max: float,
    period: float,
    t0: float,
) -> int:
    if not np.isfinite(period) or period <= 0:
        return 0
    start_k = int(np.ceil((t_min - t0) / period))
    end_k = int(np.floor((t_max - t0) / period))
    return max(0, end_k - start_k + 1)


def run_bls(
    time: np.ndarray,
    flux_detrended: np.ndarray,
    flux_err: np.ndarray | None = None,
    min_period: float = 0.5,
    max_period: float | None = None,
) -> dict[str, Any]:
    time_arr = np.asarray(time, dtype=float)
    flux_arr = np.asarray(flux_detrended, dtype=float)

    valid = np.isfinite(time_arr) & np.isfinite(flux_arr)
    dy: np.ndarray | None = None
    if flux_err is not None:
        dy = np.asarray(flux_err, dtype=float)
        valid = valid & np.isfinite(dy) & (dy > 0)

    time_arr = time_arr[valid]
    flux_arr = flux_arr[valid]
    if dy is not None:
        dy = dy[valid]

    if time_arr.size < 20:
        raise ValueError("Not enough finite points for BLS search")

    baseline_days = float(time_arr.max() - time_arr.min())
    if baseline_days <= 0:
        raise ValueError("Time baseline must be > 0")

    if max_period is None:
        max_period = min(15.0, max(2.0, 0.5 * baseline_days))

    if min_period <= 0:
        raise ValueError("min_period must be > 0")

    max_period = float(max_period)
    if max_period <= min_period:
        max_period = min_period + 0.1

    durations = np.linspace(0.04, 0.33, 25)
    period_grid = np.linspace(float(min_period), max_period, 2500)

    bls = BoxLeastSquares(time_arr, flux_arr, dy=dy)
    power_result = bls.power(period_grid, durations)

    power = np.asarray(power_result.power, dtype=float)
    finite_power = np.isfinite(power)
    if not np.any(finite_power):
        raise ValueError("BLS returned no finite powers")

    best_idx = int(np.nanargmax(power))
    best_power = float(power[best_idx])

    period_values = np.asarray(power_result.period, dtype=float)
    duration_values = np.asarray(power_result.duration, dtype=float)
    transit_time_values = np.asarray(power_result.transit_time, dtype=float)
    depth_values = np.asarray(power_result.depth, dtype=float)

    best_period = float(period_values[best_idx])
    best_duration = float(duration_values[best_idx])
    best_t0 = float(transit_time_values[best_idx])
    best_depth = float(depth_values[best_idx])

    depth_snr: float | None = None
    depth_err = getattr(power_result, "depth_err", None)
    if depth_err is not None:
        depth_err_values = np.asarray(depth_err, dtype=float)
        best_depth_err = float(depth_err_values[best_idx])
        if np.isfinite(best_depth_err) and best_depth_err > 0:
            depth_snr = float(best_depth / best_depth_err)

    power_std = float(np.nanstd(power))
    sde: float | None = None
    if np.isfinite(power_std) and power_std > 0:
        sde = float((best_power - float(np.nanmedian(power))) / power_std)

    n_transits_observed = _count_observed_transits(
        t_min=float(time_arr.min()),
        t_max=float(time_arr.max()),
        period=best_period,
        t0=best_t0,
    )

    return {
        "best_period": best_period,
        "best_duration": best_duration,
        "best_depth": best_depth,
        "best_t0": best_t0,
        "best_power": best_power,
        "depth_snr": depth_snr,
        "sde": sde,
        "n_transits_observed": n_transits_observed,
        "baseline_days": baseline_days,
        "min_period_days": float(min_period),
        "max_period_days": max_period,
        "n_durations": int(durations.size),
        "n_periods": int(period_grid.size),
    }


def phase_fold(
    time: np.ndarray,
    flux_detrended: np.ndarray,
    period: float,
    t0: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_arr = np.asarray(time, dtype=float)
    flux_arr = np.asarray(flux_detrended, dtype=float)
    if period <= 0:
        raise ValueError("period must be > 0")

    phase = ((time_arr - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux_arr[order]


def bin_folded_curve(
    phase: np.ndarray,
    folded_flux: np.ndarray,
    bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    if bins < 10:
        raise ValueError("bins must be >= 10")

    phase_arr = np.asarray(phase, dtype=float)
    flux_arr = np.asarray(folded_flux, dtype=float)
    valid = np.isfinite(phase_arr) & np.isfinite(flux_arr)
    phase_arr = phase_arr[valid]
    flux_arr = flux_arr[valid]
    if phase_arr.size == 0:
        raise ValueError("No finite folded points to bin")

    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(phase_arr, edges) - 1

    means = np.full(bins, np.nan, dtype=float)
    for i in range(bins):
        in_bin = idx == i
        if np.any(in_bin):
            means[i] = float(np.mean(flux_arr[in_bin]))

    good = np.isfinite(means)
    return centers[good], means[good]
