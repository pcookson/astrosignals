from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.optimize import least_squares  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    least_squares = None


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase values into [-0.5, 0.5)."""
    phase_arr = np.asarray(phase, dtype=float)
    return ((phase_arr + 0.5) % 1.0) - 0.5


def _sanitize_shape_params(
    duration_phase: float,
    ingress_phase: float,
) -> tuple[float, float]:
    duration = float(np.clip(duration_phase, 0.001, 0.2))
    half = 0.5 * duration
    ingress_min = 2e-4
    ingress_max = max(half - 1e-6, ingress_min)
    ingress = float(np.clip(ingress_phase, ingress_min, ingress_max))
    if ingress >= half:
        ingress = max(ingress_min, half - 1e-6)
    return duration, ingress


def _trapezoid_shape_profile(
    phase: np.ndarray,
    duration_phase: float,
    ingress_phase: float,
    phase0: float,
) -> np.ndarray:
    """Return a unit-depth profile s in [0, 1] so model = baseline - depth * s."""
    duration, ingress = _sanitize_shape_params(duration_phase, ingress_phase)
    x = np.abs(wrap_phase(np.asarray(phase, dtype=float) - float(phase0)))
    half = 0.5 * duration
    flat_half = max(half - ingress, 1e-6)

    s = np.zeros_like(x, dtype=float)
    inside_flat = x <= flat_half
    inside_ramp = (x > flat_half) & (x <= half)

    s[inside_flat] = 1.0
    if np.any(inside_ramp):
        # Linearly fall from 1 at |x|=flat_half to 0 at |x|=half.
        s[inside_ramp] = (half - x[inside_ramp]) / max(ingress, 1e-6)
        s[inside_ramp] = np.clip(s[inside_ramp], 0.0, 1.0)

    return s


def trapezoid_model(
    phase: np.ndarray,
    baseline: float,
    depth: float,
    duration_phase: float,
    ingress_phase: float,
    phase0: float,
) -> np.ndarray:
    """Simple transit trapezoid model evaluated on wrapped phase."""
    s = _trapezoid_shape_profile(
        phase=phase,
        duration_phase=duration_phase,
        ingress_phase=ingress_phase,
        phase0=phase0,
    )
    return float(baseline) - float(depth) * s


def _solve_linear_baseline_depth(
    y: np.ndarray,
    shape_profile: np.ndarray,
) -> tuple[float, float] | None:
    if y.size == 0 or shape_profile.size != y.size:
        return None

    X = np.column_stack([np.ones_like(shape_profile), -shape_profile])
    try:
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    baseline = float(coeffs[0])
    depth = float(coeffs[1])
    if not np.isfinite(baseline) or not np.isfinite(depth):
        return None
    if not (0.98 <= baseline <= 1.02):
        return None
    if not (1e-5 <= depth <= 0.05):
        return None
    return baseline, depth


def _rms_residual(y: np.ndarray, yhat: np.ndarray) -> float:
    resid = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    finite = np.isfinite(resid)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(resid[finite]))))


def _public_fit_dict(
    *,
    baseline: float,
    depth: float,
    duration_phase: float,
    ingress_phase: float,
    phase0: float,
    period_days: float,
    rms_residual: float,
) -> dict[str, float]:
    duration_hours = float(duration_phase) * float(period_days) * 24.0
    ingress_minutes = float(ingress_phase) * float(period_days) * 24.0 * 60.0
    return {
        "depth_pct_fit": float(depth) * 100.0,
        "duration_hours_fit": duration_hours,
        "ingress_minutes_fit": ingress_minutes,
        "baseline_fit": float(baseline),
        "phase0_fit": float(phase0),
        "rms_residual": float(rms_residual),
        "vshape_metric": float(ingress_phase) / max(float(duration_phase), 1e-9),
    }


def _fit_with_scipy(
    phase: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    candidate: dict[str, Any],
) -> dict[str, float] | None:
    if least_squares is None:
        return None

    depth0 = float(candidate.get("depth_pct", 0.5)) / 100.0
    duration_phase0 = (
        (float(candidate.get("duration_hours", 3.0)) / 24.0) / max(float(period_days), 1e-6)
    )
    duration_phase0 = float(np.clip(duration_phase0, 0.001, 0.2))
    ingress0 = float(np.clip(duration_phase0 * 0.2, 2e-4, max(duration_phase0 / 2.0 - 1e-6, 2e-4)))

    phase_arr = np.asarray(phase, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    valid = np.isfinite(phase_arr) & np.isfinite(flux_arr)
    phase_arr = phase_arr[valid]
    flux_arr = flux_arr[valid]
    if phase_arr.size < 10:
        return None

    def residuals(params: np.ndarray) -> np.ndarray:
        baseline, depth, duration_phase, ingress_phase, phase0 = [float(v) for v in params]
        model = trapezoid_model(
            phase_arr,
            baseline=baseline,
            depth=depth,
            duration_phase=duration_phase,
            ingress_phase=ingress_phase,
            phase0=phase0,
        )
        resid = flux_arr - model
        # Soft constraint for ingress < duration/2
        penalty = max(0.0, ingress_phase - 0.5 * duration_phase + 1e-6)
        if penalty > 0:
            resid = np.concatenate([resid, np.array([penalty * 100.0], dtype=float)])
        return resid

    x0 = np.array([1.0, np.clip(depth0, 1e-5, 0.05), duration_phase0, ingress0, 0.0], dtype=float)
    lower = np.array([0.98, 1e-5, 0.001, 0.0002, -0.02], dtype=float)
    upper = np.array([1.02, 0.05, 0.2, 0.1, 0.02], dtype=float)

    try:
        result = least_squares(residuals, x0=x0, bounds=(lower, upper), method="trf")
    except Exception:
        return None
    if not getattr(result, "success", False):
        return None

    baseline, depth, duration_phase, ingress_phase, phase0 = [float(v) for v in result.x]
    duration_phase, ingress_phase = _sanitize_shape_params(duration_phase, ingress_phase)

    model = trapezoid_model(
        phase_arr,
        baseline=baseline,
        depth=depth,
        duration_phase=duration_phase,
        ingress_phase=ingress_phase,
        phase0=phase0,
    )
    rms = _rms_residual(flux_arr, model)
    return _public_fit_dict(
        baseline=baseline,
        depth=depth,
        duration_phase=duration_phase,
        ingress_phase=ingress_phase,
        phase0=phase0,
        period_days=period_days,
        rms_residual=rms,
    )


def _fit_with_fallback(
    phase: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    candidate: dict[str, Any],
) -> dict[str, float] | None:
    phase_arr = np.asarray(phase, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    valid = np.isfinite(phase_arr) & np.isfinite(flux_arr)
    phase_arr = phase_arr[valid]
    flux_arr = flux_arr[valid]
    if phase_arr.size < 10:
        return None

    depth0 = float(candidate.get("depth_pct", 0.5)) / 100.0
    duration_phase0 = (
        (float(candidate.get("duration_hours", 3.0)) / 24.0) / max(float(period_days), 1e-6)
    )
    duration_phase0 = float(np.clip(duration_phase0, 0.001, 0.2))

    best: dict[str, float] | None = None
    best_sse = float("inf")

    def try_shape(phase0: float, duration_phase: float, ingress_frac: float) -> None:
        nonlocal best, best_sse
        duration_phase, ingress_phase = _sanitize_shape_params(
            duration_phase, max(0.0002, duration_phase * ingress_frac)
        )
        if ingress_phase >= duration_phase / 2.0:
            return
        s = _trapezoid_shape_profile(phase_arr, duration_phase, ingress_phase, phase0)
        solved = _solve_linear_baseline_depth(flux_arr, s)
        if solved is None:
            return
        baseline, depth = solved
        # Keep result near BLS depth if fallback is underconstrained.
        if depth0 > 0 and abs(depth - depth0) > 0.05:
            return
        model = baseline - depth * s
        resid = flux_arr - model
        sse = float(np.sum(resid * resid))
        if not np.isfinite(sse) or sse >= best_sse:
            return
        best_sse = sse
        best = _public_fit_dict(
            baseline=baseline,
            depth=depth,
            duration_phase=duration_phase,
            ingress_phase=ingress_phase,
            phase0=phase0,
            period_days=period_days,
            rms_residual=_rms_residual(flux_arr, model),
        )

    coarse_phase0 = np.linspace(-0.02, 0.02, 17)
    coarse_duration = np.unique(
        np.clip(
            np.array([duration_phase0 * s for s in (0.6, 0.8, 1.0, 1.2, 1.5)] + list(np.linspace(0.01, 0.1, 10))),
            0.001,
            0.2,
        )
    )
    coarse_ingress_frac = np.array([0.08, 0.12, 0.18, 0.25, 0.35, 0.45])

    for p0 in coarse_phase0:
        for dur in coarse_duration:
            for frac in coarse_ingress_frac:
                try_shape(float(p0), float(dur), float(frac))

    if best is None:
        return None

    for dphase in (0.0, -0.004, -0.002, 0.002, 0.004):
        for ddur in (0.0, -0.01, -0.005, 0.005, 0.01):
            for dfrac in (0.0, -0.08, -0.04, 0.04, 0.08):
                duration_phase = (
                    (best["duration_hours_fit"] / 24.0) / max(float(period_days), 1e-6)
                ) + ddur
                ingress_phase = (
                    best["ingress_minutes_fit"] / (24.0 * 60.0)
                ) / max(float(period_days), 1e-6)
                frac = (ingress_phase / max(duration_phase, 1e-6)) + dfrac
                try_shape(best["phase0_fit"] + dphase, duration_phase, frac)

    return best


def fit_trapezoid_to_folded(
    phase_binned: np.ndarray,
    flux_binned: np.ndarray,
    period_days: float,
    bls_candidate: dict[str, Any],
) -> dict[str, float] | None:
    """Fit a simple trapezoid transit model to binned phase-folded data."""
    if not np.isfinite(period_days) or period_days <= 0:
        return None

    fit = _fit_with_scipy(phase_binned, flux_binned, period_days, bls_candidate)
    if fit is not None:
        return fit

    return _fit_with_fallback(phase_binned, flux_binned, period_days, bls_candidate)
