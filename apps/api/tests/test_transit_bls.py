import numpy as np

from analysis.transit_bls import detrend_flux, run_bls


def test_bls_recovers_synthetic_transit_period() -> None:
    rng = np.random.default_rng(7)

    true_period = 3.29
    true_t0 = 1.2
    duration_days = 0.12
    depth = 0.01

    cadence_days = 1.0 / 48.0
    time = np.arange(0.0, 30.0, cadence_days)
    flux = 1.0 + rng.normal(0.0, 0.0015, size=time.size)

    phase_time = ((time - true_t0 + 0.5 * true_period) % true_period) - 0.5 * true_period
    in_transit = np.abs(phase_time) < (duration_days / 2.0)
    flux[in_transit] -= depth

    flux_detrended = detrend_flux(time, flux, window_days=0.75)
    result = run_bls(
        time=time,
        flux_detrended=flux_detrended,
        min_period=1.0,
        max_period=5.0,
    )

    assert abs(result["best_period"] - true_period) < 0.12
