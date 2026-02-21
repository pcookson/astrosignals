import numpy as np

from tess import extract_lightcurve_arrays


class FakeArray:
    def __init__(self, values: list[float], name: str | None = None):
        self.value = np.asarray(values, dtype=float)
        self.name = name


class FakeTime:
    def __init__(self, values: list[float], fmt: str = "btjd", scale: str = "tdb"):
        self.value = np.asarray(values, dtype=float)
        self.format = fmt
        self.scale = scale


class FakeLightCurve:
    def __init__(
        self,
        time: list[float],
        flux: list[float],
        flux_err: list[float] | None = None,
        quality: list[int] | None = None,
    ):
        self.time = FakeTime(time)
        self.flux = FakeArray(flux, name="pdcsap_flux")
        self.flux_err = FakeArray(flux_err) if flux_err is not None else None
        self.quality = FakeArray(quality) if quality is not None else None
        self.meta = {"BJDREFI": 2457000, "BJDREFF": 0.0}


def test_extract_lightcurve_arrays_applies_quality_mask() -> None:
    lc = FakeLightCurve(
        time=[1.0, 2.0, 3.0, 4.0],
        flux=[10.0, 11.0, 12.0, 13.0],
        flux_err=[1.0, 1.0, 1.0, 1.0],
        quality=[0, 1, 0, 4],
    )

    time, flux_norm, flux_err_norm, meta = extract_lightcurve_arrays(lc)

    assert len(time) == 2
    assert len(flux_norm) == 2
    assert flux_err_norm is not None
    assert len(flux_err_norm) == 2
    assert meta["quality_mask_applied"] is True
    assert meta["n_raw"] == 4
    assert meta["n_after_mask"] == 2


def test_extract_lightcurve_arrays_normalizes_flux_err_with_median_flux() -> None:
    lc = FakeLightCurve(
        time=[1.0, 2.0, 3.0],
        flux=[2.0, 4.0, 6.0],
        flux_err=[1.0, 2.0, 3.0],
        quality=[0, 0, 0],
    )

    _, _, flux_err_norm, _ = extract_lightcurve_arrays(lc)

    assert flux_err_norm is not None
    np.testing.assert_allclose(flux_err_norm, np.array([0.25, 0.5, 0.75]))
