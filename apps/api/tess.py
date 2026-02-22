import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import lightkurve as lk
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from analysis.transit_bls import bin_folded_curve, detrend_flux, phase_fold, run_bls
from analysis.transit_trapezoid import fit_trapezoid_to_folded, trapezoid_model
from cache_utils import build_cache_key
from config import CACHE_DIR

logger = logging.getLogger("astrosignals.api")
router = APIRouter()


class IngestRequest(BaseModel):
    target: str
    mission: str = "TESS"
    author: str = "SPOC"
    sector: int | None = None


class TransitSearchRequest(BaseModel):
    target: str
    mission: str = "TESS"
    author: str = "SPOC"
    sector: int | None = None
    min_period_days: float = 0.5
    max_period_days: float | None = None
    detrend_window_days: float = 0.75
    fold_bins: int = 200


def normalize_payload(payload: IngestRequest) -> dict[str, Any]:
    mission_raw = payload.mission.strip().upper()
    if mission_raw not in {"TESS", "KEPLER"}:
        raise HTTPException(
            status_code=400, detail="mission must be one of: TESS, Kepler"
        )

    mission = "TESS" if mission_raw == "TESS" else "Kepler"
    target = payload.target.strip()
    if not target:
        raise HTTPException(status_code=400, detail="target must not be empty")

    if mission == "TESS" and target.isdigit():
        target = f"TIC {target}"

    author = payload.author.strip() or "SPOC"

    return {
        "target": target,
        "mission": mission,
        "author": author,
        "sector": payload.sector,
    }


def get_cache_path(cache_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / cache_key


def cache_rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_lightcurve_from_cache(cache_path: Path) -> tuple[Any, dict[str, Any]] | None:
    meta_path = cache_path / "meta.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        saved_files = meta.get("saved_files") or []

        for file_name in saved_files:
            file_path = cache_path / file_name
            if file_path.exists():
                logger.info("Cache hit: reading %s", file_path)
                return lk.read(file_path), meta

        raise FileNotFoundError("No cached file listed in meta.json exists")
    except Exception:
        logger.warning("Cache entry is invalid, clearing %s", cache_path, exc_info=True)
        shutil.rmtree(cache_path, ignore_errors=True)
        return None


def load_json_response_from_cache(
    cache_path: Path, file_name: str, required_keys: set[str]
) -> dict[str, Any] | None:
    response_path = cache_path / file_name
    if not response_path.exists():
        return None

    try:
        cached = json.loads(response_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Cache entry is invalid, clearing %s", cache_path, exc_info=True)
        shutil.rmtree(cache_path, ignore_errors=True)
        return None

    if not required_keys.issubset(cached.keys()):
        logger.warning("Cache entry missing required keys, clearing %s", cache_path)
        shutil.rmtree(cache_path, ignore_errors=True)
        return None

    return cached


def save_json_response_to_cache(
    cache_path: Path,
    cache_key: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    response_file_name: str,
) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / response_file_name).write_text(
        json.dumps(response_payload), encoding="utf-8"
    )
    meta = {
        "request": request_payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
        "saved_files": [response_file_name],
    }
    (cache_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def candidate_source_paths(downloaded: Any) -> list[Path]:
    candidates: list[Path] = []

    for attr in ("path", "filepath", "filename", "file", "_path"):
        value = getattr(downloaded, attr, None)
        if isinstance(value, (str, Path)):
            candidates.append(Path(value))
        elif isinstance(value, (list, tuple)):
            candidates.extend(Path(v) for v in value if isinstance(v, (str, Path)))

    meta = getattr(downloaded, "meta", None)
    if isinstance(meta, dict):
        for key in ("FILENAME", "filename", "FILEPATH", "filepath", "path"):
            value = meta.get(key)
            if isinstance(value, (str, Path)):
                candidates.append(Path(value))

    deduped: list[Path] = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def save_lightcurve_to_cache(downloaded: Any, lightcurve: Any, cache_path: Path) -> list[str]:
    cache_path.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    for source in candidate_source_paths(downloaded):
        try:
            if not source.exists() or not source.is_file():
                continue
            destination = cache_path / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            saved_files.append(destination.name)
            break
        except Exception:
            logger.warning("Failed to copy candidate cache file %s", source, exc_info=True)

    if not saved_files:
        destination = cache_path / "lightcurve.fits"
        lightcurve.to_fits(path=destination, overwrite=True)
        saved_files.append(destination.name)

    return saved_files


def to_lightcurve(downloaded: Any) -> Any:
    if hasattr(downloaded, "time"):
        return downloaded
    if hasattr(downloaded, "to_lightcurve"):
        return downloaded.to_lightcurve()
    raise HTTPException(status_code=502, detail="Downloaded product is not a light curve")


def _array_from_column(column: Any) -> np.ndarray:
    if hasattr(column, "value"):
        return np.asarray(column.value, dtype=float)
    if hasattr(column, "to_value"):
        return np.asarray(column.to_value(), dtype=float)
    return np.asarray(column, dtype=float)


def extract_lightcurve_arrays(
    lc: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    if hasattr(lc.time, "value"):
        time = np.asarray(lc.time.value, dtype=float)
    else:
        time = np.asarray(lc.time.to_value(), dtype=float)
    flux = _array_from_column(lc.flux)
    n_raw = int(time.size)

    flux_err = None
    if getattr(lc, "flux_err", None) is not None:
        flux_err = _array_from_column(lc.flux_err)

    valid = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        valid = valid & np.isfinite(flux_err)

     # ---- Build masks with breakdown stats ----
    finite_mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        finite_mask &= np.isfinite(flux_err)

    n_finite = int(np.count_nonzero(finite_mask))
    n_removed_nonfinite = int(n_raw - n_finite)
    
    quality_mask_applied = False
    quality_mask = np.ones_like(finite_mask, dtype=bool)
    n_quality_pass = None
    n_removed_quality = None

    quality = getattr(lc, "quality", None)
    if quality is not None:
        quality_values = _array_from_column(quality)
        # quality must be finite and == 0
        quality_mask = np.isfinite(quality_values) & (quality_values == 0)
        quality_mask_applied = True

        n_quality_pass = int(np.count_nonzero(quality_mask))
        n_removed_quality = int(n_raw - n_quality_pass)
    else:
        logger.info("QUALITY column unavailable; applying finite filtering only")

    n_quality_pass_raw = int(np.count_nonzero(quality_mask))
    n_removed_quality_raw = int(n_raw - n_quality_pass_raw)

    valid = finite_mask & quality_mask
    n_after_mask = int(np.count_nonzero(valid))
    n_removed_quality_after_finite = int(n_finite - n_after_mask)

    time = time[valid]
    flux = flux[valid]
    if flux_err is not None:
        flux_err = flux_err[valid]
    n_after_mask = int(time.size)

    if time.size == 0:
        raise HTTPException(
            status_code=422,
            detail="No valid points available after finite/quality filtering",
        )

    median_flux = np.median(flux)
    if not np.isfinite(median_flux) or median_flux == 0:
        raise HTTPException(
            status_code=422, detail="Unable to normalize flux by median value"
        )

    flux_norm = flux / median_flux
    flux_err_norm = flux_err / median_flux if flux_err is not None else None

    time_format = str(getattr(lc.time, "format", "") or "").upper() or "BTJD"
    time_scale = str(getattr(lc.time, "scale", "") or "").upper() or "TDB"
    time_zero_point: float | None = None
    time_system_source = "default"

    lc_meta = getattr(lc, "meta", {})
    if isinstance(lc_meta, dict):
        bjd_ref_i = lc_meta.get("BJDREFI")
        bjd_ref_f = lc_meta.get("BJDREFF")
        if bjd_ref_i is not None or bjd_ref_f is not None:
            time_zero_point = float(bjd_ref_i or 0.0) + float(bjd_ref_f or 0.0)
            time_system_source = "from_lc_meta"

    if time_zero_point is None and time_format == "BTJD":
        time_zero_point = 2457000.0
        if time_system_source != "from_lc_meta":
            time_system_source = "default_btjd"

    flux_field_used = "flux"
    flux_name = getattr(lc.flux, "name", None)
    if isinstance(flux_name, str) and flux_name.strip():
        flux_field_used = flux_name

    dt_days_med = np.nan
    cadence_seconds = np.nan
    if time.size > 1:
        dt = np.diff(time)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size > 0:
            dt_days_med = float(np.median(dt))
            cadence_seconds = dt_days_med * 86400.0

    t_start = float(time.min())
    t_end = float(time.max())
    logger.info(
        "TESS ingest stats n_raw=%s n_finite=%s n_after_mask=%s removed_nonfinite=%s " 
        "removed_quality_raw=%s " 
        "removed_quality_after_finite=%s "
        "removed_quality=%s "
        "t_start=%s t_end=%s cadence_seconds=%s flux_field_used=%s quality_mask_applied=%s",
        n_raw,
        n_finite,
        n_after_mask,
        n_removed_nonfinite,
        n_removed_quality_raw,
        n_removed_quality_after_finite,
        n_removed_quality,
        t_start,
        t_end,
        cadence_seconds,
        flux_field_used,
        quality_mask_applied,
    )

    expected_ranges = (
        (10.0, 30.0),
        (100.0, 140.0),
        (500.0, 700.0),
        (1500.0, 2100.0),
    )
    if np.isfinite(cadence_seconds) and not any(
        low <= cadence_seconds <= high for low, high in expected_ranges
    ):
        logger.warning(
            "TESS cadence sanity warning cadence_seconds=%s n_points=%s",
            cadence_seconds,
            n_after_mask,
        )

    meta = {
        "time_format": time_format,
        "time_scale": time_scale,
        "time_zero_point": time_zero_point,
        "time_system_source": time_system_source,
        "quality_mask_applied": quality_mask_applied,
        "n_raw": n_raw,
        "n_after_mask": n_after_mask,
        "cadence_seconds": cadence_seconds if np.isfinite(cadence_seconds) else None,
        "flux_field_used": flux_field_used,
    }

    return time, flux_norm, flux_err_norm, meta


@router.post("/api/ingest")
def ingest(payload: IngestRequest) -> dict[str, Any]:
    normalized = normalize_payload(payload)
    cache_key = build_cache_key(normalized)
    cache_path = get_cache_path(cache_key)

    logger.info(
        "Ingest request received target=%s mission=%s author=%s sector=%s",
        normalized["target"],
        normalized["mission"],
        normalized["author"],
        normalized["sector"],
    )

    cached = load_lightcurve_from_cache(cache_path)
    cache_hit = cached is not None
    lightcurve = None

    if cache_hit:
        logger.info("Cache hit key=%s", cache_key)
        lightcurve = to_lightcurve(cached[0])
    else:
        logger.info("Cache miss key=%s", cache_key)
        try:
            search = lk.search_lightcurve(
                normalized["target"],
                mission=normalized["mission"],
                author=normalized["author"],
                sector=normalized["sector"] if normalized["sector"] is not None else None,
            )
        except Exception:
            logger.exception("Lightkurve search failed")
            raise HTTPException(
                status_code=502, detail="Failed to search for light curve data"
            )

        results_count = len(search)
        logger.info("Search results count=%s", results_count)
        if results_count == 0:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No light curves found. Try a different author, or omit sector to "
                    "broaden the search."
                ),
            )

        logger.info("Download start key=%s", cache_key)
        try:
            downloaded = search[0].download()
        except Exception:
            logger.exception("Lightkurve download failed")
            raise HTTPException(
                status_code=502, detail="Failed to download light curve data"
            )

        if downloaded is None:
            raise HTTPException(status_code=404, detail="No downloadable light curve found")

        logger.info("Download success key=%s", cache_key)
        lightcurve = to_lightcurve(downloaded)

        try:
            saved_files = save_lightcurve_to_cache(downloaded, lightcurve, cache_path)
            meta = {
                "request": normalized,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cache_key": cache_key,
                "saved_files": saved_files,
                "search_result": {"description": str(search[0])},
            }
            (cache_path / "meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("Failed to persist cache entry key=%s", cache_key)

    time, flux_norm, flux_err, ingest_meta = extract_lightcurve_arrays(lightcurve)
    logger.info("n_points=%s", time.size)

    return {
        "target": normalized["target"],
        "mission": normalized["mission"],
        "author": normalized["author"],
        "n_points": int(time.size),
        "time": time.tolist(),
        "flux": flux_norm.tolist(),
        "flux_err": flux_err.tolist() if flux_err is not None else None,
        "cache": {
            "hit": cache_hit,
            "key": cache_key,
            "path": cache_rel_path(cache_path),
        },
        "meta": ingest_meta,
    }


@router.post("/api/tess/transit-search")
def transit_search(payload: TransitSearchRequest) -> dict[str, Any]:
    started = perf_counter()

    if payload.detrend_window_days <= 0:
        raise HTTPException(status_code=400, detail="detrend_window_days must be > 0")
    if payload.min_period_days <= 0:
        raise HTTPException(status_code=400, detail="min_period_days must be > 0")
    if payload.fold_bins < 10:
        raise HTTPException(status_code=400, detail="fold_bins must be >= 10")

    normalized = normalize_payload(
        IngestRequest(
            target=payload.target,
            mission=payload.mission,
            author=payload.author,
            sector=payload.sector,
        )
    )

    cache_request = {
        "source": "TESS_TRANSIT_BLS",
        "response_schema_version": 2,
        "target": normalized["target"],
        "mission": normalized["mission"],
        "author": normalized["author"],
        "sector": normalized["sector"],
        "min_period_days": float(payload.min_period_days),
        "max_period_days": (
            None if payload.max_period_days is None else float(payload.max_period_days)
        ),
        "detrend_window_days": float(payload.detrend_window_days),
        "fold_bins": int(payload.fold_bins),
    }
    cache_key = build_cache_key(cache_request)
    cache_path = get_cache_path(cache_key)

    cached = load_json_response_from_cache(
        cache_path=cache_path,
        file_name="transit_search_response.json",
        required_keys={"found"},
    )
    if cached is not None:
        logger.info("Transit search cache hit key=%s", cache_key)
        return {
            **cached,
            "cache": {
                "hit": True,
                "key": cache_key,
                "path": cache_rel_path(cache_path),
            },
        }

    ingest_result = ingest(
        IngestRequest(
            target=payload.target,
            mission=payload.mission,
            author=payload.author,
            sector=payload.sector,
        )
    )
    time = np.asarray(ingest_result["time"], dtype=float)
    flux_norm = np.asarray(ingest_result["flux"], dtype=float)
    flux_err_norm = (
        np.asarray(ingest_result["flux_err"], dtype=float)
        if ingest_result.get("flux_err") is not None
        else None
    )
    ingest_meta = ingest_result.get("meta") or {}

    baseline_days = float(time.max() - time.min()) if time.size > 1 else 0.0
    if time.size < 20 or baseline_days <= payload.min_period_days:
        response_payload = {
            "found": False,
            "reason": "Insufficient baseline or points for a reliable transit search",
            "diagnostics": {
                "baseline_days": baseline_days,
                "cadence_seconds": ingest_meta.get("cadence_seconds"),
                "search": {
                    "min_period_days": float(payload.min_period_days),
                    "max_period_days": payload.max_period_days,
                    "n_durations": 0,
                    "n_periods": 0,
                },
            },
        }
        save_json_response_to_cache(
            cache_path=cache_path,
            cache_key=cache_key,
            request_payload=cache_request,
            response_payload=response_payload,
            response_file_name="transit_search_response.json",
        )
        return {
            **response_payload,
            "cache": {
                "hit": False,
                "key": cache_key,
                "path": cache_rel_path(cache_path),
            },
        }

    flux_detrended = detrend_flux(
        time=time,
        flux=flux_norm,
        window_days=float(payload.detrend_window_days),
    )
    try:
        bls_result = run_bls(
            time=time,
            flux_detrended=flux_detrended,
            flux_err=flux_err_norm,
            min_period=float(payload.min_period_days),
            max_period=payload.max_period_days,
        )
    except ValueError as exc:
        response_payload = {"found": False, "reason": str(exc)}
        save_json_response_to_cache(
            cache_path=cache_path,
            cache_key=cache_key,
            request_payload=cache_request,
            response_payload=response_payload,
            response_file_name="transit_search_response.json",
        )
        return {
            **response_payload,
            "cache": {
                "hit": False,
                "key": cache_key,
                "path": cache_rel_path(cache_path),
            },
        }

    phase, folded_flux = phase_fold(
        time=time,
        flux_detrended=flux_detrended,
        period=float(bls_result["best_period"]),
        t0=float(bls_result["best_t0"]),
    )
    phase_binned, flux_binned = bin_folded_curve(
        phase=phase,
        folded_flux=folded_flux,
        bins=int(payload.fold_bins),
    )

    candidate = {
        "period_days": float(bls_result["best_period"]),
        "t0_btjd": float(bls_result["best_t0"]),
        "duration_hours": float(bls_result["best_duration"]) * 24.0,
        "depth_pct": float(bls_result["best_depth"]) * 100.0,
        "depth_snr": bls_result["depth_snr"],
        "sde": bls_result["sde"],
        "n_transits_observed": int(bls_result["n_transits_observed"]),
        "time_system": {
            "format": ingest_meta.get("time_format", "BTJD"),
            "zero_point": ingest_meta.get("time_zero_point"),
            "scale": ingest_meta.get("time_scale", "TDB"),
        },
    }

    folded_payload: dict[str, Any] = {
        "phase": phase_binned.tolist(),
        "flux": flux_binned.tolist(),
        "bins": int(payload.fold_bins),
    }

    try:
        fit = fit_trapezoid_to_folded(
            phase_binned=phase_binned,
            flux_binned=flux_binned,
            period_days=float(candidate["period_days"]),
            bls_candidate=candidate,
        )
        candidate["fit"] = fit

        if fit is not None:
            model_phase = np.linspace(-0.5, 0.5, 1000, endpoint=False, dtype=float)
            duration_phase_fit = (
                (float(fit["duration_hours_fit"]) / 24.0)
                / max(float(candidate["period_days"]), 1e-9)
            )
            ingress_phase_fit = (
                (float(fit["ingress_minutes_fit"]) / (24.0 * 60.0))
                / max(float(candidate["period_days"]), 1e-9)
            )
            model_flux = trapezoid_model(
                model_phase,
                baseline=float(fit["baseline_fit"]),
                depth=float(fit["depth_pct_fit"]) / 100.0,
                duration_phase=duration_phase_fit,
                ingress_phase=ingress_phase_fit,
                phase0=float(fit["phase0_fit"]),
            )
            folded_payload["model"] = {
                "phase": model_phase.tolist(),
                "flux": model_flux.tolist(),
            }
            logger.info(
                "Transit trapezoid fit target=%s period=%s depth_pct_fit=%s duration_hr_fit=%s ingress_min_fit=%s rms=%s vshape=%s",
                normalized["target"],
                candidate["period_days"],
                fit["depth_pct_fit"],
                fit["duration_hours_fit"],
                fit["ingress_minutes_fit"],
                fit["rms_residual"],
                fit["vshape_metric"],
            )
        else:
            logger.info(
                "Transit trapezoid fit skipped/failed target=%s period=%s reason=no_fit",
                normalized["target"],
                candidate["period_days"],
            )
    except Exception:
        candidate["fit"] = None
        logger.warning(
            "Transit trapezoid fit failed target=%s period=%s",
            normalized["target"],
            candidate["period_days"],
            exc_info=True,
        )

    response_payload = {
        "found": True,
        "candidate": candidate,
        "folded": folded_payload,
        "diagnostics": {
            "baseline_days": float(bls_result["baseline_days"]),
            "cadence_seconds": ingest_meta.get("cadence_seconds"),
            "search": {
                "min_period_days": float(bls_result["min_period_days"]),
                "max_period_days": float(bls_result["max_period_days"]),
                "n_durations": int(bls_result["n_durations"]),
                "n_periods": int(bls_result["n_periods"]),
            },
        },
    }

    elapsed_ms = (perf_counter() - started) * 1000.0
    logger.info(
        "TESS transit search target=%s n_points=%s baseline_days=%s best_period=%s best_depth_pct=%s best_duration_hr=%s sde=%s elapsed_ms=%s",
        normalized["target"],
        time.size,
        float(bls_result["baseline_days"]),
        candidate["period_days"],
        candidate["depth_pct"],
        candidate["duration_hours"],
        candidate["sde"],
        round(elapsed_ms, 2),
    )

    save_json_response_to_cache(
        cache_path=cache_path,
        cache_key=cache_key,
        request_payload=cache_request,
        response_payload=response_payload,
        response_file_name="transit_search_response.json",
    )
    return {
        **response_payload,
        "cache": {
            "hit": False,
            "key": cache_key,
            "path": cache_rel_path(cache_path),
        },
    }
