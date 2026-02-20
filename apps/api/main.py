import json
import logging
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightkurve as lk
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cache_utils import build_cache_key
from config import CACHE_DIR

logger = logging.getLogger("astrosignals.api")

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    target: str
    mission: str = "TESS"
    author: str = "SPOC"
    sector: int | None = None


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


def extract_lightcurve_arrays(lc: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if hasattr(lc.time, "value"):
        time = np.asarray(lc.time.value, dtype=float)
    else:
        time = np.asarray(lc.time.to_value(), dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)

    flux_err = None
    if getattr(lc, "flux_err", None) is not None:
        flux_err = np.asarray(lc.flux_err.value, dtype=float)

    valid = ~(np.isnan(time) | np.isnan(flux))
    time = time[valid]
    flux = flux[valid]
    if flux_err is not None:
        flux_err = flux_err[valid]

    if time.size == 0:
        raise HTTPException(
            status_code=422, detail="No valid points available after NaN filtering"
        )

    median_flux = np.median(flux)
    if not np.isfinite(median_flux) or median_flux == 0:
        raise HTTPException(
            status_code=422, detail="Unable to normalize flux by median value"
        )

    flux_norm = flux / median_flux
    return time, flux_norm, flux_err


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/api/ingest")
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

    time, flux_norm, flux_err = extract_lightcurve_arrays(lightcurve)
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
    }
