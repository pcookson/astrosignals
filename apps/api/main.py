import logging
from typing import Any

import lightkurve as lk
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
logger = logging.getLogger("astrosignals.api")

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

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


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/api/ingest")
def ingest(payload: IngestRequest) -> dict[str, Any]:
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

    logger.info(
        "Ingest request received target=%s mission=%s author=%s sector=%s",
        target,
        mission,
        payload.author,
        payload.sector,
    )

    try:
        search = lk.search_lightcurve(
            target,
            mission=mission,
            author=payload.author,
            sector=payload.sector if payload.sector is not None else None,
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

    try:
        downloaded = search[0].download()
    except Exception:
        logger.exception("Lightkurve download failed")
        raise HTTPException(
            status_code=502, detail="Failed to download light curve data"
        )

    if downloaded is None:
        raise HTTPException(status_code=404, detail="No downloadable light curve found")

    logger.info("Download success")
    lc = downloaded if hasattr(downloaded, "time") else downloaded.to_lightcurve()

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
    logger.info("n_points=%s", time.size)

    return {
        "target": target,
        "mission": mission,
        "author": payload.author,
        "n_points": int(time.size),
        "time": time.tolist(),
        "flux": flux_norm.tolist(),
        "flux_err": flux_err.tolist() if flux_err is not None else None,
    }
