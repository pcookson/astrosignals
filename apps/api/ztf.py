import json
import logging
import shutil
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cache_utils import build_cache_key
from config import CACHE_DIR

logger = logging.getLogger("astrosignals.api")
router = APIRouter()
ZTF_API_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"


class ZtfIngestRequest(BaseModel):
    objectId: str | None = None
    ra: float | None = None
    dec: float | None = None
    radiusArcsec: float | None = 2
    band: str | None = "r"
    badCatflagsMask: int | None = 32768


def get_cache_path(cache_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / cache_key


def cache_rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_ztf_response_from_cache(cache_path: Path) -> dict[str, Any] | None:
    response_path = cache_path / "ztf_response.json"
    if not response_path.exists():
        return None

    try:
        cached = json.loads(response_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning(
            "ZTF cache entry is invalid, clearing %s", cache_path, exc_info=True
        )
        shutil.rmtree(cache_path, ignore_errors=True)
        return None

    required_keys = {"source", "selected", "available", "n_points", "time", "mag"}
    if not required_keys.issubset(cached.keys()):
        logger.warning(
            "ZTF cache entry missing required keys, clearing %s", cache_path
        )
        shutil.rmtree(cache_path, ignore_errors=True)
        return None

    return cached


def save_ztf_response_to_cache(
    cache_path: Path,
    cache_key: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    response_file_name = "ztf_response.json"
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


def normalized_column_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {normalized_column_name(col): col for col in columns}
    for candidate in candidates:
        match = normalized.get(normalized_column_name(candidate))
        if match is not None:
            return match
    return None


@router.post("/api/ingest/ztf")
def ingest_ztf(payload: ZtfIngestRequest) -> dict[str, Any]:
    logger.info(
        "ZTF ingest request received objectId=%s ra=%s dec=%s radiusArcsec=%s band=%s",
        payload.objectId,
        payload.ra,
        payload.dec,
        payload.radiusArcsec,
        payload.band,
    )

    object_id = (payload.objectId or "").strip()
    has_object_id = bool(object_id)
    has_ra = payload.ra is not None
    has_dec = payload.dec is not None

    if has_object_id and not object_id.isdigit():
        raise HTTPException(
            status_code=400,
            detail=(
                "For now, provide RA/Dec or a numeric ZTF object ID (digits). "
                "ZTF18... names will be supported later."
            ),
        )

    if not has_object_id and (has_ra != has_dec):
        raise HTTPException(
            status_code=400,
            detail="Provide both ra and dec for positional queries",
        )

    if not has_object_id and not (has_ra and has_dec):
        raise HTTPException(
            status_code=400,
            detail="Provide either numeric objectId, or ra+dec",
        )

    band = (payload.band or "r").strip().lower()
    if band not in {"g", "r", "i"}:
        raise HTTPException(status_code=400, detail="band must be one of: g, r, i")

    radius_arcsec = 2.0 if payload.radiusArcsec is None else float(payload.radiusArcsec)
    if radius_arcsec <= 0:
        raise HTTPException(status_code=400, detail="radiusArcsec must be > 0")
    radius_deg = radius_arcsec / 3600.0

    bad_catflags_mask = 32768 if payload.badCatflagsMask is None else payload.badCatflagsMask

    normalized_request = {
        "source": "ZTF",
        "objectId": object_id if has_object_id else None,
        "ra": float(payload.ra) if has_ra else None,
        "dec": float(payload.dec) if has_dec else None,
        "radiusArcsec": float(radius_arcsec),
        "band": band,
        "badCatflagsMask": int(bad_catflags_mask),
    }
    cache_key = build_cache_key(normalized_request)
    cache_path = get_cache_path(cache_key)

    cached = load_ztf_response_from_cache(cache_path)
    if cached is not None:
        logger.info("ZTF cache hit key=%s", cache_key)
        return {
            **cached,
            "cache": {
                "hit": True,
                "key": cache_key,
                "path": cache_rel_path(cache_path),
            },
        }

    logger.info("ZTF cache miss key=%s", cache_key)

    params: dict[str, Any] = {
        "BANDNAME": band,
        "FORMAT": "csv",
        "BAD_CATFLAGS_MASK": bad_catflags_mask,
    }
    if has_object_id:
        params["ID"] = object_id
    else:
        params["POS"] = f"CIRCLE {payload.ra} {payload.dec} {radius_deg}"

    try:
        response = requests.get(ZTF_API_URL, params=params, timeout=30)
    except requests.RequestException as exc:
        logger.exception("Failed to fetch ZTF data from IRSA")
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to reach IRSA ZTF API. "
                f"Request error: {exc}. Try again shortly."
            ),
        )

    logger.info("ZTF query URL=%s", response.url)
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"IRSA ZTF API returned status {response.status_code}. "
                "Try a larger radius or a different band."
            ),
        )

    try:
        dataframe = pd.read_csv(StringIO(response.text), comment="#")
    except Exception:
        logger.exception("Failed to parse ZTF CSV response")
        raise HTTPException(
            status_code=502,
            detail="Failed to parse IRSA ZTF CSV response",
        )

    if dataframe.empty:
        raise HTTPException(
            status_code=404,
            detail="No ZTF light curve points found. Try a larger radius or different band.",
        )

    columns = list(dataframe.columns)
    oid_col = find_column(columns, ["oid"])
    time_col = find_column(columns, ["mjd"])
    mag_col = find_column(columns, ["mag", "magnitude"])
    magerr_col = find_column(columns, ["magerr", "mag_err", "e_mag"])
    filtercode_col = find_column(columns, ["filtercode"])

    if oid_col is None or time_col is None or mag_col is None:
        raise HTTPException(
            status_code=502,
            detail=(
                "ZTF response missing required columns. "
                f"Expected oid/mjd/mag. Available columns: {columns}"
            ),
        )

    logger.info("ZTF parsed rows=%s", len(dataframe))
    dataframe = dataframe.copy()
    dataframe[oid_col] = pd.to_numeric(dataframe[oid_col], errors="coerce")
    dataframe[time_col] = pd.to_numeric(dataframe[time_col], errors="coerce")
    dataframe[mag_col] = pd.to_numeric(dataframe[mag_col], errors="coerce")

    valid_mask = (
        np.isfinite(dataframe[oid_col].to_numpy(dtype=float))
        & np.isfinite(dataframe[time_col].to_numpy(dtype=float))
        & np.isfinite(dataframe[mag_col].to_numpy(dtype=float))
    )
    if magerr_col is not None:
        dataframe[magerr_col] = pd.to_numeric(dataframe[magerr_col], errors="coerce")
        magerr_values = dataframe[magerr_col].to_numpy(dtype=float)
        valid_mask = valid_mask & np.isfinite(magerr_values) & (magerr_values <= 0.5)

    dataframe = dataframe[valid_mask]

    unique_oids = sorted(int(oid) for oid in dataframe[oid_col].dropna().unique().tolist())
    if len(unique_oids) == 0:
        raise HTTPException(status_code=404, detail="No points returned")

    if len(unique_oids) > 1:
        oid_counts = dataframe.groupby(oid_col).size().sort_values(ascending=False)
        default_oid = int(oid_counts.index[0])
    else:
        default_oid = unique_oids[0]

    dataframe = dataframe[dataframe[oid_col] == default_oid]
    if dataframe.empty:
        raise HTTPException(status_code=404, detail="No points returned")

    time = dataframe[time_col].to_numpy(dtype=float)
    mag = dataframe[mag_col].to_numpy(dtype=float)
    magerr: np.ndarray | None = None
    if magerr_col is not None:
        magerr = dataframe[magerr_col].to_numpy(dtype=float)

    unique_filters: list[str] = []
    if filtercode_col is not None:
        unique_filters = sorted(
            str(value)
            for value in dataframe[filtercode_col].dropna().astype(str).unique().tolist()
        )

    logger.info(
        "ZTF returned n_points=%s selected_oid=%s n_oids=%s",
        time.size,
        default_oid,
        len(unique_oids),
    )
    response_payload = {
        "source": "ZTF",
        "selected": {
            "oid": default_oid,
            "filtercodes": unique_filters,
        },
        "available": {
            "oids": unique_oids,
            "n_oids": len(unique_oids),
        },
        "n_points": int(time.size),
        "time": time.tolist(),
        "mag": mag.tolist(),
        "magerr": magerr.tolist() if magerr is not None else None,
    }
    try:
        save_ztf_response_to_cache(
            cache_path=cache_path,
            cache_key=cache_key,
            request_payload=normalized_request,
            response_payload=response_payload,
        )
    except Exception:
        logger.exception("Failed to persist ZTF cache entry key=%s", cache_key)

    return {
        **response_payload,
        "cache": {
            "hit": False,
            "key": cache_key,
            "path": cache_rel_path(cache_path),
        },
    }
