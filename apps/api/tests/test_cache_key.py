import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cache_utils import build_cache_key


def test_cache_key_stability_same_inputs() -> None:
    payload = {
        "target": "TIC 25155310",
        "mission": "TESS",
        "author": "SPOC",
        "sector": None,
    }

    assert build_cache_key(payload) == build_cache_key(payload)


def test_cache_key_changes_for_sector() -> None:
    base = {
        "target": "TIC 25155310",
        "mission": "TESS",
        "author": "SPOC",
        "sector": None,
    }
    with_sector = {
        "target": "TIC 25155310",
        "mission": "TESS",
        "author": "SPOC",
        "sector": 10,
    }

    assert build_cache_key(base) != build_cache_key(with_sector)


def test_cache_key_changes_for_source_payloads() -> None:
    tess_payload = {
        "source": "TESS",
        "target": "TIC 25155310",
        "mission": "TESS",
        "author": "SPOC",
        "sector": None,
    }
    ztf_payload = {
        "source": "ZTF",
        "objectId": None,
        "ra": 298.0025,
        "dec": 29.87147,
        "radiusArcsec": 5.0,
        "band": "r",
        "badCatflagsMask": 32768,
    }

    assert build_cache_key(tess_payload) != build_cache_key(ztf_payload)
