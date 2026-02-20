import hashlib
import json
from typing import Any


def build_cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]
