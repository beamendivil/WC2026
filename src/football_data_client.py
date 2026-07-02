import hashlib
import json
import time
from datetime import datetime, timedelta, timezone

import requests

from src.api_config import (
    API_MAX_RETRIES,
    API_MAX_RETRY_AFTER_SECONDS,
    API_RESPONSE_CACHE_DIR,
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_MIN_REQUEST_INTERVAL_SECONDS,
    get_football_data_key,
)


class FootballDataClient:
    """Rate-limited, disk-cached football-data.org v4 adapter."""

    def __init__(
        self,
        api_key=None,
        base_url=FOOTBALL_DATA_BASE_URL,
        cache_dir=API_RESPONSE_CACHE_DIR,
        min_request_interval=FOOTBALL_DATA_MIN_REQUEST_INTERVAL_SECONDS,
    ):
        self.api_key = api_key or get_football_data_key()
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.min_request_interval = min_request_interval
        self._last_request_at = None

    def _cache_path(self, endpoint, params):
        payload = json.dumps(
            {"provider": "football-data.org", "endpoint": endpoint, "params": params},
            sort_keys=True,
        ).encode()
        digest = hashlib.sha256(payload).hexdigest()[:20]
        return self.cache_dir / f"fd-{digest}.json"

    def _read_cache(self, endpoint, params, ttl):
        if not ttl:
            return None
        path = self._cache_path(endpoint, params)
        if not path.exists():
            return None
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        )
        if age > timedelta(seconds=ttl):
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, endpoint, params, payload):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(endpoint, params).write_text(json.dumps(payload))

    def _throttle(self):
        if self._last_request_at is None:
            return
        delay = self.min_request_interval - (
            time.monotonic() - self._last_request_at
        )
        if delay > 0:
            time.sleep(delay)

    def _get(self, endpoint, params=None, cache_ttl=0):
        if not self.api_key:
            raise ValueError("Missing FOOTBALL_DATA_API_KEY environment variable.")
        params = params or {}
        cached = self._read_cache(endpoint, params, cache_ttl)
        if cached is not None:
            return cached

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        for attempt in range(API_MAX_RETRIES + 1):
            self._throttle()
            response = requests.get(
                url,
                headers={"X-Auth-Token": self.api_key},
                params=params,
                timeout=30,
            )
            self._last_request_at = time.monotonic()
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= API_MAX_RETRIES:
                    response.raise_for_status()
                retry_after = response.headers.get("X-RequestCounter-Reset")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "", 1).isdigit()
                    else 2 ** attempt
                )
                time.sleep(min(delay, API_MAX_RETRY_AFTER_SECONDS))
                continue
            response.raise_for_status()
            payload = response.json()
            if cache_ttl:
                self._write_cache(endpoint, params, payload)
            return payload
        return {}

    def teams(self, competition="WC", season=2026):
        payload = self._get(
            f"competitions/{competition}/teams",
            {"season": season},
            cache_ttl=86400,
        )
        return payload.get("teams", [])

    def matches(self, competition="WC", season=2026):
        payload = self._get(
            f"competitions/{competition}/matches",
            {"season": season},
            cache_ttl=120,
        )
        return payload.get("matches", [])

    def standings(self, competition="WC", season=2026):
        payload = self._get(
            f"competitions/{competition}/standings",
            {"season": season},
            cache_ttl=900,
        )
        return payload.get("standings", [])

    def scorers(self, competition="WC", season=2026):
        payload = self._get(
            f"competitions/{competition}/scorers",
            {"season": season, "limit": 200},
            cache_ttl=3600,
        )
        return payload.get("scorers", [])
