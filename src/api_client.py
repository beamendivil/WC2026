import hashlib
import json
import time
from datetime import datetime, timedelta, timezone

import requests

from src.api_config import (
    API_FOOTBALL_BASE_URL,
    API_MAX_RETRIES,
    API_MAX_RETRY_AFTER_SECONDS,
    API_MIN_REQUEST_INTERVAL_SECONDS,
    API_RESPONSE_CACHE_DIR,
    get_api_key,
)


class APIFootballClient:
    """Small API-Football adapter.

    Keeping provider logic here makes it easier to swap API-Football for
    another provider later without rewriting the Streamlit app.
    """

    def __init__(
        self,
        api_key=None,
        base_url=API_FOOTBALL_BASE_URL,
        cache_dir=API_RESPONSE_CACHE_DIR,
        min_request_interval=API_MIN_REQUEST_INTERVAL_SECONDS,
    ):
        self.api_key = api_key or get_api_key()
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.min_request_interval = min_request_interval
        self._last_request_at = None

    def _cache_path(self, endpoint, params):
        payload = json.dumps(
            {"endpoint": endpoint, "params": params}, sort_keys=True
        ).encode()
        digest = hashlib.sha256(payload).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

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
            return json.loads(path.read_text()).get("response", [])
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, endpoint, params, response):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(endpoint, params)
        path.write_text(json.dumps({"response": response}))

    def _throttle(self):
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            delay = self.min_request_interval - elapsed
            if delay > 0:
                time.sleep(delay)

    def _get(self, endpoint, params=None, cache_ttl=0):
        """Call one API-Football endpoint and return its response list."""
        if not self.api_key:
            raise ValueError("Missing API_FOOTBALL_KEY environment variable.")

        params = params or {}
        cached = self._read_cache(endpoint, params, cache_ttl)
        if cached is not None:
            return cached

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"x-apisports-key": self.api_key}
        for attempt in range(API_MAX_RETRIES + 1):
            self._throttle()
            response = requests.get(
                url, headers=headers, params=params, timeout=30
            )
            self._last_request_at = time.monotonic()
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= API_MAX_RETRIES:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "", 1).isdigit()
                    else 2 ** attempt
                )
                time.sleep(min(delay, API_MAX_RETRY_AFTER_SECONDS))
                continue

            response.raise_for_status()
            payload = response.json()
            errors = payload.get("errors")
            if errors:
                message = (
                    "; ".join(f"{key}: {value}" for key, value in errors.items())
                    if isinstance(errors, dict)
                    else str(errors)
                )
                raise RuntimeError(f"API-Football error: {message}")
            result = payload.get("response", [])
            if cache_ttl:
                self._write_cache(endpoint, params, result)
            return result
        return []

    def fixtures(self, league=1, season=2026):
        return self._get(
            "fixtures", {"league": league, "season": season}, cache_ttl=120
        )

    def fixture_details(self, fixture_ids):
        """Fetch full fixture payloads, including player statistics, in a batch."""
        ids = "-".join(str(fixture_id) for fixture_id in fixture_ids)
        return self._get("fixtures", {"ids": ids}, cache_ttl=86400)

    def teams(self, league=1, season=2026):
        return self._get(
            "teams", {"league": league, "season": season}, cache_ttl=86400
        )

    def countries(self):
        return self._get("countries", cache_ttl=604800)

    def standings(self, league=1, season=2026):
        return self._get(
            "standings", {"league": league, "season": season}, cache_ttl=900
        )

    def injuries(self, league=1, season=2026):
        return self._get(
            "injuries", {"league": league, "season": season}, cache_ttl=14400
        )

    def odds(self, league=1, season=2026):
        return self._get(
            "odds", {"league": league, "season": season}, cache_ttl=300
        )

    def statistics(self, fixture_id):
        return self._get(
            "fixtures/statistics", {"fixture": fixture_id}, cache_ttl=86400
        )

    def events(self, fixture_id):
        return self._get(
            "fixtures/events", {"fixture": fixture_id}, cache_ttl=86400
        )
