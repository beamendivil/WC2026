import requests

from src.api_config import API_FOOTBALL_BASE_URL, get_api_key


class APIFootballClient:
    """Small API-Football adapter.

    Keeping provider logic here makes it easier to swap API-Football for
    another provider later without rewriting the Streamlit app.
    """

    def __init__(self, api_key=None, base_url=API_FOOTBALL_BASE_URL):
        self.api_key = api_key or get_api_key()
        self.base_url = base_url.rstrip("/")

    def _get(self, endpoint, params=None):
        """Call one API-Football endpoint and return its response list."""
        if not self.api_key:
            raise ValueError("Missing API_FOOTBALL_KEY environment variable.")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"x-apisports-key": self.api_key}
        response = requests.get(url, headers=headers, params=params or {}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", [])

    def fixtures(self, league=1, season=2026):
        return self._get("fixtures", {"league": league, "season": season})

    def teams(self, league=1, season=2026):
        return self._get("teams", {"league": league, "season": season})

    def standings(self, league=1, season=2026):
        return self._get("standings", {"league": league, "season": season})

    def injuries(self, league=1, season=2026):
        return self._get("injuries", {"league": league, "season": season})

    def odds(self, league=1, season=2026):
        return self._get("odds", {"league": league, "season": season})

    def statistics(self, fixture_id):
        return self._get("fixtures/statistics", {"fixture": fixture_id})

    def events(self, fixture_id):
        return self._get("fixtures/events", {"fixture": fixture_id})
