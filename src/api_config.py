import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_KEY_ENV = "API_FOOTBALL_KEY"
API_FOOTBALL_LEAGUE = 1
API_FOOTBALL_SEASON = 2026
CACHE_TTL_HOURS = 12

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LIVE_TEAMS_CSV = DATA_DIR / "live_teams.csv"
LIVE_FIXTURES_CSV = DATA_DIR / "live_fixtures.csv"
LIVE_STANDINGS_CSV = DATA_DIR / "live_standings.csv"
LIVE_MATCH_STATS_CSV = DATA_DIR / "live_match_stats.csv"
LIVE_ODDS_CSV = DATA_DIR / "live_odds.csv"
API_CACHE_META = DATA_DIR / "api_cache_timestamp.txt"


def load_local_env():
    """Load project .env values when python-dotenv is not installed."""
    if load_dotenv:
        load_dotenv(ENV_FILE)
        return

    if not ENV_FILE.exists():
        return

    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_local_env()


def get_api_key():
    """Read the API-Football key from the environment or .env file."""
    api_key = os.getenv(API_FOOTBALL_KEY_ENV, "").strip()
    if not api_key or api_key.lower().startswith("your_"):
        return ""
    return api_key


def has_api_key():
    """Return True when API credentials are available."""
    return bool(get_api_key())
