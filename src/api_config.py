import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv:
    load_dotenv()

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_KEY_ENV = "API_FOOTBALL_KEY"
API_FOOTBALL_LEAGUE = 1
API_FOOTBALL_SEASON = 2026
CACHE_TTL_HOURS = 12

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LIVE_TEAMS_CSV = DATA_DIR / "live_teams.csv"
LIVE_FIXTURES_CSV = DATA_DIR / "live_fixtures.csv"
LIVE_STANDINGS_CSV = DATA_DIR / "live_standings.csv"
LIVE_MATCH_STATS_CSV = DATA_DIR / "live_match_stats.csv"
LIVE_ODDS_CSV = DATA_DIR / "live_odds.csv"
API_CACHE_META = DATA_DIR / "api_cache_timestamp.txt"


def get_api_key():
    """Read the API-Football key from the environment or .env file."""
    return os.getenv(API_FOOTBALL_KEY_ENV, "").strip()


def has_api_key():
    """Return True when API credentials are available."""
    return bool(get_api_key())
