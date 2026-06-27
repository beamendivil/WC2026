import json
from datetime import datetime, timedelta

import pandas as pd
from pandas.errors import EmptyDataError

from src.api_client import APIFootballClient
from src.api_config import API_RETRY_COOLDOWN_MINUTES
from src.api_config import API_CACHE_META, API_FOOTBALL_LEAGUE, API_FOOTBALL_SEASON
from src.api_config import CACHE_TTL_HOURS, LIVE_FIXTURES_CSV, LIVE_MATCH_STATS_CSV
from src.api_config import LIVE_COUNTRIES_CSV, LIVE_ODDS_CSV, LIVE_STANDINGS_CSV
from src.api_config import LIVE_TEAMS_CSV
from src.api_config import has_api_key
from src.data_loader import HOST_TEAMS, load_sample_data


COUNTRY_FALLBACK_TEAM_COUNT = 48
COUNTRY_FALLBACK_GROUPS = list("ABCDEFGHIJKL")
COUNTRY_FALLBACK_REQUIRED_TEAMS = ["Scotland"]
COUNTRY_FALLBACK_REPLACEMENTS = {"Wales": "Scotland"}


def parse_timestamp(value):
    """Parse an ISO timestamp from cache metadata."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def read_cache_state():
    """Read API cache metadata with backward compatibility for old timestamps."""
    if not API_CACHE_META.exists():
        return {}

    text = API_CACHE_META.read_text().strip()
    if not text:
        return {}

    try:
        state = json.loads(text)
    except json.JSONDecodeError:
        return {"last_success": text}

    return state if isinstance(state, dict) else {}


def write_cache_state(**updates):
    """Persist API refresh metadata without exposing credentials."""
    API_CACHE_META.parent.mkdir(parents=True, exist_ok=True)
    state = read_cache_state()
    state.update(updates)
    API_CACHE_META.write_text(json.dumps(state, indent=2, sort_keys=True))


def cache_is_fresh(ttl_hours=CACHE_TTL_HOURS):
    """Return True when local live CSVs were updated recently."""
    timestamp = parse_timestamp(read_cache_state().get("last_success"))
    if not timestamp:
        return False
    return datetime.now() - timestamp < timedelta(hours=ttl_hours)


def refresh_is_on_cooldown(cooldown_minutes=API_RETRY_COOLDOWN_MINUTES):
    """Return True when an API refresh was attempted too recently."""
    timestamp = parse_timestamp(read_cache_state().get("last_attempt"))
    if not timestamp:
        return False
    return datetime.now() - timestamp < timedelta(minutes=cooldown_minutes)


def minutes_until_retry(cooldown_minutes=API_RETRY_COOLDOWN_MINUTES):
    """Return whole minutes until the next non-forced refresh is allowed."""
    timestamp = parse_timestamp(read_cache_state().get("last_attempt"))
    if not timestamp:
        return 0
    retry_at = timestamp + timedelta(minutes=cooldown_minutes)
    seconds = max(0, (retry_at - datetime.now()).total_seconds())
    return int((seconds + 59) // 60)


def live_cache_exists():
    """Return True when the minimum local live dataset is readable."""
    if not LIVE_TEAMS_CSV.exists() or LIVE_TEAMS_CSV.stat().st_size <= 1:
        return False
    try:
        return not pd.read_csv(LIVE_TEAMS_CSV).empty
    except (EmptyDataError, OSError):
        return False


def mark_cache_attempt():
    """Record that a provider refresh was attempted."""
    write_cache_state(last_attempt=datetime.now().isoformat())


def mark_cache_updated():
    """Write a local timestamp after successful API refresh."""
    now = datetime.now().isoformat()
    write_cache_state(last_attempt=now, last_success=now, last_error="")


def mark_cache_error(error):
    """Record a failed provider refresh so reruns do not retry immediately."""
    write_cache_state(last_attempt=datetime.now().isoformat(), last_error=str(error))


def safe_get(client_method, *args, **kwargs):
    """Call an optional endpoint without breaking the whole refresh."""
    try:
        return client_method(*args, **kwargs)
    except Exception:
        return []


def flatten_teams(raw_teams):
    """Normalize API-Football teams into the model's expected columns."""
    rows = []
    for item in raw_teams:
        team = item.get("team", {})
        team_name = team.get("name", "Unknown")
        rows.append(
            {
                "team": team_name,
                "team_id": team.get("id"),
                "group": "TBD",
                # API-Football does not provide FIFA rank or Elo here.
                "fifa_rank": 100,
                "elo_rating": 1500,
                "recent_form_score": 5,
                "goals_for": 10,
                "goals_against": 10,
                # API-Football fixture stats may provide shots/possession, but
                # xG/xGA are placeholders unless a provider supplies them.
                "xg": 0,
                "xga": 0,
                "xg_for": 0,
                "xg_against": 0,
                "host_advantage": 1
                if str(team_name).strip().lower() in HOST_TEAMS
                else 0,
                "travel_distance_km": 0,
                "rest_days": 4,
                "venue_city": "Unknown",
                "venue_country": "Unknown",
                "host_country": "Unknown",
                "match_status": "TBD",
                "opponent": "TBD",
            }
        )
    return pd.DataFrame(rows)


def flatten_countries(raw_countries):
    """Normalize API-Football countries into a local directory CSV."""
    rows = []
    for item in raw_countries:
        country_name = item.get("name")
        if not country_name:
            continue
        rows.append(
            {
                "name": country_name,
                "code": item.get("code"),
                "flag": item.get("flag"),
            }
        )
    return pd.DataFrame(rows)


def build_country_fallback_teams(countries_df):
    """Create a tournament-shaped candidate list from the countries endpoint."""
    if countries_df.empty or "name" not in countries_df.columns:
        return pd.DataFrame()

    available_countries = {
        str(country).strip().lower(): str(country).strip()
        for country in countries_df["name"].dropna()
    }
    sample = load_sample_data().head(COUNTRY_FALLBACK_TEAM_COUNT).copy()
    sample["team"] = sample["team"].replace(COUNTRY_FALLBACK_REPLACEMENTS)

    for team_name in COUNTRY_FALLBACK_REQUIRED_TEAMS:
        if team_name.lower() not in available_countries:
            continue
        if team_name not in sample["team"].values:
            sample.iloc[-1, sample.columns.get_loc("team")] = team_name

    sample["data_source_note"] = "API-Football /countries fallback"
    return sample


def flatten_fixtures(raw_fixtures):
    """Normalize fixture responses into a compact local CSV."""
    rows = []
    for item in raw_fixtures:
        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        venue = fixture.get("venue", {}) or {}
        status = fixture.get("status", {}) or {}
        league = item.get("league", {}) or {}
        goals = item.get("goals", {}) or {}
        score = item.get("score", {}) or {}
        penalty = score.get("penalty", {}) or {}
        rows.append(
            {
                "fixture_id": fixture.get("id"),
                "date": fixture.get("date"),
                "round": league.get("round"),
                "group": league.get("round", "TBD"),
                "team_home": teams.get("home", {}).get("name"),
                "team_away": teams.get("away", {}).get("name"),
                "team_home_id": teams.get("home", {}).get("id"),
                "team_away_id": teams.get("away", {}).get("id"),
                "goals_home": goals.get("home"),
                "goals_away": goals.get("away"),
                "penalty_home": penalty.get("home"),
                "penalty_away": penalty.get("away"),
                "venue_city": venue.get("city", "Unknown"),
                "venue_country": "Unknown",
                "match_status": status.get("short", "TBD"),
            }
        )
    return pd.DataFrame(rows)


def flatten_standings(raw_standings):
    """Normalize standings where available."""
    rows = []
    for item in raw_standings:
        league = item.get("league", {})
        for group in league.get("standings", []) or []:
            for standing in group:
                team = standing.get("team", {})
                all_stats = standing.get("all", {})
                goals = all_stats.get("goals", {})
                rows.append(
                    {
                        "team": team.get("name"),
                        "team_id": team.get("id"),
                        "group": standing.get("group", "TBD"),
                        "rank": standing.get("rank"),
                        "points": standing.get("points", 0),
                        "goals_for": goals.get("for", 0),
                        "goals_against": goals.get("against", 0),
                        "goal_difference": standing.get("goalsDiff", 0),
                        "form": standing.get("form", ""),
                    }
                )
    return pd.DataFrame(rows)


def flatten_odds(raw_odds):
    """Save raw-ish odds summary when the endpoint is available."""
    rows = []
    for item in raw_odds:
        fixture = item.get("fixture", {})
        rows.append(
            {
                "fixture_id": fixture.get("id"),
                "bookmakers_count": len(item.get("bookmakers", []) or []),
            }
        )
    return pd.DataFrame(rows)


def flatten_match_stats(raw_stats):
    """Normalize fixture statistics/events into a compact local CSV.

    API-Football statistics are real event/stat fields when available. xG is
    still a placeholder unless a provider supplies expected-goals data.
    """
    rows = []
    for item in raw_stats:
        team = item.get("team", {})
        stats = {stat.get("type"): stat.get("value") for stat in item.get("statistics", [])}
        rows.append(
            {
                "team": team.get("name"),
                "team_id": team.get("id"),
                "shots": stats.get("Total Shots", 0) or 0,
                "shots_on_target": stats.get("Shots on Goal", 0) or 0,
                "possession": stats.get("Ball Possession", "0%"),
                "corners": stats.get("Corner Kicks", 0) or 0,
                "xg": 0,
                "xga": 0,
            }
        )
    return pd.DataFrame(rows)


def merge_live_model_data(teams_df, standings_df, fixtures_df):
    """Combine live teams with standings/fixture hints for model input."""
    if teams_df.empty:
        return teams_df

    live = teams_df.copy()
    if not standings_df.empty:
        live = live.drop(columns=["goals_for", "goals_against"], errors="ignore")
        live = live.merge(
            standings_df[
                ["team_id", "group", "goals_for", "goals_against", "form"]
            ],
            on="team_id",
            how="left",
            suffixes=("", "_standing"),
        )
        live["group"] = live["group_standing"].fillna(live["group"])
        live["recent_form_score"] = live["form"].apply(form_to_score)
        live = live.drop(columns=["group_standing", "form"], errors="ignore")

    if not fixtures_df.empty:
        fixture_lookup = build_fixture_lookup(fixtures_df)
        live["opponent"] = live["team"].map(
            lambda team: fixture_lookup.get(team, {}).get("opponent", "TBD")
        )
        live["venue_city"] = live["team"].map(
            lambda team: fixture_lookup.get(team, {}).get("venue_city", "Unknown")
        )
        live["venue_country"] = live["team"].map(
            lambda team: fixture_lookup.get(team, {}).get("venue_country", "Unknown")
        )
        live["match_status"] = live["team"].map(
            lambda team: fixture_lookup.get(team, {}).get("match_status", "TBD")
        )

    return live


def form_to_score(form):
    """Convert API-Football form strings like WWDLD into a 0-10 score."""
    if not isinstance(form, str) or not form:
        return 5
    points = {"W": 3, "D": 1, "L": 0}
    values = [points[result] for result in form[-5:] if result in points]
    if not values:
        return 5
    return sum(values) / (len(values) * 3) * 10


def build_fixture_lookup(fixtures_df):
    """Map each team to its first known opponent and venue."""
    lookup = {}
    for _, row in fixtures_df.iterrows():
        home = row.get("team_home")
        away = row.get("team_away")
        if home and home not in lookup:
            lookup[home] = {
                "opponent": away or "TBD",
                "venue_city": row.get("venue_city", "Unknown"),
                "venue_country": row.get("venue_country", "Unknown"),
                "match_status": row.get("match_status", "TBD"),
            }
        if away and away not in lookup:
            lookup[away] = {
                "opponent": home or "TBD",
                "venue_city": row.get("venue_city", "Unknown"),
                "venue_country": row.get("venue_country", "Unknown"),
                "match_status": row.get("match_status", "TBD"),
            }
    return lookup


def save_live_data(force=False):
    """Fetch API-Football data and save normalized local CSV snapshots."""
    has_cached_data = live_cache_exists()

    if has_cached_data and cache_is_fresh() and not force:
        return True, "Using cached live API CSV files."

    if not has_api_key():
        if has_cached_data:
            return True, "Using cached live API CSV files; API_FOOTBALL_KEY is missing."
        return False, "Missing API_FOOTBALL_KEY. Using sample CSV data."

    if refresh_is_on_cooldown() and not force:
        minutes = minutes_until_retry()
        retry_message = (
            "Next API refresh is available "
            f"in about {minutes} minute{'s' if minutes != 1 else ''}."
        )
        if has_cached_data:
            return True, f"Using cached live API CSV files; {retry_message}"
        return False, f"API refresh was attempted recently. {retry_message}"

    mark_cache_attempt()

    try:
        client = APIFootballClient()
        raw_teams = client.teams(API_FOOTBALL_LEAGUE, API_FOOTBALL_SEASON)
        raw_fixtures = client.fixtures(API_FOOTBALL_LEAGUE, API_FOOTBALL_SEASON)
        raw_standings = safe_get(
            client.standings, API_FOOTBALL_LEAGUE, API_FOOTBALL_SEASON
        )
        raw_odds = safe_get(client.odds, API_FOOTBALL_LEAGUE, API_FOOTBALL_SEASON)
        raw_countries = safe_get(client.countries)

        fixtures_df = flatten_fixtures(raw_fixtures)
        standings_df = flatten_standings(raw_standings)
        countries_df = flatten_countries(raw_countries)
        teams_df = merge_live_model_data(
            flatten_teams(raw_teams), standings_df, fixtures_df
        )
        used_country_fallback = False
        if teams_df.empty:
            teams_df = build_country_fallback_teams(countries_df)
            used_country_fallback = True
            if teams_df.empty:
                raise ValueError(
                    "API-Football returned no teams for "
                    f"league={API_FOOTBALL_LEAGUE}, season={API_FOOTBALL_SEASON}, "
                    "and the /countries fallback was unavailable."
                )
        odds_df = flatten_odds(raw_odds)

        stats_frames = []
        fixture_ids = fixtures_df.get(
            "fixture_id", pd.Series(dtype=object)
        ).dropna().head(25)
        for fixture_id in fixture_ids:
            stats = safe_get(client.statistics, fixture_id)
            if stats:
                frame = flatten_match_stats(stats)
                frame["fixture_id"] = fixture_id
                stats_frames.append(frame)
        stats_df = (
            pd.concat(stats_frames, ignore_index=True)
            if stats_frames
            else pd.DataFrame()
        )

        LIVE_TEAMS_CSV.parent.mkdir(parents=True, exist_ok=True)
        teams_df.to_csv(LIVE_TEAMS_CSV, index=False)
        fixtures_df.to_csv(LIVE_FIXTURES_CSV, index=False)
        standings_df.to_csv(LIVE_STANDINGS_CSV, index=False)
        stats_df.to_csv(LIVE_MATCH_STATS_CSV, index=False)
        odds_df.to_csv(LIVE_ODDS_CSV, index=False)
        countries_df.to_csv(LIVE_COUNTRIES_CSV, index=False)
        mark_cache_updated()
    except Exception as error:
        mark_cache_error(error)
        if has_cached_data:
            return True, "API refresh failed; using cached live API CSV files."
        return False, "API refresh failed. Using sample CSV data."

    if used_country_fallback:
        return True, "Updated live API CSV files from API-Football /countries fallback."
    return True, "Updated live API CSV files."


def load_live_model_data(force=False):
    """Return normalized live teams, falling back gracefully when unavailable."""
    ok, message = save_live_data(force=force)
    if not ok or not LIVE_TEAMS_CSV.exists():
        return pd.DataFrame(), message

    try:
        live = pd.read_csv(LIVE_TEAMS_CSV)
    except EmptyDataError:
        return pd.DataFrame(), "Cached live API CSV is empty. Using sample CSV data."
    return live, message


def load_cached_fixtures():
    """Load cached tournament fixtures without triggering another API request."""
    if not LIVE_FIXTURES_CSV.exists() or LIVE_FIXTURES_CSV.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(LIVE_FIXTURES_CSV)
    except (EmptyDataError, OSError):
        return pd.DataFrame()


if __name__ == "__main__":
    success, status = save_live_data(force=True)
    print(status)
    raise SystemExit(0 if success else 1)
