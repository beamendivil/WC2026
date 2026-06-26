from datetime import datetime, timedelta

import pandas as pd

from src.api_client import APIFootballClient
from src.api_config import API_CACHE_META, API_FOOTBALL_LEAGUE, API_FOOTBALL_SEASON
from src.api_config import CACHE_TTL_HOURS, LIVE_FIXTURES_CSV, LIVE_MATCH_STATS_CSV
from src.api_config import LIVE_ODDS_CSV, LIVE_STANDINGS_CSV, LIVE_TEAMS_CSV
from src.api_config import has_api_key
from src.data_loader import HOST_TEAMS


def cache_is_fresh(ttl_hours=CACHE_TTL_HOURS):
    """Return True when local live CSVs were updated recently."""
    if not API_CACHE_META.exists():
        return False
    timestamp = datetime.fromisoformat(API_CACHE_META.read_text().strip())
    return datetime.now() - timestamp < timedelta(hours=ttl_hours)


def mark_cache_updated():
    """Write a local timestamp after successful API refresh."""
    API_CACHE_META.parent.mkdir(parents=True, exist_ok=True)
    API_CACHE_META.write_text(datetime.now().isoformat())


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


def flatten_fixtures(raw_fixtures):
    """Normalize fixture responses into a compact local CSV."""
    rows = []
    for item in raw_fixtures:
        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        venue = fixture.get("venue", {}) or {}
        status = fixture.get("status", {}) or {}
        league = item.get("league", {}) or {}
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
    if not has_api_key():
        return False, "Missing API_FOOTBALL_KEY. Using sample CSV data."

    if cache_is_fresh() and not force and LIVE_TEAMS_CSV.exists():
        return True, "Using cached live API CSV files."

    client = APIFootballClient()
    raw_teams = client.teams(API_FOOTBALL_LEAGUE, API_FOOTBALL_SEASON)
    raw_fixtures = client.fixtures(API_FOOTBALL_LEAGUE, API_FOOTBALL_SEASON)
    raw_standings = safe_get(
        client.standings, API_FOOTBALL_LEAGUE, API_FOOTBALL_SEASON
    )
    raw_odds = safe_get(client.odds, API_FOOTBALL_LEAGUE, API_FOOTBALL_SEASON)

    fixtures_df = flatten_fixtures(raw_fixtures)
    standings_df = flatten_standings(raw_standings)
    teams_df = merge_live_model_data(
        flatten_teams(raw_teams), standings_df, fixtures_df
    )
    odds_df = flatten_odds(raw_odds)

    stats_frames = []
    for fixture_id in fixtures_df.get("fixture_id", pd.Series(dtype=object)).dropna().head(25):
        stats = safe_get(client.statistics, fixture_id)
        if stats:
            frame = flatten_match_stats(stats)
            frame["fixture_id"] = fixture_id
            stats_frames.append(frame)
    stats_df = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()

    LIVE_TEAMS_CSV.parent.mkdir(parents=True, exist_ok=True)
    teams_df.to_csv(LIVE_TEAMS_CSV, index=False)
    fixtures_df.to_csv(LIVE_FIXTURES_CSV, index=False)
    standings_df.to_csv(LIVE_STANDINGS_CSV, index=False)
    stats_df.to_csv(LIVE_MATCH_STATS_CSV, index=False)
    odds_df.to_csv(LIVE_ODDS_CSV, index=False)
    mark_cache_updated()

    return True, "Updated live API CSV files."


def load_live_model_data(force=False):
    """Return normalized live teams, falling back gracefully when unavailable."""
    ok, message = save_live_data(force=force)
    if not ok or not LIVE_TEAMS_CSV.exists():
        return pd.DataFrame(), message

    live = pd.read_csv(LIVE_TEAMS_CSV)
    return live, message


if __name__ == "__main__":
    success, status = save_live_data(force=True)
    print(status)
    raise SystemExit(0 if success else 1)
