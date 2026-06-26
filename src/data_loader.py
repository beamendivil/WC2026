from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_TEAMS_PATH = DATA_DIR / "sample_teams.csv"
SAMPLE_MATCHES_PATH = DATA_DIR / "sample_matches.csv"

REQUIRED_COLUMNS = [
    "team",
    "group",
    "fifa_rank",
    "recent_form_score",
    "goals_for",
    "goals_against",
    "host_advantage",
]

MINIMUM_COLUMNS = ["team", "group"]

REQUIRED_DEFAULTS = {
    "fifa_rank": 100,
    "recent_form_score": 5,
    "goals_for": 10,
    "goals_against": 10,
    "host_advantage": 0,
}

HOST_TEAMS = {"united states", "usa", "usmnt", "mexico", "canada"}

OPTIONAL_COLUMNS = [
    "team_id",
    "host_country",
    "venue_country",
    "venue_city",
    "elo_rating",
    "last_five_form",
    "xg_for",
    "xg_against",
    "xg",
    "xga",
    "shots",
    "shots_on_target",
    "big_chances",
    "corners",
    "possession",
    "injury_impact",
    "suspension_impact",
    "expected_lineup_score",
    "goalkeeper_score",
    "rest_days",
    "travel_distance_km",
    "climate_difference",
    "altitude_difference",
    "opponent_strength",
    "venue_familiarity",
    "temperature_f",
    "altitude_m",
    "venue_impact",
    "market_implied_prob",
    "current_points",
    "current_goal_difference",
    "current_goals_for",
    "match_status",
    "opponent",
]

NUMERIC_OPTIONAL_DEFAULTS = {
    "elo_rating": 1500,
    "xg_for": 0,
    "xg_against": 0,
    "xg": 0,
    "xga": 0,
    "shots": 0,
    "shots_on_target": 0,
    "big_chances": 0,
    "corners": 0,
    "possession": 50,
    "injury_impact": 0,
    "suspension_impact": 0,
    "expected_lineup_score": 0,
    "goalkeeper_score": 0,
    "rest_days": 4,
    "travel_distance_km": 0,
    "climate_difference": 0,
    "altitude_difference": 0,
    "opponent_strength": 5,
    "venue_familiarity": 0.5,
    "temperature_f": 75,
    "altitude_m": 0,
    "venue_impact": 0,
    "market_implied_prob": 0,
    "current_points": 0,
    "current_goal_difference": 0,
    "current_goals_for": 0,
}


def create_sample_dataframe():
    """Fallback data used when data/sample_teams.csv is not available."""
    sample_rows = [
        ["United States", "A", 16, 7.4, 18, 10, 1],
        ["Canada", "A", 31, 6.9, 14, 11, 1],
        ["Mexico", "A", 14, 7.2, 17, 9, 1],
        ["New Zealand", "A", 103, 5.1, 9, 16, 0],
        ["Argentina", "B", 1, 9.4, 31, 8, 0],
        ["Morocco", "B", 12, 8.0, 20, 8, 0],
        ["Japan", "B", 18, 7.8, 22, 12, 0],
        ["South Africa", "B", 59, 6.0, 11, 13, 0],
        ["France", "C", 2, 9.0, 29, 10, 0],
        ["Senegal", "C", 17, 7.6, 19, 10, 0],
        ["Australia", "C", 24, 6.8, 15, 13, 0],
        ["Jamaica", "C", 53, 5.9, 12, 16, 0],
        ["Brazil", "D", 5, 8.6, 27, 12, 0],
        ["Switzerland", "D", 19, 7.2, 16, 11, 0],
        ["Egypt", "D", 36, 6.7, 14, 12, 0],
        ["Panama", "D", 45, 6.1, 13, 15, 0],
        ["England", "E", 4, 8.7, 25, 9, 0],
        ["Colombia", "E", 10, 8.1, 21, 11, 0],
        ["South Korea", "E", 23, 7.0, 18, 14, 0],
        ["Qatar", "E", 58, 5.8, 11, 17, 0],
        ["Spain", "F", 3, 8.8, 28, 10, 0],
        ["Uruguay", "F", 11, 8.2, 21, 10, 0],
        ["Serbia", "F", 32, 6.8, 16, 15, 0],
        ["Ghana", "F", 60, 5.9, 13, 18, 0],
        ["Portugal", "G", 6, 8.5, 26, 11, 0],
        ["Croatia", "G", 13, 7.7, 18, 12, 0],
        ["Tunisia", "G", 41, 6.3, 12, 14, 0],
        ["Honduras", "G", 78, 5.2, 10, 19, 0],
        ["Netherlands", "H", 7, 8.4, 24, 10, 0],
        ["Denmark", "H", 21, 7.0, 17, 12, 0],
        ["Nigeria", "H", 38, 6.6, 15, 15, 0],
        ["Saudi Arabia", "H", 56, 5.7, 12, 18, 0],
        ["Belgium", "I", 8, 8.0, 22, 12, 0],
        ["Ecuador", "I", 30, 6.9, 16, 13, 0],
        ["Wales", "I", 34, 6.4, 13, 14, 0],
        ["Costa Rica", "I", 52, 5.6, 11, 17, 0],
        ["Germany", "J", 9, 8.1, 23, 13, 0],
        ["Chile", "J", 44, 6.2, 14, 16, 0],
        ["Iran", "J", 20, 6.9, 17, 15, 0],
        ["Cameroon", "J", 49, 6.0, 13, 16, 0],
        ["Italy", "K", 15, 7.5, 19, 12, 0],
        ["Poland", "K", 28, 6.8, 16, 14, 0],
        ["Algeria", "K", 43, 6.3, 14, 15, 0],
        ["Iraq", "K", 55, 5.8, 12, 18, 0],
        ["Austria", "L", 25, 7.0, 17, 13, 0],
        ["Norway", "L", 33, 6.7, 18, 16, 0],
        ["Peru", "L", 46, 6.0, 12, 15, 0],
        ["Uzbekistan", "L", 57, 5.7, 11, 17, 0],
    ]
    return pd.DataFrame(sample_rows, columns=REQUIRED_COLUMNS)


def load_sample_data():
    """Load the included sample CSV, falling back to embedded demo data."""
    try:
        return pd.read_csv(SAMPLE_TEAMS_PATH)
    except FileNotFoundError:
        return create_sample_dataframe()


def load_sample_matches():
    """Load optional sample match placeholders for future fixture work."""
    try:
        return pd.read_csv(SAMPLE_MATCHES_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


def validate_team_data(df):
    """Return a list of validation errors for the team dataframe."""
    missing_columns = [col for col in MINIMUM_COLUMNS if col not in df.columns]
    errors = []

    if missing_columns:
        errors.append(f"Missing columns: {', '.join(missing_columns)}")
        return errors

    if len(df) != 48:
        errors.append("The tournament simulator expects exactly 48 teams.")

    group_sizes = df.groupby("group")["team"].count()
    if len(group_sizes) != 12 or not (group_sizes == 4).all():
        errors.append("Expected 12 groups with exactly 4 teams in each group.")

    if df["team"].duplicated().any():
        errors.append("Team names must be unique.")

    return errors


def add_safe_defaults(df):
    """Add safe defaults so a partial CSV can still run."""
    teams = df.copy()

    for column, default_value in REQUIRED_DEFAULTS.items():
        if column not in teams.columns:
            teams[column] = default_value

    return teams


def prepare_optional_columns(teams):
    """Add optional model columns with neutral defaults when they are missing."""
    for column, default_value in NUMERIC_OPTIONAL_DEFAULTS.items():
        if column not in teams.columns:
            teams[column] = default_value
        teams[column] = pd.to_numeric(teams[column], errors="coerce").fillna(
            default_value
        )

    if "last_five_form" not in teams.columns:
        teams["last_five_form"] = ""
    for column in [
        "host_country",
        "venue_country",
        "venue_city",
        "match_status",
        "opponent",
    ]:
        if column not in teams.columns:
            teams[column] = "Unknown"
        teams[column] = teams[column].fillna("Unknown")

    return teams
