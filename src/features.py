import numpy as np
import pandas as pd

from src.bracket import CONFIRMED_GROUP_POSITIONS
from src.data_loader import HOST_TEAMS, REQUIRED_COLUMNS, REQUIRED_DEFAULTS
from src.data_loader import prepare_optional_columns
from src.historical_model import enrich_with_historical_features


def infer_host_advantage(team_name):
    """Return 1 for the 2026 host nations and 0 for all other teams."""
    return 1 if str(team_name).strip().lower() in HOST_TEAMS else 0


def parse_last_five_form(form_text):
    """Convert a form string like W-W-D-L-W into a 0 to 10 score."""
    if pd.isna(form_text) or str(form_text).strip() == "":
        return np.nan

    results = [
        result.strip().upper()
        for result in str(form_text).replace(",", "-").replace(" ", "-").split("-")
        if result.strip()
    ]
    points = {"W": 3, "D": 1, "L": 0}
    valid_results = [points[result] for result in results if result in points]

    if not valid_results:
        return np.nan

    return sum(valid_results[-5:]) / (len(valid_results[-5:]) * 3) * 10


def normalize_series(series, higher_is_better=True):
    """Scale a numeric series to 0 through 10."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    minimum = numeric.min()
    maximum = numeric.max()

    if maximum == minimum:
        return pd.Series(5, index=series.index)

    scaled = (numeric - minimum) / (maximum - minimum) * 10
    if not higher_is_better:
        scaled = 10 - scaled
    return scaled


def safe_ratio_score(series):
    """Scale positive counting stats without dividing by zero."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    maximum = numeric.max()
    if maximum <= 0:
        return pd.Series(5, index=series.index)
    return numeric / maximum * 10


def normalize_value(value, maximum, default=0):
    """Normalize a single value to 0 through 1 for row-level formulas."""
    value = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default).iloc[0]
    if maximum <= 0:
        return 0
    return min(max(value / maximum, 0), 1)


def calculate_tournament_difficulty(row):
    """Calculate a logistics-based Tournament Difficulty Index score.

    This feature is inspired by Victor Matheson's discussion of World Cup
    hosting context and tournament logistics. It is not an economic claim and
    does not use economic impact variables such as tourism, GDP, public
    infrastructure spending, or local business activity.

    Higher scores mean a harder path.
    """
    opponent_strength = normalize_value(row.get("opponent_strength", 5), 10, 5)
    travel_distance = normalize_value(row.get("travel_distance_km", 0), 12000, 0)
    rest_days = pd.to_numeric(
        pd.Series([row.get("rest_days", 4)]), errors="coerce"
    ).fillna(4).iloc[0]
    rest_disadvantage = normalize_value(max(0, 5 - rest_days), 5, 0)
    climate_difference = normalize_value(row.get("climate_difference", 0), 40, 0)
    altitude_difference = normalize_value(row.get("altitude_difference", 0), 3000, 0)
    venue_familiarity = normalize_value(row.get("venue_familiarity", 0.5), 1, 0.5)
    venue_familiarity_penalty = 1 - venue_familiarity

    return (
        0.30 * opponent_strength
        + 0.20 * travel_distance
        + 0.15 * rest_disadvantage
        + 0.15 * climate_difference
        + 0.10 * altitude_difference
        + 0.10 * venue_familiarity_penalty
    ) * 100


def add_strength_scores(
    df,
    ranking_weight,
    form_weight,
    host_weight,
    elo_weight,
    xg_weight,
    player_weight,
    context_weight,
    market_weight,
):
    """Calculate the model inputs and total strength score for each team."""
    teams = df.copy()
    teams = prepare_optional_columns(teams)
    teams = enrich_with_historical_features(teams)
    teams["fifa_rank"] = pd.to_numeric(teams["fifa_rank"], errors="coerce")
    teams["recent_form_score"] = pd.to_numeric(
        teams["recent_form_score"], errors="coerce"
    )
    teams["goals_for"] = pd.to_numeric(teams["goals_for"], errors="coerce")
    teams["goals_against"] = pd.to_numeric(teams["goals_against"], errors="coerce")
    teams["host_advantage"] = pd.to_numeric(teams["host_advantage"], errors="coerce")

    for column, default_value in REQUIRED_DEFAULTS.items():
        teams[column] = teams[column].fillna(default_value)
    teams = teams.dropna(subset=REQUIRED_COLUMNS)

    inferred_host = teams["team"].apply(infer_host_advantage)
    teams["host_advantage"] = np.maximum(teams["host_advantage"], inferred_host)

    max_rank = max(teams["fifa_rank"].max(), 1)

    # Lower FIFA rankings are better, so rank 1 gets the highest score.
    teams["rank_score"] = ((max_rank - teams["fifa_rank"] + 1) / max_rank) * 10
    teams["rank_score"] = teams["rank_score"] * ranking_weight
    teams["elo_rating"] = teams["historical_elo"]
    teams["elo_score"] = normalize_series(teams["historical_elo"]) * elo_weight
    teams["attack_score"] = normalize_series(teams["historical_attack"])

    # Fewer goals conceded should produce a higher defense score.
    teams["defense_score"] = normalize_series(
        teams["historical_defense"], higher_is_better=False
    )
    teams["host_advantage_component"] = teams["host_advantage"] * host_weight

    parsed_form_score = teams["last_five_form"].apply(parse_last_five_form)
    teams["derived_form_score"] = parsed_form_score.fillna(
        teams["historical_form"]
    )
    teams["recent_form_component"] = teams["derived_form_score"] * form_weight

    teams["xg_difference"] = teams["xg_for"] - teams["xg_against"]
    teams["xg_component"] = normalize_series(teams["xg_difference"]) * xg_weight
    teams["shot_component"] = (
        normalize_series(teams["shots_on_target"]) * 0.45
        + normalize_series(teams["big_chances"]) * 0.35
        + normalize_series(teams["possession"]) * 0.20
    ) * (xg_weight * 0.5)

    teams["player_component"] = (
        teams["expected_lineup_score"]
        + teams["goalkeeper_score"]
        - teams["injury_impact"]
        - teams["suspension_impact"]
    ) * player_weight

    # Matheson-inspired context only: host, travel, rest, venue, climate.
    teams["context_component"] = (
        normalize_series(teams["rest_days"]) * 0.45
        + normalize_series(teams["travel_distance_km"], higher_is_better=False) * 0.25
        + normalize_series(abs(teams["temperature_f"] - 70), higher_is_better=False)
        * 0.15
        + normalize_series(teams["altitude_m"], higher_is_better=False) * 0.15
        + teams["venue_impact"]
    ) * context_weight

    teams["market_component"] = (
        normalize_series(teams["market_implied_prob"]) * market_weight
    )
    matches_played = teams["current_matches_played"].replace(0, np.nan)
    points_per_match = (teams["current_points"] / matches_played).fillna(0)
    goal_difference_per_match = (
        teams["current_goal_difference"] / matches_played
    ).fillna(0)
    goals_per_match = (teams["current_goals_for"] / matches_played).fillna(0)
    teams["tournament_position_component"] = (
        points_per_match * 2.0
        + goal_difference_per_match * 0.8
        + goals_per_match * 0.4
    ).clip(lower=-2, upper=10)
    confirmed_positions = {
        team: position
        for positions in CONFIRMED_GROUP_POSITIONS.values()
        for position, team in positions.items()
    }
    confirmed_position_bonus = {1: 1.5, 2: 0.75, 3: 0.25}
    teams["confirmed_group_finish_component"] = teams["team"].map(
        lambda team: confirmed_position_bonus.get(
            confirmed_positions.get(team), 0
        )
    )

    teams["strength_score"] = (
        teams["rank_score"]
        + teams["elo_score"]
        + teams["recent_form_component"]
        + teams["attack_score"]
        + teams["defense_score"]
        + teams["host_advantage_component"]
        + teams["xg_component"]
        + teams["shot_component"]
        + teams["player_component"]
        + teams["context_component"]
        + teams["market_component"]
        + teams["tournament_position_component"]
        + teams["confirmed_group_finish_component"]
    )
    return teams


def calculate_tournament_difficulty_index(teams):
    """Estimate each team's route difficulty before the tournament starts."""
    difficulty_rows = []
    top_32_strength = teams.sort_values("strength_score", ascending=False).head(32)
    max_strength = max(teams["strength_score"].max(), 1)

    for _, team in teams.iterrows():
        group_opponents = teams[
            (teams["group"] == team["group"]) & (teams["team"] != team["team"])
        ]
        likely_knockout_opponents = top_32_strength[
            top_32_strength["team"] != team["team"]
        ]

        group_difficulty = group_opponents["strength_score"].mean()
        knockout_difficulty = likely_knockout_opponents["strength_score"].mean()
        row = team.copy()
        row["opponent_strength"] = group_difficulty / max_strength * 10
        raw_index = calculate_tournament_difficulty(row)
        difficulty_rows.append(
            {
                "team": team["team"],
                "group": team["group"],
                "host_country": team["host_country"],
                "venue_country": team["venue_country"],
                "venue_city": team["venue_city"],
                "opponent_strength": row["opponent_strength"],
                "travel_distance_km": team["travel_distance_km"],
                "rest_days": team["rest_days"],
                "climate_difference": team["climate_difference"],
                "altitude_difference": team["altitude_difference"],
                "venue_familiarity": team["venue_familiarity"],
                "group_difficulty": group_difficulty,
                "projected_knockout_difficulty": knockout_difficulty,
                "logistics_difficulty_score": raw_index,
                "difficulty_raw": raw_index,
            }
        )

    difficulty = pd.DataFrame(difficulty_rows)
    difficulty["tournament_difficulty_index"] = difficulty["difficulty_raw"]
    return difficulty.sort_values("tournament_difficulty_index", ascending=False)
