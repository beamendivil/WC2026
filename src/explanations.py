import pandas as pd


MODEL_COLUMNS = [
    "team",
    "group",
    "fifa_rank",
    "fifa_points",
    "ranking_date",
    "elo_rating",
    "historical_elo",
    "historical_form",
    "historical_attack",
    "historical_defense",
    "historical_matches",
    "last_five_form",
    "derived_form_score",
    "goals_for",
    "goals_against",
    "xg_difference",
    "host_advantage",
    "host_country",
    "venue_country",
    "venue_city",
    "travel_distance_km",
    "rest_days",
    "climate_difference",
    "altitude_difference",
    "opponent_strength",
    "venue_familiarity",
    "venue_impact",
    "rank_score",
    "elo_score",
    "recent_form_component",
    "attack_score",
    "defense_score",
    "xg_component",
    "shot_component",
    "player_component",
    "context_component",
    "market_component",
    "confirmed_group_finish_component",
    "tournament_position_component",
    "strength_score",
]

DIFFICULTY_COLUMNS = [
    "team",
    "group",
    "tournament_difficulty_index",
    "opponent_strength",
    "travel_distance_km",
    "rest_days",
    "climate_difference",
    "altitude_difference",
    "venue_familiarity",
    "host_country",
    "venue_country",
    "venue_city",
]

DIFFICULTY_EXPLANATION = (
    "Higher values mean a tougher projected route. The score uses opponent "
    "strength, travel distance, rest disadvantage, climate difference, altitude "
    "difference, and a venue-familiarity penalty."
)

MATHESON_NOTE = (
    "The Tournament Difficulty Index uses Matheson-style World Cup context only "
    "as inspiration for host advantage, travel burden, venue effects, and "
    "tournament complexity. Economic impact variables such as tourism, "
    "infrastructure spending, or GDP are intentionally not used for match "
    "prediction."
)


def build_data_readiness_table(raw_teams):
    """Show which data tiers are already present in the current dataset."""
    tier_columns = {
        "Tier 1: rankings, historical results, standings, bracket": [
            "fifa_rank",
            "fifa_points",
            "ranking_date",
            "historical_elo",
            "historical_form",
            "historical_attack",
            "historical_defense",
            "historical_matches",
            "goals_for",
            "goals_against",
            "current_points",
            "current_goal_difference",
            "current_goals_for",
        ],
        "Tier 2: team form and match stats": [
            "last_five_form",
            "xg_for",
            "xg_against",
            "shots",
            "shots_on_target",
            "big_chances",
            "corners",
            "possession",
        ],
        "Tier 3: player data": [
            "injury_impact",
            "suspension_impact",
            "expected_lineup_score",
            "goalkeeper_score",
        ],
        "Tier 5: context": [
            "host_country",
            "venue_country",
            "venue_city",
            "rest_days",
            "travel_distance_km",
            "climate_difference",
            "altitude_difference",
            "opponent_strength",
            "venue_familiarity",
            "temperature_f",
            "altitude_m",
            "venue_impact",
        ],
        "Tier 6: market intelligence": ["market_implied_prob"],
    }

    rows = []
    for tier, columns in tier_columns.items():
        present_columns = [column for column in columns if column in raw_teams.columns]
        rows.append(
            {
                "data_tier": tier,
                "columns_available": len(present_columns),
                "columns_possible": len(columns),
                "coverage": f"{len(present_columns)}/{len(columns)}",
            }
        )

    return pd.DataFrame(rows)


def describe_host_advantage(value):
    """Turn the host flag into plain English."""
    return "Host boost" if value > 0 else "No host boost"


def format_score(value):
    """Format model scores consistently for dashboard text."""
    return f"{value:.1f}"


def find_biggest_factor(row, factor_columns, biggest=True):
    """Find the highest or lowest contribution among model components."""
    values = {
        label: pd.to_numeric(pd.Series([row.get(column, 0)]), errors="coerce")
        .fillna(0)
        .iloc[0]
        for column, label in factor_columns.items()
    }
    if biggest:
        return max(values, key=values.get)
    return min(values, key=values.get)


def find_biggest_negative_factor(row):
    """Find the clearest plain-English drag on a team's prediction."""
    weakness_scores = {
        "tough tournament path": row.get("tournament_difficulty_index", 0) / 10,
        "no host advantage": 5 if row.get("host_advantage", 0) <= 0 else 0,
        "recent form": max(0, 10 - row.get("derived_form_score", 5)),
        "attack": max(0, 10 - row.get("attack_score", 5)),
        "defense": max(0, 10 - row.get("defense_score", 5)),
        "travel burden": row.get("travel_distance_km", 0) / 1000,
        "short rest": max(0, 4 - row.get("rest_days", 4)),
    }
    return max(weakness_scores, key=weakness_scores.get)


def build_team_explanations(teams, difficulty_index):
    """Create short, plain-English explanations for each team prediction."""
    factor_columns = {
        "rank_score": "ranking",
        "elo_score": "Elo rating",
        "recent_form_component": "recent form",
        "attack_score": "attack",
        "defense_score": "defense",
        "host_advantage_component": "host advantage",
        "xg_component": "xG",
        "shot_component": "shots",
        "player_component": "player availability",
        "context_component": "travel/rest/venue context",
        "market_component": "market signal",
        "tournament_position_component": "current standings",
    }
    merged = teams.merge(
        difficulty_index[["team", "tournament_difficulty_index"]],
        on="team",
        how="left",
    )

    rows = []
    for _, row in merged.iterrows():
        biggest_positive = find_biggest_factor(row, factor_columns, biggest=True)
        biggest_negative = find_biggest_negative_factor(row)
        explanation = (
            f"{row['team']} has a strength score of "
            f"{format_score(row['strength_score'])} and a tournament difficulty "
            f"of {format_score(row['tournament_difficulty_index'])}. "
            f"{describe_host_advantage(row['host_advantage'])}; form "
            f"{format_score(row['derived_form_score'])}, attack "
            f"{format_score(row['attack_score'])}, defense "
            f"{format_score(row['defense_score'])}. Biggest plus: "
            f"{biggest_positive}. Biggest drag: {biggest_negative}."
        )
        rows.append(
            {
                "team": row["team"],
                "strength_score": row["strength_score"],
                "tournament_difficulty": row["tournament_difficulty_index"],
                "host_advantage": describe_host_advantage(row["host_advantage"]),
                "recent_form": row["derived_form_score"],
                "attack_score": row["attack_score"],
                "defense_score": row["defense_score"],
                "biggest_positive_factor": biggest_positive,
                "biggest_negative_factor": biggest_negative,
                "explanation": explanation,
            }
        )

    return pd.DataFrame(rows).sort_values("strength_score", ascending=False)
