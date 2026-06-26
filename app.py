from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


REQUIRED_COLUMNS = [
    "team",
    "group",
    "fifa_rank",
    "recent_form_score",
    "goals_for",
    "goals_against",
    "host_advantage",
]

GROUP_ORDER = list("ABCDEFGHIJKL")


@st.cache_data
def load_sample_teams() -> pd.DataFrame:
    """Load the built-in sample data so the app works without an upload."""
    rows = [
        {"team": "Argentina", "group": "A", "fifa_rank": 1, "recent_form_score": 9.6, "goals_for": 2.4, "goals_against": 0.7, "host_advantage": 0},
        {"team": "USA", "group": "A", "fifa_rank": 16, "recent_form_score": 7.4, "goals_for": 1.9, "goals_against": 1.1, "host_advantage": 1},
        {"team": "Senegal", "group": "A", "fifa_rank": 18, "recent_form_score": 7.6, "goals_for": 1.8, "goals_against": 1.0, "host_advantage": 0},
        {"team": "New Zealand", "group": "A", "fifa_rank": 95, "recent_form_score": 5.5, "goals_for": 1.1, "goals_against": 1.7, "host_advantage": 0},
        {"team": "France", "group": "B", "fifa_rank": 2, "recent_form_score": 9.4, "goals_for": 2.5, "goals_against": 0.8, "host_advantage": 0},
        {"team": "Mexico", "group": "B", "fifa_rank": 15, "recent_form_score": 7.2, "goals_for": 1.8, "goals_against": 1.2, "host_advantage": 1},
        {"team": "Japan", "group": "B", "fifa_rank": 17, "recent_form_score": 7.9, "goals_for": 1.9, "goals_against": 0.9, "host_advantage": 0},
        {"team": "Ghana", "group": "B", "fifa_rank": 61, "recent_form_score": 6.4, "goals_for": 1.4, "goals_against": 1.5, "host_advantage": 0},
        {"team": "Brazil", "group": "C", "fifa_rank": 3, "recent_form_score": 9.3, "goals_for": 2.6, "goals_against": 0.9, "host_advantage": 0},
        {"team": "Switzerland", "group": "C", "fifa_rank": 19, "recent_form_score": 7.5, "goals_for": 1.7, "goals_against": 1.0, "host_advantage": 0},
        {"team": "Morocco", "group": "C", "fifa_rank": 12, "recent_form_score": 8.2, "goals_for": 1.8, "goals_against": 0.8, "host_advantage": 0},
        {"team": "Jamaica", "group": "C", "fifa_rank": 67, "recent_form_score": 6.1, "goals_for": 1.3, "goals_against": 1.6, "host_advantage": 0},
        {"team": "England", "group": "D", "fifa_rank": 4, "recent_form_score": 9.0, "goals_for": 2.3, "goals_against": 0.8, "host_advantage": 0},
        {"team": "Denmark", "group": "D", "fifa_rank": 21, "recent_form_score": 7.3, "goals_for": 1.7, "goals_against": 1.0, "host_advantage": 0},
        {"team": "Ecuador", "group": "D", "fifa_rank": 24, "recent_form_score": 7.4, "goals_for": 1.6, "goals_against": 1.0, "host_advantage": 0},
        {"team": "China", "group": "D", "fifa_rank": 79, "recent_form_score": 5.8, "goals_for": 1.2, "goals_against": 1.7, "host_advantage": 0},
        {"team": "Spain", "group": "E", "fifa_rank": 5, "recent_form_score": 9.1, "goals_for": 2.2, "goals_against": 0.7, "host_advantage": 0},
        {"team": "Croatia", "group": "E", "fifa_rank": 11, "recent_form_score": 8.0, "goals_for": 1.8, "goals_against": 0.9, "host_advantage": 0},
        {"team": "Egypt", "group": "E", "fifa_rank": 34, "recent_form_score": 7.0, "goals_for": 1.5, "goals_against": 1.1, "host_advantage": 0},
        {"team": "Costa Rica", "group": "E", "fifa_rank": 54, "recent_form_score": 6.2, "goals_for": 1.2, "goals_against": 1.5, "host_advantage": 0},
        {"team": "Portugal", "group": "F", "fifa_rank": 6, "recent_form_score": 8.8, "goals_for": 2.2, "goals_against": 0.8, "host_advantage": 0},
        {"team": "Uruguay", "group": "F", "fifa_rank": 13, "recent_form_score": 8.1, "goals_for": 1.9, "goals_against": 0.9, "host_advantage": 0},
        {"team": "South Korea", "group": "F", "fifa_rank": 23, "recent_form_score": 7.2, "goals_for": 1.7, "goals_against": 1.1, "host_advantage": 0},
        {"team": "Iraq", "group": "F", "fifa_rank": 58, "recent_form_score": 6.0, "goals_for": 1.2, "goals_against": 1.6, "host_advantage": 0},
        {"team": "Germany", "group": "G", "fifa_rank": 7, "recent_form_score": 8.7, "goals_for": 2.1, "goals_against": 0.9, "host_advantage": 0},
        {"team": "Netherlands", "group": "G", "fifa_rank": 8, "recent_form_score": 8.6, "goals_for": 2.1, "goals_against": 0.8, "host_advantage": 0},
        {"team": "Nigeria", "group": "G", "fifa_rank": 31, "recent_form_score": 7.1, "goals_for": 1.6, "goals_against": 1.2, "host_advantage": 0},
        {"team": "Canada", "group": "G", "fifa_rank": 30, "recent_form_score": 7.0, "goals_for": 1.6, "goals_against": 1.3, "host_advantage": 1},
        {"team": "Belgium", "group": "H", "fifa_rank": 9, "recent_form_score": 8.4, "goals_for": 2.0, "goals_against": 1.0, "host_advantage": 0},
        {"team": "Colombia", "group": "H", "fifa_rank": 14, "recent_form_score": 8.0, "goals_for": 1.8, "goals_against": 0.9, "host_advantage": 0},
        {"team": "Serbia", "group": "H", "fifa_rank": 32, "recent_form_score": 6.9, "goals_for": 1.5, "goals_against": 1.2, "host_advantage": 0},
        {"team": "Panama", "group": "H", "fifa_rank": 44, "recent_form_score": 6.4, "goals_for": 1.3, "goals_against": 1.4, "host_advantage": 0},
        {"team": "Italy", "group": "I", "fifa_rank": 10, "recent_form_score": 8.3, "goals_for": 1.9, "goals_against": 0.8, "host_advantage": 0},
        {"team": "Austria", "group": "I", "fifa_rank": 25, "recent_form_score": 7.4, "goals_for": 1.7, "goals_against": 1.0, "host_advantage": 0},
        {"team": "Tunisia", "group": "I", "fifa_rank": 41, "recent_form_score": 6.6, "goals_for": 1.3, "goals_against": 1.2, "host_advantage": 0},
        {"team": "Saudi Arabia", "group": "I", "fifa_rank": 53, "recent_form_score": 6.2, "goals_for": 1.2, "goals_against": 1.5, "host_advantage": 0},
        {"team": "Norway", "group": "J", "fifa_rank": 22, "recent_form_score": 7.7, "goals_for": 1.8, "goals_against": 1.0, "host_advantage": 0},
        {"team": "Turkey", "group": "J", "fifa_rank": 27, "recent_form_score": 7.3, "goals_for": 1.7, "goals_against": 1.1, "host_advantage": 0},
        {"team": "Cameroon", "group": "J", "fifa_rank": 43, "recent_form_score": 6.5, "goals_for": 1.4, "goals_against": 1.3, "host_advantage": 0},
        {"team": "South Africa", "group": "J", "fifa_rank": 57, "recent_form_score": 6.1, "goals_for": 1.2, "goals_against": 1.4, "host_advantage": 0},
        {"team": "Poland", "group": "K", "fifa_rank": 28, "recent_form_score": 7.0, "goals_for": 1.6, "goals_against": 1.1, "host_advantage": 0},
        {"team": "Ukraine", "group": "K", "fifa_rank": 26, "recent_form_score": 7.2, "goals_for": 1.6, "goals_against": 1.0, "host_advantage": 0},
        {"team": "Algeria", "group": "K", "fifa_rank": 36, "recent_form_score": 6.8, "goals_for": 1.4, "goals_against": 1.1, "host_advantage": 0},
        {"team": "Venezuela", "group": "K", "fifa_rank": 47, "recent_form_score": 6.3, "goals_for": 1.3, "goals_against": 1.3, "host_advantage": 0},
        {"team": "Sweden", "group": "L", "fifa_rank": 29, "recent_form_score": 7.1, "goals_for": 1.6, "goals_against": 1.0, "host_advantage": 0},
        {"team": "Iran", "group": "L", "fifa_rank": 20, "recent_form_score": 7.5, "goals_for": 1.7, "goals_against": 0.9, "host_advantage": 0},
        {"team": "Paraguay", "group": "L", "fifa_rank": 38, "recent_form_score": 6.7, "goals_for": 1.4, "goals_against": 1.1, "host_advantage": 0},
        {"team": "Qatar", "group": "L", "fifa_rank": 55, "recent_form_score": 6.0, "goals_for": 1.2, "goals_against": 1.5, "host_advantage": 0},
    ]
    return pd.DataFrame(rows)


def prepare_team_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean an uploaded or sample dataset."""
    teams = df.copy()
    teams.columns = [column.strip() for column in teams.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in teams.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    teams = teams[REQUIRED_COLUMNS].copy()
    teams["team"] = teams["team"].astype(str).str.strip()
    teams["group"] = teams["group"].astype(str).str.strip().str.upper()

    numeric_columns = [
        "fifa_rank",
        "recent_form_score",
        "goals_for",
        "goals_against",
        "host_advantage",
    ]
    for column in numeric_columns:
        teams[column] = pd.to_numeric(teams[column], errors="coerce")

    if teams[numeric_columns].isna().any().any():
        raise ValueError("Numeric columns contain invalid or missing values.")

    if teams["team"].duplicated().any():
        duplicates = ", ".join(sorted(teams.loc[teams["team"].duplicated(), "team"].unique()))
        raise ValueError(f"Team names must be unique. Duplicate teams: {duplicates}")

    group_sizes = teams.groupby("group").size()
    if len(group_sizes) != 12 or not (group_sizes == 4).all():
        raise ValueError("The tournament requires 12 groups with exactly 4 teams in each group.")
    if set(teams["group"]) != set(GROUP_ORDER):
        raise ValueError("Group names must be the letters A through L.")

    return teams.sort_values(["group", "team"]).reset_index(drop=True)


def calculate_strength_scores(
    teams: pd.DataFrame,
    ranking_weight: float,
    recent_form_weight: float,
    host_advantage_weight: float,
) -> pd.DataFrame:
    """Build the strength score used across match prediction and simulation."""
    strengths = teams.copy()
    max_rank = strengths["fifa_rank"].max()
    max_goals_against = strengths["goals_against"].max()

    strengths["rank_score"] = (
        (max_rank - strengths["fifa_rank"] + 1) / max_rank
    ) * 100 * ranking_weight
    strengths["recent_form_component"] = strengths["recent_form_score"] * 10 * recent_form_weight
    strengths["attack_score"] = strengths["goals_for"] * 12
    strengths["defense_score"] = (max_goals_against - strengths["goals_against"] + 0.5) * 12
    strengths["host_component"] = strengths["host_advantage"] * 10 * host_advantage_weight

    strengths["strength_score"] = (
        strengths["rank_score"]
        + strengths["recent_form_component"]
        + strengths["attack_score"]
        + strengths["defense_score"]
        + strengths["host_component"]
    )
    return strengths


def logistic_win_probability(strength_difference: float) -> float:
    """Convert a strength gap into a smooth win probability."""
    scaled_difference = np.clip(strength_difference / 18, -60, 60)
    return float(1 / (1 + np.exp(-scaled_difference)))


def predict_match(team_a: pd.Series, team_b: pd.Series) -> dict[str, float | str]:
    """Return the predicted winner and win probability for a single matchup."""
    win_probability_a = logistic_win_probability(team_a["strength_score"] - team_b["strength_score"])
    if win_probability_a >= 0.5:
        return {
            "predicted_winner": team_a["team"],
            "win_probability": win_probability_a,
            "opponent": team_b["team"],
        }
    return {
        "predicted_winner": team_b["team"],
        "win_probability": 1 - win_probability_a,
        "opponent": team_a["team"],
    }


def generate_scoreline(
    team_a: pd.Series,
    team_b: pd.Series,
    strength_difference: float,
    outcome: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Create a simple scoreline that matches the chosen outcome."""
    mean_a = np.clip(
        0.5
        + team_a["goals_for"] * 0.45
        + team_b["goals_against"] * 0.18
        + team_a["host_advantage"] * 0.2
        + max(strength_difference, 0) / 90,
        0.2,
        4.0,
    )
    mean_b = np.clip(
        0.5
        + team_b["goals_for"] * 0.45
        + team_a["goals_against"] * 0.18
        + team_b["host_advantage"] * 0.2
        + max(-strength_difference, 0) / 90,
        0.2,
        4.0,
    )

    goals_a = int(rng.poisson(mean_a))
    goals_b = int(rng.poisson(mean_b))

    if outcome == "draw":
        shared_goals = int(round((goals_a + goals_b) / 2))
        goals_a = shared_goals
        goals_b = shared_goals
    elif outcome == "team_a" and goals_a <= goals_b:
        goals_a = goals_b + 1
    elif outcome == "team_b" and goals_b <= goals_a:
        goals_b = goals_a + 1

    return min(goals_a, 6), min(goals_b, 6)


def simulate_match(
    team_a: pd.Series,
    team_b: pd.Series,
    rng: np.random.Generator,
    knockout: bool = False,
) -> dict[str, float | int | str]:
    """Simulate one match using strength scores, a logistic win model, and optional draws."""
    strength_difference = team_a["strength_score"] - team_b["strength_score"]
    raw_win_probability_a = logistic_win_probability(strength_difference)
    draw_probability = 0.0 if knockout else float(np.clip(0.28 - abs(strength_difference) / 220, 0.08, 0.28))

    team_a_win_probability = raw_win_probability_a * (1 - draw_probability)
    team_b_win_probability = (1 - raw_win_probability_a) * (1 - draw_probability)

    if knockout:
        outcome = "team_a" if rng.random() < raw_win_probability_a else "team_b"
    else:
        outcome = rng.choice(
            ["team_a", "draw", "team_b"],
            p=[team_a_win_probability, draw_probability, team_b_win_probability],
        )

    goals_a, goals_b = generate_scoreline(team_a, team_b, strength_difference, outcome, rng)

    if knockout and goals_a == goals_b:
        if raw_win_probability_a >= 0.5:
            goals_a += 1
        else:
            goals_b += 1

    if goals_a > goals_b:
        winner = team_a["team"]
    elif goals_b > goals_a:
        winner = team_b["team"]
    else:
        winner = "Draw"

    return {
        "team_a": team_a["team"],
        "team_b": team_b["team"],
        "goals_a": goals_a,
        "goals_b": goals_b,
        "winner": winner,
        "team_a_win_probability": team_a_win_probability if not knockout else raw_win_probability_a,
    }


def rank_group_table(group_table: pd.DataFrame) -> pd.DataFrame:
    """Rank a group by points, goal difference, and goals scored."""
    ranked = group_table.copy()
    ranked["goal_difference"] = ranked["group_goals_for"] - ranked["group_goals_against"]
    return ranked.sort_values(
        by=["points", "goal_difference", "group_goals_for", "strength_score", "team"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def simulate_group_stage(
    teams: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, str | int]]]:
    """Play the 12 groups and return standings plus the 32 knockout qualifiers."""
    standings_rows: list[dict[str, str | float | int]] = []
    group_tables: dict[str, pd.DataFrame] = {}
    qualifiers: list[dict[str, str | float | int]] = []
    third_place_rows: list[dict[str, str | float | int]] = []

    for group_name in GROUP_ORDER:
        group_teams = teams[teams["group"] == group_name].reset_index(drop=True)
        group_table = group_teams.copy()
        group_table["points"] = 0
        group_table["wins"] = 0
        group_table["draws"] = 0
        group_table["losses"] = 0
        group_table["group_goals_for"] = 0
        group_table["group_goals_against"] = 0

        for team_a_index, team_b_index in combinations(group_table.index, 2):
            team_a = group_teams.loc[team_a_index]
            team_b = group_teams.loc[team_b_index]
            result = simulate_match(team_a, team_b, rng, knockout=False)

            group_table.loc[team_a_index, "group_goals_for"] += result["goals_a"]
            group_table.loc[team_a_index, "group_goals_against"] += result["goals_b"]
            group_table.loc[team_b_index, "group_goals_for"] += result["goals_b"]
            group_table.loc[team_b_index, "group_goals_against"] += result["goals_a"]

            if result["winner"] == "Draw":
                group_table.loc[team_a_index, ["points", "draws"]] += [1, 1]
                group_table.loc[team_b_index, ["points", "draws"]] += [1, 1]
            elif result["winner"] == team_a["team"]:
                group_table.loc[team_a_index, ["points", "wins"]] += [3, 1]
                group_table.loc[team_b_index, "losses"] += 1
            else:
                group_table.loc[team_b_index, ["points", "wins"]] += [3, 1]
                group_table.loc[team_a_index, "losses"] += 1

        ranked_group = rank_group_table(group_table)
        ranked_group["group_position"] = ranked_group.index + 1
        group_tables[group_name] = ranked_group
        standings_rows.extend(ranked_group.to_dict("records"))

        for _, row in ranked_group.iloc[:2].iterrows():
            qualifiers.append(
                {
                    **row.to_dict(),
                    "qualification_label": "Group winner" if row["group_position"] == 1 else "Group runner-up",
                    "qualification_seed": int(row["group_position"]),
                }
            )

        third_row = ranked_group.iloc[2].to_dict()
        third_place_rows.append(
            {
                **third_row,
                "qualification_label": "Third place",
                "qualification_seed": 3,
            }
        )

    best_third_place_teams = pd.DataFrame(third_place_rows).sort_values(
        by=["points", "goal_difference", "group_goals_for", "strength_score", "team"],
        ascending=[False, False, False, False, True],
    ).head(8)

    qualifiers.extend(best_third_place_teams.to_dict("records"))
    qualifiers_df = pd.DataFrame(qualifiers).sort_values(
        by=["qualification_seed", "points", "goal_difference", "group_goals_for", "strength_score", "team"],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)
    qualifiers_df["knockout_seed"] = qualifiers_df.index + 1

    standings_df = pd.DataFrame(standings_rows).sort_values(["group", "group_position"])
    return qualifiers_df, group_tables, standings_df.to_dict("records")


def build_grouped_team_records(strengths: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    """Convert the scored teams table into plain Python records grouped by letter."""
    grouped_records: dict[str, list[dict[str, object]]] = {}
    for group_name in GROUP_ORDER:
        group_rows = strengths.loc[strengths["group"] == group_name].to_dict("records")
        grouped_records[group_name] = group_rows
    return grouped_records


def sort_group_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Sort group-stage rows using the tournament tiebreakers."""
    return sorted(
        rows,
        key=lambda row: (
            -int(row["points"]),
            -int(row["goal_difference"]),
            -int(row["group_goals_for"]),
            -float(row["strength_score"]),
            str(row["team"]),
        ),
    )


def simulate_group_stage_fast(
    grouped_teams: dict[str, list[dict[str, object]]],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Fast qualifier-only group simulation used for bulk probability runs."""
    qualifiers: list[dict[str, object]] = []
    third_place_rows: list[dict[str, object]] = []

    for group_name in GROUP_ORDER:
        teams = grouped_teams[group_name]
        rows = {
            str(team["team"]): {
                **team,
                "points": 0,
                "group_goals_for": 0,
                "group_goals_against": 0,
                "goal_difference": 0,
            }
            for team in teams
        }

        for team_a, team_b in combinations(teams, 2):
            result = simulate_match(team_a, team_b, rng, knockout=False)
            row_a = rows[str(team_a["team"])]
            row_b = rows[str(team_b["team"])]

            row_a["group_goals_for"] += int(result["goals_a"])
            row_a["group_goals_against"] += int(result["goals_b"])
            row_b["group_goals_for"] += int(result["goals_b"])
            row_b["group_goals_against"] += int(result["goals_a"])

            if result["winner"] == "Draw":
                row_a["points"] += 1
                row_b["points"] += 1
            elif result["winner"] == team_a["team"]:
                row_a["points"] += 3
            else:
                row_b["points"] += 3

        ranked_rows = []
        for row in rows.values():
            row["goal_difference"] = int(row["group_goals_for"]) - int(row["group_goals_against"])
            ranked_rows.append(row)

        ranked_rows = sort_group_rows(ranked_rows)

        for position, row in enumerate(ranked_rows, start=1):
            row["group_position"] = position
            if position == 1:
                qualifiers.append({**row, "qualification_label": "Group winner", "qualification_seed": 1})
            elif position == 2:
                qualifiers.append({**row, "qualification_label": "Group runner-up", "qualification_seed": 2})
            elif position == 3:
                third_place_rows.append({**row, "qualification_label": "Third place", "qualification_seed": 3})

    best_third_place_teams = sort_group_rows(third_place_rows)[:8]
    qualifiers.extend(best_third_place_teams)
    return sorted(
        qualifiers,
        key=lambda row: (
            int(row["qualification_seed"]),
            -int(row["points"]),
            -int(row["goal_difference"]),
            -int(row["group_goals_for"]),
            -float(row["strength_score"]),
            str(row["team"]),
        ),
    )


def simulate_knockout_round_fast(
    participants: list[dict[str, object]],
    round_name: str,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Fast knockout simulation used when only the eventual champion is needed."""
    winners: list[dict[str, object]] = []

    if round_name == "Round of 32":
        pairings = [(index, len(participants) - 1 - index) for index in range(len(participants) // 2)]
    else:
        pairings = [(index, index + 1) for index in range(0, len(participants), 2)]

    for first_index, second_index in pairings:
        team_a = participants[first_index]
        team_b = participants[second_index]
        result = simulate_match(team_a, team_b, rng, knockout=True)
        winners.append(team_a if result["winner"] == team_a["team"] else team_b)

    return winners


def simulate_tournament_winner(
    grouped_teams: dict[str, list[dict[str, object]]],
    seed: int | None = None,
) -> str:
    """Fast champion-only simulation for repeated Monte Carlo runs."""
    rng = np.random.default_rng(seed)
    qualifiers = simulate_group_stage_fast(grouped_teams, rng)
    round_of_32_winners = simulate_knockout_round_fast(qualifiers, "Round of 32", rng)
    round_of_16_winners = simulate_knockout_round_fast(round_of_32_winners, "Round of 16", rng)
    quarterfinal_winners = simulate_knockout_round_fast(round_of_16_winners, "Quarterfinals", rng)
    semifinal_winners = simulate_knockout_round_fast(quarterfinal_winners, "Semifinals", rng)
    final_winner = simulate_knockout_round_fast(semifinal_winners, "Final", rng)
    return str(final_winner[0]["team"])


def simulate_knockout_round(
    teams: pd.DataFrame,
    round_name: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, str | int]]]:
    """Simulate one knockout round and return winners plus match summaries."""
    participants = teams.reset_index(drop=True)
    match_results: list[dict[str, str | int]] = []
    winners: list[pd.Series] = []

    if round_name == "Round of 32":
        pairings = [(index, len(participants) - 1 - index) for index in range(len(participants) // 2)]
    else:
        pairings = [(index, index + 1) for index in range(0, len(participants), 2)]

    for first_index, second_index in pairings:
        team_a = participants.iloc[first_index]
        team_b = participants.iloc[second_index]
        result = simulate_match(team_a, team_b, rng, knockout=True)
        match_results.append(
            {
                "round": round_name,
                "team_a": result["team_a"],
                "team_b": result["team_b"],
                "score": f"{result['goals_a']}-{result['goals_b']}",
                "winner": result["winner"],
            }
        )
        winner_name = result["winner"]
        winners.append(participants.loc[participants["team"] == winner_name].iloc[0])

    return pd.DataFrame(winners).reset_index(drop=True), match_results


def simulate_tournament_from_strengths(
    strengths: pd.DataFrame,
    seed: int | None = None,
    include_details: bool = True,
) -> dict[str, object]:
    """Run one full tournament simulation from a pre-scored teams table."""
    rng = np.random.default_rng(seed)
    qualifiers, group_tables, standings = simulate_group_stage(strengths, rng)

    round_of_32_winners, round_of_32_results = simulate_knockout_round(qualifiers, "Round of 32", rng)
    round_of_16_winners, round_of_16_results = simulate_knockout_round(round_of_32_winners, "Round of 16", rng)
    quarterfinal_winners, quarterfinal_results = simulate_knockout_round(round_of_16_winners, "Quarterfinals", rng)
    semifinal_winners, semifinal_results = simulate_knockout_round(quarterfinal_winners, "Semifinals", rng)
    final_winner_df, final_results = simulate_knockout_round(semifinal_winners, "Final", rng)

    champion = final_winner_df.iloc[0]["team"]
    if not include_details:
        return {"champion": champion}

    knockout_results = (
        round_of_32_results
        + round_of_16_results
        + quarterfinal_results
        + semifinal_results
        + final_results
    )

    return {
        "champion": champion,
        "strengths": strengths,
        "group_tables": group_tables,
        "standings": standings,
        "qualifiers": qualifiers,
        "knockout_results": pd.DataFrame(knockout_results),
    }


def simulate_tournament(
    teams: pd.DataFrame,
    ranking_weight: float,
    recent_form_weight: float,
    host_advantage_weight: float,
    seed: int | None = None,
) -> dict[str, object]:
    """Run one full tournament simulation."""
    strengths = calculate_strength_scores(teams, ranking_weight, recent_form_weight, host_advantage_weight)
    return simulate_tournament_from_strengths(strengths, seed=seed, include_details=True)


@st.cache_data(show_spinner=False)
def run_tournament_simulations(
    teams: pd.DataFrame,
    simulations: int,
    ranking_weight: float,
    recent_form_weight: float,
    host_advantage_weight: float,
) -> pd.DataFrame:
    """Run many tournament simulations and calculate champion probability."""
    strengths = calculate_strength_scores(teams, ranking_weight, recent_form_weight, host_advantage_weight)
    grouped_teams = build_grouped_team_records(strengths)
    champion_counts: dict[str, int] = {team_name: 0 for team_name in teams["team"]}

    for _ in range(simulations):
        champion = simulate_tournament_winner(grouped_teams)
        champion_counts[champion] += 1

    probability_table = pd.DataFrame(
        {
            "team": list(champion_counts.keys()),
            "titles_won": list(champion_counts.values()),
        }
    )
    probability_table["champion_probability"] = probability_table["titles_won"] / simulations
    return probability_table.sort_values(["champion_probability", "team"], ascending=[False, True]).reset_index(drop=True)


def render_app() -> None:
    st.set_page_config(page_title="World Cup 2026 Winner Predictor", layout="wide")
    st.title("World Cup 2026 Winner Predictor")
    st.write(
        """
        Welcome to a simple World Cup simulator built with Python and Streamlit.
        It estimates team strength, simulates the 48-team tournament, and shows
        which countries are most likely to win the 2026 FIFA World Cup.
        """
    )

    st.sidebar.header("Simulation Controls")
    simulations = st.sidebar.number_input("Number of simulations", min_value=1000, max_value=10000, value=1000, step=250)
    host_advantage_weight = st.sidebar.slider("Host advantage weight", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    recent_form_weight = st.sidebar.slider("Recent form weight", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    ranking_weight = st.sidebar.slider("Ranking weight", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    uploaded_file = st.sidebar.file_uploader("Upload a teams CSV", type="csv")

    try:
        if uploaded_file is not None:
            base_teams = prepare_team_data(pd.read_csv(uploaded_file))
            data_source = "Uploaded CSV"
        else:
            base_teams = prepare_team_data(load_sample_teams())
            data_source = "Built-in sample data"
    except ValueError as error:
        st.error(str(error))
        st.stop()

    strength_table = calculate_strength_scores(
        base_teams,
        ranking_weight=ranking_weight,
        recent_form_weight=recent_form_weight,
        host_advantage_weight=host_advantage_weight,
    )

    st.subheader("Team data")
    st.caption(f"Source: {data_source}")
    st.dataframe(
        strength_table[
            REQUIRED_COLUMNS
            + ["rank_score", "attack_score", "defense_score", "strength_score"]
        ].round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Match predictor")
    team_names = strength_table["team"].tolist()
    team_a_name = st.selectbox("Team 1", team_names, index=0)
    team_b_options = [team_name for team_name in team_names if team_name != team_a_name]
    team_b_name = st.selectbox("Team 2", team_b_options, index=0)

    team_a = strength_table.loc[strength_table["team"] == team_a_name].iloc[0]
    team_b = strength_table.loc[strength_table["team"] == team_b_name].iloc[0]
    match_prediction = predict_match(team_a, team_b)

    st.metric(
        "Predicted winner",
        match_prediction["predicted_winner"],
        f"{match_prediction['win_probability']:.1%} win probability",
    )

    with st.spinner("Running tournament simulations..."):
        champion_probabilities = run_tournament_simulations(
            base_teams,
            simulations=int(simulations),
            ranking_weight=ranking_weight,
            recent_form_weight=recent_form_weight,
            host_advantage_weight=host_advantage_weight,
        )
        sample_tournament = simulate_tournament(
            base_teams,
            ranking_weight=ranking_weight,
            recent_form_weight=recent_form_weight,
            host_advantage_weight=host_advantage_weight,
            seed=42,
        )

    st.subheader("Champion probabilities")
    st.dataframe(
        champion_probabilities.assign(
            champion_probability=lambda frame: (frame["champion_probability"] * 100).round(2).astype(str) + "%"
        ),
        use_container_width=True,
        hide_index=True,
    )

    top_10 = champion_probabilities.head(10).copy()
    top_10["champion_probability_pct"] = top_10["champion_probability"] * 100
    figure = px.bar(
        top_10.sort_values("champion_probability_pct"),
        x="champion_probability_pct",
        y="team",
        orientation="h",
        labels={"champion_probability_pct": "Champion probability (%)", "team": "Team"},
        title="Top 10 most likely winners",
    )
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Sample tournament preview")
    st.write(f"Sample simulated champion: **{sample_tournament['champion']}**")

    qualifiers = sample_tournament["qualifiers"][
        ["team", "group", "qualification_label", "points", "goal_difference", "group_goals_for", "knockout_seed"]
    ]
    st.markdown("**Knockout qualifiers**")
    st.dataframe(qualifiers, use_container_width=True, hide_index=True)

    st.markdown("**Knockout bracket results**")
    st.dataframe(sample_tournament["knockout_results"], use_container_width=True, hide_index=True)

    with st.expander("Show group stage tables from the sample tournament"):
        for group_name in GROUP_ORDER:
            st.markdown(f"**Group {group_name}**")
            st.dataframe(sample_tournament["group_tables"][group_name], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_app()
