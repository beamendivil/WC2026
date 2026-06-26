import itertools

import numpy as np
import pandas as pd

from src.model import estimate_goals, predict_match_winner


def simulate_group_match(team_a, team_b):
    """Simulate a group match with points, goals, and possible draws."""
    winner, result_type = predict_match_winner(team_a, team_b, allow_draw=True)

    if result_type == "draw":
        goals = np.random.choice([0, 1, 2], p=[0.25, 0.55, 0.20])
        return goals, goals, None

    goals_a = estimate_goals(team_a, team_b)
    goals_b = estimate_goals(team_b, team_a)

    # Force the scoreline to match the simulated winner.
    if winner == team_a["team"] and goals_a <= goals_b:
        goals_a = goals_b + 1
    elif winner == team_b["team"] and goals_b <= goals_a:
        goals_b = goals_a + 1

    return goals_a, goals_b, winner


def build_empty_group_table(group_df):
    """Create a standings table for a group."""
    table = group_df[["team", "group", "strength_score"]].copy()
    table["points"] = 0
    table["goals_for_sim"] = 0
    table["goals_against_sim"] = 0
    table["goal_difference"] = 0
    return table.set_index("team")


def update_group_table(table, team_a, team_b, goals_a, goals_b):
    """Apply one match result to the standings table."""
    table.loc[team_a, "goals_for_sim"] += goals_a
    table.loc[team_a, "goals_against_sim"] += goals_b
    table.loc[team_b, "goals_for_sim"] += goals_b
    table.loc[team_b, "goals_against_sim"] += goals_a

    if goals_a > goals_b:
        table.loc[team_a, "points"] += 3
    elif goals_b > goals_a:
        table.loc[team_b, "points"] += 3
    else:
        table.loc[team_a, "points"] += 1
        table.loc[team_b, "points"] += 1


def rank_group_table(table):
    """Rank teams by points, goal difference, goals for, then strength."""
    ranked = table.copy()
    ranked["goal_difference"] = (
        ranked["goals_for_sim"] - ranked["goals_against_sim"]
    )
    return ranked.sort_values(
        ["points", "goal_difference", "goals_for_sim", "strength_score"],
        ascending=[False, False, False, False],
    ).reset_index()


def simulate_group_stage(teams):
    """Simulate all 12 groups and return 32 advancing teams."""
    group_winners_and_runners_up = []
    third_place_teams = []

    for _, group_df in teams.groupby("group", sort=True):
        standings = build_empty_group_table(group_df)
        team_records = group_df.to_dict("records")

        for team_a, team_b in itertools.combinations(team_records, 2):
            goals_a, goals_b, _ = simulate_group_match(team_a, team_b)
            update_group_table(
                standings, team_a["team"], team_b["team"], goals_a, goals_b
            )

        ranked_group = rank_group_table(standings)
        group_winners_and_runners_up.append(ranked_group.head(2))
        third_place_teams.append(ranked_group.iloc[[2]])

    automatic_qualifiers = pd.concat(group_winners_and_runners_up, ignore_index=True)
    third_place_rankings = pd.concat(third_place_teams, ignore_index=True).sort_values(
        ["points", "goal_difference", "goals_for_sim", "strength_score"],
        ascending=[False, False, False, False],
    )
    best_third_place_teams = third_place_rankings.head(8)

    return pd.concat([automatic_qualifiers, best_third_place_teams], ignore_index=True)


def simulate_knockout_round(teams, route_difficulty):
    """Simulate one knockout round and return the winners."""
    winners = []

    for index in range(0, len(teams), 2):
        team_a = teams.iloc[index]
        team_b = teams.iloc[index + 1]
        winner_name, _ = predict_match_winner(team_a, team_b, allow_draw=False)
        route_difficulty[team_a["team"]] += team_b["strength_score"]
        route_difficulty[team_b["team"]] += team_a["strength_score"]
        winners.append(team_a if winner_name == team_a["team"] else team_b)

    return pd.DataFrame(winners).reset_index(drop=True)


def seed_knockout_teams(qualified_teams):
    """Pair high-performing teams against lower-performing teams."""
    seeded = qualified_teams.sort_values(
        ["points", "goal_difference", "goals_for_sim", "strength_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    bracket_order = []
    for index in range(len(seeded) // 2):
        bracket_order.append(seeded.iloc[index])
        bracket_order.append(seeded.iloc[-(index + 1)])

    return pd.DataFrame(bracket_order).reset_index(drop=True)


def simulate_tournament(teams):
    """Run one full tournament and return champion details."""
    qualified_teams = simulate_group_stage(teams)
    knockout_teams = seed_knockout_teams(qualified_teams)
    route_difficulty = {team: 0 for team in teams["team"]}

    while len(knockout_teams) > 1:
        knockout_teams = simulate_knockout_round(knockout_teams, route_difficulty)

    champion = knockout_teams.iloc[0]["team"]
    return {
        "champion": champion,
        "champion_route_difficulty": route_difficulty[champion],
    }


def run_simulations(teams, number_of_simulations, progress_callback=None):
    """Run many tournaments and convert champion counts into probabilities."""
    champion_counts = {team: 0 for team in teams["team"]}
    route_difficulty_totals = {team: 0 for team in teams["team"]}

    for simulation_number in range(number_of_simulations):
        simulation_result = simulate_tournament(teams)
        champion = simulation_result["champion"]
        champion_counts[champion] += 1
        route_difficulty_totals[champion] += simulation_result[
            "champion_route_difficulty"
        ]

        if progress_callback and simulation_number % max(
            1, number_of_simulations // 100
        ) == 0:
            progress_callback((simulation_number + 1) / number_of_simulations)

    results = pd.DataFrame(
        {
            "team": list(champion_counts.keys()),
            "championships": list(champion_counts.values()),
        }
    )
    results["champion_probability"] = (
        results["championships"] / number_of_simulations * 100
    )
    results["avg_champion_route_difficulty"] = results.apply(
        lambda row: route_difficulty_totals[row["team"]] / row["championships"]
        if row["championships"] > 0
        else 0,
        axis=1,
    )
    return results.sort_values("champion_probability", ascending=False)
