import itertools

import numpy as np
import pandas as pd

from src.bracket import CONFIRMED_GROUP_POSITIONS
from src.bracket import (
    CONFIRMED_MATCH_CONTEXTS,
    CONFIRMED_KNOCKOUT_WINNERS,
    CONFIRMED_ROUND_OF_32,
    CONFIRMED_THIRD_PLACE_QUALIFIERS,
)
from src.bracket import KNOCKOUT_PATHS, build_round_of_32
from src.model import advancement_probability, estimate_goals, predict_match_winner


KNOCKOUT_ROUND_NAMES = {
    32: "Round of 32",
    16: "Round of 16",
    8: "Quarterfinals",
    4: "Semifinals",
    2: "Final",
}

COMPLETED_MATCH_STATUSES = {"FT", "AET", "PEN"}


def apply_match_context(team_a, team_b, match_number):
    """Attach bounded venue and altitude effects for a specific fixture."""
    context = CONFIRMED_MATCH_CONTEXTS.get(match_number)
    if not context:
        return team_a, team_b

    contextualized = []
    altitude_ratio = min(max(float(context.get("altitude_m", 0)) / 2500, 0), 1)
    for team in (team_a, team_b):
        adjusted = team.copy()
        team_name = team.get("team", getattr(team, "name", None))
        is_home_team = team_name == context.get("home_team")
        adjusted["stadium"] = context.get("stadium", "Unknown")
        adjusted["venue_city"] = context.get("city", "Unknown")
        adjusted["venue_country"] = context.get("country", "Unknown")
        adjusted["altitude_m"] = context.get("altitude_m", 0)
        # The host receives familiarity plus altitude acclimation. The visitor
        # receives a smaller, bounded altitude-strain penalty.
        adjusted["match_context_component"] = (
            0.75 + 1.5 * altitude_ratio
            if is_home_team
            else -0.75 * altitude_ratio
        )
        contextualized.append(adjusted)
    return tuple(contextualized)


def simulate_group_match(team_a, team_b):
    """Simulate a group match with points, goals, and possible draws."""
    winner, result_type = predict_match_winner(team_a, team_b, allow_draw=True)

    if result_type == "draw":
        goals = np.random.choice([0, 1, 2], p=[0.25, 0.55, 0.20])
        return goals, goals, None

    # Sample from the scoring model conditional on the already sampled outcome.
    # Rejection sampling preserves plausible margins instead of rewriting a draw
    # into an artificial one-goal victory.
    for _ in range(100):
        goals_a = estimate_goals(team_a, team_b)
        goals_b = estimate_goals(team_b, team_a)
        if winner == team_a["team"] and goals_a > goals_b:
            break
        if winner == team_b["team"] and goals_b > goals_a:
            break
    else:
        goals_a, goals_b = (
            (1, 0) if winner == team_a["team"] else (0, 1)
        )

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


def find_completed_result(fixtures, team_a, team_b):
    """Return a completed fixture score for two teams when one is available."""
    if fixtures is None or fixtures.empty:
        return None
    required = {"team_home", "team_away", "match_status", "goals_home", "goals_away"}
    if not required.issubset(fixtures.columns):
        return None

    match = fixtures.loc[
        (
            (fixtures["team_home"] == team_a["team"])
            & (fixtures["team_away"] == team_b["team"])
        )
        | (
            (fixtures["team_home"] == team_b["team"])
            & (fixtures["team_away"] == team_a["team"])
        )
    ]
    match = match.loc[match["match_status"].isin(COMPLETED_MATCH_STATUSES)]
    if match.empty:
        return None

    fixture = match.iloc[-1]
    goals_home = pd.to_numeric(fixture["goals_home"], errors="coerce")
    goals_away = pd.to_numeric(fixture["goals_away"], errors="coerce")
    if pd.isna(goals_home) or pd.isna(goals_away):
        return None
    if fixture["team_home"] == team_a["team"]:
        return int(goals_home), int(goals_away)
    return int(goals_away), int(goals_home)


def find_completed_knockout_winner(fixtures, team_a, team_b):
    """Return the winner of a completed knockout fixture, including penalties."""
    if fixtures is None or fixtures.empty:
        return None
    required = {"team_home", "team_away", "match_status", "goals_home", "goals_away"}
    if not required.issubset(fixtures.columns):
        return None
    names = {team_a["team"], team_b["team"]}
    matches = fixtures.loc[
        fixtures["team_home"].isin(names) & fixtures["team_away"].isin(names)
    ]
    matches = matches.loc[matches["match_status"].isin(COMPLETED_MATCH_STATUSES)]
    if matches.empty:
        return None
    fixture = matches.iloc[-1]
    home_goals = pd.to_numeric(fixture["goals_home"], errors="coerce")
    away_goals = pd.to_numeric(fixture["goals_away"], errors="coerce")
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    if home_goals > away_goals:
        return fixture["team_home"]
    if away_goals > home_goals:
        return fixture["team_away"]
    home_penalties = pd.to_numeric(fixture.get("penalty_home"), errors="coerce")
    away_penalties = pd.to_numeric(fixture.get("penalty_away"), errors="coerce")
    if pd.isna(home_penalties) or pd.isna(away_penalties):
        return None
    if home_penalties == away_penalties:
        return None
    return (
        fixture["team_home"]
        if home_penalties > away_penalties
        else fixture["team_away"]
    )


def simulate_group_stage(teams, fixtures=None):
    """Simulate all 12 groups and return 32 advancing teams."""
    group_winners_and_runners_up = []
    third_place_teams = []

    for group_name, group_df in teams.groupby("group", sort=True):
        team_records = group_df.to_dict("records")
        standings = {
            team["team"]: {
                **team,
                "points": 0,
                "goals_for_sim": 0,
                "goals_against_sim": 0,
                "goal_difference": 0,
            }
            for team in team_records
        }

        for team_a, team_b in itertools.combinations(team_records, 2):
            completed_result = find_completed_result(fixtures, team_a, team_b)
            if completed_result:
                goals_a, goals_b = completed_result
            else:
                goals_a, goals_b, _ = simulate_group_match(team_a, team_b)

            standing_a = standings[team_a["team"]]
            standing_b = standings[team_b["team"]]
            standing_a["goals_for_sim"] += goals_a
            standing_a["goals_against_sim"] += goals_b
            standing_b["goals_for_sim"] += goals_b
            standing_b["goals_against_sim"] += goals_a
            if goals_a > goals_b:
                standing_a["points"] += 3
            elif goals_b > goals_a:
                standing_b["points"] += 3
            else:
                standing_a["points"] += 1
                standing_b["points"] += 1

        for standing in standings.values():
            standing["goal_difference"] = (
                standing["goals_for_sim"] - standing["goals_against_sim"]
            )
            standing["eligible_for_knockout"] = not standing.get(
                "eliminated", False
            )
        ranked_group = pd.DataFrame(standings.values()).sort_values(
            [
                "eligible_for_knockout",
                "points",
                "goal_difference",
                "goals_for_sim",
                "strength_score",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        confirmed_positions = CONFIRMED_GROUP_POSITIONS.get(group_name, {})
        if confirmed_positions:
            confirmed_names = set(confirmed_positions.values())
            unresolved = ranked_group.loc[
                ~ranked_group["team"].isin(confirmed_names)
            ].to_dict("records")
            ordered_rows = []
            for position in range(1, len(ranked_group) + 1):
                confirmed_team = confirmed_positions.get(position)
                if confirmed_team:
                    ordered_rows.append(
                        ranked_group.loc[
                            ranked_group["team"] == confirmed_team
                        ].iloc[0]
                    )
                else:
                    ordered_rows.append(pd.Series(unresolved.pop(0)))
            ranked_group = pd.DataFrame(ordered_rows).reset_index(drop=True)

        ranked_group["group_position"] = range(1, len(ranked_group) + 1)
        group_winners_and_runners_up.append(ranked_group.head(2))
        third_place_teams.append(ranked_group.iloc[[2]])

    automatic_qualifiers = pd.concat(group_winners_and_runners_up, ignore_index=True)
    third_place_rankings = pd.concat(third_place_teams, ignore_index=True).sort_values(
        ["points", "goal_difference", "goals_for_sim", "strength_score"],
        ascending=[False, False, False, False],
    )
    locked_third_place_teams = third_place_rankings.loc[
        third_place_rankings["team"].isin(CONFIRMED_THIRD_PLACE_QUALIFIERS)
    ]
    remaining_third_place_teams = third_place_rankings.loc[
        ~third_place_rankings["team"].isin(CONFIRMED_THIRD_PLACE_QUALIFIERS)
        & ~third_place_rankings["eliminated"]
    ]
    best_third_place_teams = pd.concat(
        [
            locked_third_place_teams,
            remaining_third_place_teams.head(8 - len(locked_third_place_teams)),
        ],
        ignore_index=True,
    )

    return pd.concat([automatic_qualifiers, best_third_place_teams], ignore_index=True)


def simulate_knockout_round(teams, route_difficulty, round_name=None):
    """Simulate one knockout round and return its winners and optional pairings."""
    winners = []
    pairings = []

    for index in range(0, len(teams), 2):
        team_a = teams.iloc[index]
        team_b = teams.iloc[index + 1]
        probability_a = advancement_probability(team_a, team_b)
        winner_name, _ = predict_match_winner(team_a, team_b, allow_draw=False)
        route_difficulty[team_a["team"]] += team_b["strength_score"]
        route_difficulty[team_b["team"]] += team_a["strength_score"]
        winners.append(team_a if winner_name == team_a["team"] else team_b)

        if round_name:
            pairings.append(
                {
                    "round": round_name,
                    "match": index // 2 + 1,
                    "team_a": team_a["team"],
                    "team_b": team_b["team"],
                    "team_a_win_probability": probability_a * 100,
                    "team_b_win_probability": (1 - probability_a) * 100,
                    "predicted_winner": winner_name,
                }
            )

    winners_df = pd.DataFrame(winners).reset_index(drop=True)
    if round_name:
        return winners_df, pairings
    return winners_df


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


def simulate_official_bracket(qualified_teams, all_teams=None, fixtures=None):
    """Simulate FIFA's fixed knockout match-number progression."""
    team_pool = all_teams if all_teams is not None else qualified_teams
    route_difficulty = {
        team: 0 for team in team_pool["team"]
    }
    winners_by_match = {}
    pairings = []

    def play_match(round_name, match_number, team_a, team_b):
        team_a, team_b = apply_match_context(team_a, team_b, match_number)
        probability_a = advancement_probability(team_a, team_b)
        winner_name = find_completed_knockout_winner(
            fixtures, team_a, team_b
        )
        if winner_name is None:
            winner_name = CONFIRMED_KNOCKOUT_WINNERS.get(match_number)
        if winner_name is None:
            winner_name, _ = predict_match_winner(
                team_a, team_b, allow_draw=False
            )
        elif winner_name not in {team_a["team"], team_b["team"]}:
            raise ValueError(
                f"Confirmed winner {winner_name} did not play match {match_number}."
            )
        winner = team_a if winner_name == team_a["team"] else team_b
        route_difficulty[team_a["team"]] += team_b["strength_score"]
        route_difficulty[team_b["team"]] += team_a["strength_score"]
        winners_by_match[match_number] = winner
        pairings.append(
            {
                "round": round_name,
                "match": match_number,
                "team_a": team_a["team"],
                "team_b": team_b["team"],
                "team_a_win_probability": probability_a * 100,
                "team_b_win_probability": (1 - probability_a) * 100,
                "predicted_winner": winner_name,
            }
        )

    round_of_32 = {
        match_number: (team_a, team_b)
        for match_number, team_a, team_b in build_round_of_32(qualified_teams)
    }
    teams_by_name = {
        row["team"]: row for _, row in team_pool.iterrows()
    }
    for match_number, (team_a_name, team_b_name) in CONFIRMED_ROUND_OF_32.items():
        round_of_32[match_number] = (
            teams_by_name[team_a_name],
            teams_by_name[team_b_name],
        )

    for match_number, (team_a, team_b) in sorted(round_of_32.items()):
        play_match("Round of 32", match_number, team_a, team_b)

    for round_name, matches in KNOCKOUT_PATHS.items():
        for match_number, (source_a, source_b) in matches.items():
            play_match(
                round_name,
                match_number,
                winners_by_match[source_a],
                winners_by_match[source_b],
            )

    champion = winners_by_match[104]["team"]
    return pd.DataFrame(pairings), champion, route_difficulty[champion]


def simulate_tournament(teams, fixtures=None):
    """Run one full tournament and return champion details."""
    qualified_teams = simulate_group_stage(teams, fixtures)
    _, champion, champion_route_difficulty = simulate_official_bracket(
        qualified_teams, teams, fixtures
    )
    return {
        "champion": champion,
        "champion_route_difficulty": champion_route_difficulty,
    }


def predict_tournament_bracket(teams, fixtures=None):
    """Project one tournament and retain every knockout pairing."""
    qualified_teams = simulate_group_stage(teams, fixtures)
    pairings, champion, _ = simulate_official_bracket(
        qualified_teams, teams, fixtures
    )
    return pairings, champion


def run_simulations(
    teams, number_of_simulations, fixtures=None, progress_callback=None
):
    """Run many tournaments and convert champion counts into probabilities."""
    eligible_teams = (
        teams.loc[~teams["eliminated"]]
        if "eliminated" in teams.columns
        else teams
    )
    champion_counts = {team: 0 for team in eligible_teams["team"]}
    route_difficulty_totals = {team: 0 for team in eligible_teams["team"]}

    for simulation_number in range(number_of_simulations):
        simulation_result = simulate_tournament(teams, fixtures)
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


def run_pairing_simulations(
    teams, number_of_simulations, fixtures=None, progress_callback=None
):
    """Estimate how often every knockout pairing occurs."""
    pairing_counts = {}

    for simulation_number in range(number_of_simulations):
        pairings, _ = predict_tournament_bracket(teams, fixtures)
        for pairing in pairings.itertuples(index=False):
            teams_in_pairing = tuple(sorted((pairing.team_a, pairing.team_b)))
            key = (pairing.round, *teams_in_pairing)
            pairing_counts[key] = pairing_counts.get(key, 0) + 1

        if progress_callback and simulation_number % max(
            1, number_of_simulations // 100
        ) == 0:
            progress_callback((simulation_number + 1) / number_of_simulations)

    rows = [
        {
            "round": round_name,
            "team_a": team_a,
            "team_b": team_b,
            "pairing_probability": count / number_of_simulations * 100,
            "simulations": count,
        }
        for (round_name, team_a, team_b), count in pairing_counts.items()
    ]
    probabilities = pd.DataFrame(rows).sort_values(
        ["round", "pairing_probability"], ascending=[True, False]
    )
    return enforce_confirmed_round_of_32(probabilities, number_of_simulations)


def enforce_confirmed_round_of_32(pairing_probabilities, number_of_simulations):
    """Guarantee confirmed fixtures cannot coexist with simulated alternatives."""
    confirmed_teams = {
        team
        for pairing in CONFIRMED_ROUND_OF_32.values()
        for team in pairing
    }
    is_round_of_32 = pairing_probabilities["round"] == "Round of 32"
    involves_confirmed_team = pairing_probabilities["team_a"].isin(
        confirmed_teams
    ) | pairing_probabilities["team_b"].isin(confirmed_teams)
    probabilities = pairing_probabilities.loc[
        ~(is_round_of_32 & involves_confirmed_team)
    ].copy()

    confirmed_rows = pd.DataFrame(
        [
            {
                "round": "Round of 32",
                "team_a": min(team_a, team_b),
                "team_b": max(team_a, team_b),
                "pairing_probability": 100.0,
                "simulations": number_of_simulations,
            }
            for team_a, team_b in CONFIRMED_ROUND_OF_32.values()
        ]
    )
    return pd.concat([probabilities, confirmed_rows], ignore_index=True).sort_values(
        ["round", "pairing_probability"], ascending=[True, False]
    )
