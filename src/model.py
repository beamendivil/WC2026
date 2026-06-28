import numpy as np

from src.historical_model import historical_match_probabilities


def win_probability(team_a_strength, team_b_strength):
    """Convert strength difference into Team A win probability."""
    strength_difference = team_a_strength - team_b_strength
    return 1 / (1 + np.exp(-strength_difference / 6))


def match_probabilities(team_a, team_b):
    """Return regulation probabilities with a safe heuristic fallback."""
    if "historical_elo" in team_a and "historical_elo" in team_b:
        return historical_match_probabilities(team_a, team_b)

    probability_a = win_probability(
        team_a["strength_score"], team_b["strength_score"]
    )
    draw_probability = max(0.12, 0.28 - abs(probability_a - 0.5) * 0.35)
    return {
        "team_a": probability_a * (1 - draw_probability),
        "draw": draw_probability,
        "team_b": (1 - probability_a) * (1 - draw_probability),
    }


def advancement_probability(team_a, team_b):
    """Estimate Team A's chance to advance from a knockout match."""
    probabilities = match_probabilities(team_a, team_b)
    decisive_total = probabilities["team_a"] + probabilities["team_b"]
    decisive_a = (
        probabilities["team_a"] / decisive_total if decisive_total else 0.5
    )
    return probabilities["team_a"] + probabilities["draw"] * decisive_a


def predict_match_winner(team_a, team_b, allow_draw=False):
    """Simulate one match and return the result."""
    probabilities = match_probabilities(team_a, team_b)
    random_value = np.random.random()

    if random_value < probabilities["team_a"]:
        return team_a["team"], "regulation"
    if random_value < probabilities["team_a"] + probabilities["draw"]:
        if allow_draw:
            return None, "draw"

        decisive_total = probabilities["team_a"] + probabilities["team_b"]
        decisive_a = (
            probabilities["team_a"] / decisive_total if decisive_total else 0.5
        )
        extra_time_winner = (
            team_a["team"] if np.random.random() < decisive_a else team_b["team"]
        )
        resolution = "extra_time" if np.random.random() < 0.65 else "penalties"
        return extra_time_winner, resolution

    return team_b["team"], "regulation"


def estimate_goals(attacking_team, defending_team):
    """Create simple simulated goals from attack and defense indicators."""
    expected_goals = (
        0.35
        + attacking_team.get("historical_attack", 1.25) * 0.75
        + max(
            0,
            defending_team.get("historical_defense", 1.25)
            - attacking_team.get("historical_attack", 1.25),
        )
        * -0.25
    )
    return np.random.poisson(max(expected_goals, 0.2))
