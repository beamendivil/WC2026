import numpy as np

from src.historical_model import historical_match_probabilities


def win_probability(team_a_strength, team_b_strength):
    """Convert strength difference into Team A win probability."""
    strength_difference = team_a_strength - team_b_strength
    return 1 / (1 + np.exp(-strength_difference / 6))


def _logit(probability):
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def _live_adjustment(team):
    """Return only the timely inputs not already represented by the base model."""
    columns = (
        "xg_component",
        "shot_component",
        "player_component",
        "context_component",
        "market_component",
        "tournament_position_component",
        "match_context_component",
    )
    return sum(float(team.get(column, 0) or 0) for column in columns)


def _apply_live_adjustment(probabilities, team_a, team_b):
    """Apply a deliberately bounded update to the calibrated decisive odds."""
    decisive_total = probabilities["team_a"] + probabilities["team_b"]
    if decisive_total <= 0:
        return probabilities
    decisive_a = probabilities["team_a"] / decisive_total
    adjustment = np.clip(
        0.08 * (_live_adjustment(team_a) - _live_adjustment(team_b)),
        -0.60,
        0.60,
    )
    adjusted_a = 1 / (1 + np.exp(-(_logit(decisive_a) + adjustment)))
    return {
        "team_a": decisive_total * adjusted_a,
        "draw": probabilities["draw"],
        "team_b": decisive_total * (1 - adjusted_a),
    }


def match_probabilities(team_a, team_b):
    """Return regulation probabilities with a safe heuristic fallback."""
    if "historical_elo" in team_a and "historical_elo" in team_b:
        match_site_advantage = float(team_a.get("host_advantage", 0)) - float(
            team_b.get("host_advantage", 0)
        )
        if match_site_advantage < 0:
            reversed_probabilities = historical_match_probabilities(
                team_b, team_a, home_advantage=abs(match_site_advantage)
            )
            probabilities = {
                "team_a": reversed_probabilities["team_b"],
                "draw": reversed_probabilities["draw"],
                "team_b": reversed_probabilities["team_a"],
            }
        else:
            probabilities = historical_match_probabilities(
                team_a, team_b, home_advantage=match_site_advantage
            )
        return _apply_live_adjustment(probabilities, team_a, team_b)

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
    regulation_decisive_a = (
        probabilities["team_a"] / decisive_total if decisive_total else 0.5
    )
    extra_time_a = 1 / (
        1 + np.exp(-0.65 * _logit(regulation_decisive_a))
    )
    goalkeeper_difference = float(team_a.get("goalkeeper_score", 5)) - float(
        team_b.get("goalkeeper_score", 5)
    )
    penalties_a = 1 / (
        1
        + np.exp(
            -(
                0.15 * _logit(regulation_decisive_a)
                + 0.10 * np.clip(goalkeeper_difference, -3, 3)
            )
        )
    )
    tiebreak_a = 0.65 * extra_time_a + 0.35 * penalties_a
    return probabilities["team_a"] + probabilities["draw"] * tiebreak_a


def predict_match_winner(team_a, team_b, allow_draw=False):
    """Simulate one match and return the result."""
    probabilities = match_probabilities(team_a, team_b)
    random_value = np.random.random()

    if random_value < probabilities["team_a"]:
        return team_a["team"], "regulation"
    if random_value < probabilities["team_a"] + probabilities["draw"]:
        if allow_draw:
            return None, "draw"

        probability_a = advancement_probability(team_a, team_b)
        draw_share_a = np.clip(
            (probability_a - probabilities["team_a"]) / probabilities["draw"],
            0,
            1,
        )
        winner = team_a["team"] if np.random.random() < draw_share_a else team_b["team"]
        resolution = "extra_time" if np.random.random() < 0.65 else "penalties"
        return winner, resolution

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
