import numpy as np


def win_probability(team_a_strength, team_b_strength):
    """Convert strength difference into Team A win probability."""
    strength_difference = team_a_strength - team_b_strength
    return 1 / (1 + np.exp(-strength_difference / 6))


def predict_match_winner(team_a, team_b, allow_draw=False):
    """Simulate one match and return the result."""
    prob_a = win_probability(team_a["strength_score"], team_b["strength_score"])

    if allow_draw:
        draw_chance = max(0.12, 0.28 - abs(prob_a - 0.5) * 0.35)
        decisive_chance = 1 - draw_chance
        adjusted_prob_a = prob_a * decisive_chance
        random_value = np.random.random()

        if random_value < adjusted_prob_a:
            return team_a["team"], "win"
        if random_value < adjusted_prob_a + draw_chance:
            return None, "draw"
        return team_b["team"], "win"

    winner = team_a["team"] if np.random.random() < prob_a else team_b["team"]
    return winner, "win"


def estimate_goals(attacking_team, defending_team):
    """Create simple simulated goals from attack and defense indicators."""
    expected_goals = (
        0.7
        + attacking_team["attack_score"] / 7
        + attacking_team["strength_score"] / 35
        - defending_team["defense_score"] / 12
    )
    return np.random.poisson(max(expected_goals, 0.2))
