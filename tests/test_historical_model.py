import unittest

import pandas as pd

from src.data_loader import add_safe_defaults, load_sample_data
from src.features import add_strength_scores
from src.historical_model import HISTORICAL_RESULTS_PATH
from src.historical_model import default_team_state, load_historical_model
from src.historical_model import update_team_state
from src.model import advancement_probability, match_probabilities


class HistoricalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teams = add_strength_scores(
            add_safe_defaults(load_sample_data()),
            ranking_weight=1,
            form_weight=1,
            host_weight=2,
            elo_weight=1,
            xg_weight=1,
            player_weight=0.5,
            context_weight=0.5,
            market_weight=0.5,
        ).set_index("team")

    def test_all_tournament_teams_have_real_history_and_rankings(self):
        self.assertTrue((self.teams["historical_matches"] > 0).all())
        self.assertTrue(self.teams["fifa_points"].notna().all())
        self.assertEqual(set(self.teams["ranking_date"]), {"2026-06-11"})

    def test_current_tournament_snapshot_enriches_team_form(self):
        mexico = self.teams.loc["Mexico"]

        self.assertEqual(mexico["current_matches_played"], 4)
        self.assertEqual(mexico["current_points"], 12)
        self.assertEqual(mexico["current_goal_difference"], 8)
        self.assertEqual(mexico["last_five_form"], "W-W-W-W-W")

    def test_probabilities_are_complete_and_bounded(self):
        argentina = self.teams.loc["Argentina"]
        cabo_verde = self.teams.loc["Cabo Verde"]
        probabilities = match_probabilities(argentina, cabo_verde)

        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertTrue(all(0 <= value <= 1 for value in probabilities.values()))
        self.assertGreater(advancement_probability(argentina, cabo_verde), 0.5)

    def test_chronological_validation_beats_frequency_baseline(self):
        metrics = load_historical_model().metrics
        self.assertGreater(metrics["validation_matches"], 1000)
        self.assertLess(
            metrics["validation_log_loss"], metrics["baseline_log_loss"]
        )

    def test_colombia_portugal_draw_is_recorded(self):
        results = pd.read_csv(HISTORICAL_RESULTS_PATH)
        match = results.loc[
            (results["date"] == "2026-06-27")
            & (results["home_team"] == "Colombia")
            & (results["away_team"] == "Portugal")
        ].iloc[0]

        self.assertEqual((match["home_score"], match["away_score"]), (0, 0))

    def test_goal_updates_limit_single_match_outliers(self):
        updated = update_team_state(
            default_team_state(),
            opponent_elo=1500,
            goals_for=9,
            goals_against=0,
            result=1,
            weight=1.5,
        )

        self.assertLess(updated["attack"], 2)

    def test_host_context_improves_mexico_probability(self):
        mexico = self.teams.loc["Mexico"].copy()
        ecuador = self.teams.loc["Ecuador"].copy()
        hosted = match_probabilities(mexico, ecuador)["team_a"]
        mexico["host_advantage"] = 0
        neutral = match_probabilities(mexico, ecuador)["team_a"]

        self.assertGreater(hosted, neutral)

    def test_probability_is_independent_of_pairing_display_order(self):
        mexico = self.teams.loc["Mexico"]
        ecuador = self.teams.loc["Ecuador"]
        mexico_first = advancement_probability(mexico, ecuador)
        ecuador_first = advancement_probability(ecuador, mexico)

        self.assertAlmostEqual(mexico_first, 1 - ecuador_first)

    def test_live_inputs_adjust_but_do_not_overwhelm_base_probability(self):
        mexico = self.teams.loc["Mexico"].copy()
        ecuador = self.teams.loc["Ecuador"].copy()
        baseline = match_probabilities(mexico, ecuador)["team_a"]
        mexico["market_component"] += 4
        updated = match_probabilities(mexico, ecuador)["team_a"]

        self.assertGreater(updated, baseline)
        self.assertLess(updated - baseline, 0.15)


if __name__ == "__main__":
    unittest.main()
