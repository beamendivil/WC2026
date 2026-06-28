import itertools
import unittest

import pandas as pd

from src.bracket import CONFIRMED_GROUP_POSITIONS, CONFIRMED_ROUND_OF_32
from src.bracket import CONFIRMED_THIRD_PLACE_QUALIFIERS
from src.bracket import build_round_of_32, load_third_place_mapping
from src.data_loader import add_safe_defaults, load_sample_data
from src.features import add_strength_scores
from src.simulator import predict_tournament_bracket, simulate_group_stage


def sample_teams():
    return add_strength_scores(
        add_safe_defaults(load_sample_data()),
        ranking_weight=1,
        form_weight=1,
        host_weight=2,
        elo_weight=1,
        xg_weight=1,
        player_weight=0.5,
        context_weight=0.5,
        market_weight=0.5,
    )


class OfficialBracketTests(unittest.TestCase):
    def test_annex_c_contains_every_eight_group_combination(self):
        mapping = load_third_place_mapping()
        expected = {
            "".join(groups)
            for groups in itertools.combinations("ABCDEFGHIJKL", 8)
        }

        self.assertEqual(set(mapping.index), expected)
        self.assertEqual(len(mapping), 495)
        for qualifying_groups, row in mapping.iterrows():
            self.assertEqual(set(row), set(qualifying_groups))

    def test_round_of_32_uses_official_match_slots(self):
        qualified = simulate_group_stage(sample_teams())
        matches = {
            match_number: (team_a, team_b)
            for match_number, team_a, team_b in build_round_of_32(qualified)
        }

        self.assertEqual(set(matches), set(range(73, 89)))
        self.assertEqual(matches[73][0]["group_position"], 2)
        self.assertEqual(matches[73][0]["group"], "A")
        self.assertEqual(matches[73][1]["group_position"], 2)
        self.assertEqual(matches[73][1]["group"], "B")
        self.assertEqual(matches[74][0]["group_position"], 1)
        self.assertEqual(matches[74][0]["group"], "E")
        self.assertEqual(matches[74][1]["group_position"], 3)
        self.assertIn(matches[74][1]["group"], set("ABCDF"))

    def test_completed_group_results_are_not_resimulated(self):
        fixtures = pd.DataFrame(
            [
                ["Mexico", "South Africa", "FT", 1, 0],
                ["Mexico", "South Korea", "FT", 1, 0],
                ["Mexico", "Czechia", "FT", 1, 0],
                ["South Africa", "South Korea", "FT", 1, 0],
                ["South Africa", "Czechia", "FT", 1, 0],
                ["South Korea", "Czechia", "FT", 1, 0],
            ],
            columns=[
                "team_home",
                "team_away",
                "match_status",
                "goals_home",
                "goals_away",
            ],
        )

        qualified = simulate_group_stage(sample_teams(), fixtures)
        group_a = qualified.loc[qualified["group"] == "A"].sort_values(
            "group_position"
        )

        self.assertEqual(
            group_a[["team", "points"]].head(2).values.tolist(),
            [["Mexico", 9], ["South Africa", 6]],
        )

    def test_confirmed_round_of_32_pairings_override_simulation(self):
        pairings, _ = predict_tournament_bracket(sample_teams())
        round_of_32 = pairings.loc[pairings["round"] == "Round of 32"].set_index(
            "match"
        )

        for match_number, expected_teams in CONFIRMED_ROUND_OF_32.items():
            actual_teams = (
                round_of_32.loc[match_number, "team_a"],
                round_of_32.loc[match_number, "team_b"],
            )
            self.assertEqual(actual_teams, expected_teams)

    def test_confirmed_group_positions_condition_simulation(self):
        qualified = simulate_group_stage(sample_teams())
        positions = qualified.set_index(["group", "group_position"])["team"]

        for group, confirmed_positions in CONFIRMED_GROUP_POSITIONS.items():
            for position, team in confirmed_positions.items():
                if position <= 2 or team in CONFIRMED_THIRD_PLACE_QUALIFIERS:
                    self.assertEqual(positions.loc[(group, position)], team)

        qualified_thirds = set(
            qualified.loc[qualified["group_position"] == 3, "team"]
        )
        self.assertTrue(
            CONFIRMED_THIRD_PLACE_QUALIFIERS.issubset(qualified_thirds)
        )


if __name__ == "__main__":
    unittest.main()
