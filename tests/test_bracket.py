import itertools
import unittest

import pandas as pd

from src.bracket import build_round_of_32, load_third_place_mapping
from src.data_loader import add_safe_defaults, load_sample_data
from src.features import add_strength_scores
from src.simulator import simulate_group_stage


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
                ["United States", "Canada", "FT", 1, 0],
                ["United States", "Mexico", "FT", 1, 0],
                ["United States", "New Zealand", "FT", 1, 0],
                ["Canada", "Mexico", "FT", 1, 0],
                ["Canada", "New Zealand", "FT", 1, 0],
                ["Mexico", "New Zealand", "FT", 1, 0],
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
            group_a[["team", "points"]].values.tolist(),
            [["United States", 9], ["Canada", 6]],
        )


if __name__ == "__main__":
    unittest.main()
