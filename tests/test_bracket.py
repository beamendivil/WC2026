import itertools
import unittest

import pandas as pd

from src.bracket import (
    CONFIRMED_GROUP_POSITIONS,
    CONFIRMED_KNOCKOUT_WINNERS,
    CONFIRMED_ROUND_OF_32,
)
from src.bracket import CONFIRMED_THIRD_PLACE_QUALIFIERS
from src.bracket import (
    build_round_of_32,
    confirmed_knockout_pairings,
    load_third_place_mapping,
)
from src.data_loader import add_safe_defaults, load_sample_data
from src.features import add_strength_scores
from src.simulator import predict_tournament_bracket, simulate_group_stage
from src.simulator import enforce_confirmed_round_of_32
from src.update_data import load_bundled_world_cup_fixtures


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

    def test_bundled_world_cup_results_feed_group_simulation(self):
        fixtures = load_bundled_world_cup_fixtures()
        colombia_portugal = fixtures.loc[
            (fixtures["team_home"] == "Colombia")
            & (fixtures["team_away"] == "Portugal")
        ].iloc[0]

        self.assertGreaterEqual(len(fixtures), 46)
        self.assertEqual(colombia_portugal["match_status"], "FT")
        self.assertEqual(
            (colombia_portugal["goals_home"], colombia_portugal["goals_away"]),
            (0, 0),
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

    def test_mexico_ecuador_pairing_is_locked(self):
        self.assertEqual(CONFIRMED_ROUND_OF_32[79], ("Mexico", "Ecuador"))

    def test_completed_knockout_winner_advances_in_every_simulation(self):
        expected_winners = {
            73: "Canada",
            74: "Paraguay",
            75: "Morocco",
            76: "Brazil",
            77: "France",
            78: "Norway",
            79: "Mexico",
            82: "Belgium",
            80: "England",
        }
        self.assertEqual(CONFIRMED_KNOCKOUT_WINNERS, expected_winners)

        for _ in range(20):
            pairings, _ = predict_tournament_bracket(sample_teams())
            round_of_16 = pairings.set_index("match")
            self.assertEqual(round_of_16.loc[89, "team_b"], "France")
            self.assertEqual(round_of_16.loc[89, "team_a"], "Paraguay")
            self.assertEqual(round_of_16.loc[90, "team_a"], "Canada")
            self.assertEqual(round_of_16.loc[90, "team_b"], "Morocco")
            self.assertEqual(round_of_16.loc[91, "team_a"], "Brazil")
            self.assertEqual(round_of_16.loc[91, "team_b"], "Norway")
            self.assertEqual(round_of_16.loc[92, "team_a"], "Mexico")
            self.assertEqual(round_of_16.loc[92, "team_b"], "England")
            self.assertEqual(round_of_16.loc[94, "team_b"], "Belgium")

    def test_live_match_is_not_prematurely_locked(self):
        self.assertNotIn(81, CONFIRMED_KNOCKOUT_WINNERS)

    def test_round_of_16_matchups_with_completed_feeders_are_confirmed(self):
        confirmed = confirmed_knockout_pairings()

        self.assertEqual(
            confirmed[90]["teams"], frozenset(("Canada", "Morocco"))
        )
        self.assertEqual(
            confirmed[89]["teams"], frozenset(("Paraguay", "France"))
        )
        self.assertEqual(
            confirmed[91]["teams"], frozenset(("Brazil", "Norway"))
        )
        self.assertEqual(
            confirmed[92]["teams"], frozenset(("Mexico", "England"))
        )
        self.assertNotIn(94, confirmed)

    def test_completed_fixture_automatically_sets_knockout_winner(self):
        fixtures = pd.DataFrame(
            [
                {
                    "team_home": "South Africa",
                    "team_away": "Canada",
                    "match_status": "PEN",
                    "goals_home": 1,
                    "goals_away": 1,
                    "penalty_home": 3,
                    "penalty_away": 4,
                }
            ]
        )

        pairings, _ = predict_tournament_bracket(sample_teams(), fixtures)
        round_of_16 = pairings.set_index("match")
        self.assertEqual(round_of_16.loc[90, "team_a"], "Canada")

    def test_confirmed_pairing_removes_stale_alternatives(self):
        stale = pd.DataFrame(
            [
                ["Round of 32", "Mexico", "Uruguay", 70.0, 700],
                ["Round of 32", "Ecuador", "Belgium", 30.0, 300],
            ],
            columns=[
                "round",
                "team_a",
                "team_b",
                "pairing_probability",
                "simulations",
            ],
        )
        corrected = enforce_confirmed_round_of_32(stale, 1000)
        mexico_matches = corrected.loc[
            (corrected["round"] == "Round of 32")
            & (
                (corrected["team_a"] == "Mexico")
                | (corrected["team_b"] == "Mexico")
            )
        ]

        self.assertEqual(len(mexico_matches), 1)
        self.assertEqual(
            set(mexico_matches.iloc[0][["team_a", "team_b"]]),
            {"Mexico", "Ecuador"},
        )
        self.assertEqual(mexico_matches.iloc[0]["pairing_probability"], 100)

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
