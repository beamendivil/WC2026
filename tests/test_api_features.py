import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from src.api_client import APIFootballClient
from src.update_data import aggregate_player_features
from src.update_data import flatten_odds, flatten_player_stats


class APIFeatureTests(unittest.TestCase):
    @patch("src.api_client.requests.get")
    def test_provider_errors_are_not_silently_treated_as_empty_data(self, get):
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "errors": {"access": "Account unavailable"},
            "response": [],
        }
        get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "Account unavailable"):
            APIFootballClient(api_key="test").teams()

        self.assertEqual(get.call_count, 1)

    @patch("src.api_client.time.sleep")
    @patch("src.api_client.requests.get")
    def test_rate_limit_retry_is_bounded_and_honors_retry_after(
        self, get, sleep
    ):
        limited = Mock(status_code=429, headers={"Retry-After": "2"})
        success = Mock(status_code=200, headers={})
        success.raise_for_status.return_value = None
        success.json.return_value = {"errors": {}, "response": [{"id": 1}]}
        get.side_effect = [limited, success]

        with TemporaryDirectory() as directory:
            result = APIFootballClient(
                api_key="test",
                cache_dir=Path(directory),
                min_request_interval=0,
            )._get("fixtures")

        self.assertEqual(result, [{"id": 1}])
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(2.0)

    @patch("src.api_client.requests.get")
    def test_slow_endpoint_cache_avoids_a_second_network_call(self, get):
        response = Mock(status_code=200, headers={})
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "errors": {},
            "response": [{"name": "Mexico"}],
        }
        get.return_value = response

        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            first = APIFootballClient(
                api_key="test",
                cache_dir=cache_dir,
                min_request_interval=0,
            ).countries()
            second = APIFootballClient(
                api_key="test",
                cache_dir=cache_dir,
                min_request_interval=0,
            ).countries()

        self.assertEqual(first, second)
        self.assertEqual(get.call_count, 1)

    def test_match_winner_odds_are_converted_to_no_vig_probabilities(self):
        odds = flatten_odds(
            [
                {
                    "fixture": {"id": 92},
                    "bookmakers": [
                        {
                            "bets": [
                                {
                                    "name": "Match Winner",
                                    "values": [
                                        {"value": "Home", "odd": "2.00"},
                                        {"value": "Draw", "odd": "3.50"},
                                        {"value": "Away", "odd": "4.00"},
                                    ],
                                }
                            ]
                        }
                    ],
                }
            ]
        ).iloc[0]

        total = (
            odds["market_home_probability"]
            + odds["market_draw_probability"]
            + odds["market_away_probability"]
        )
        self.assertAlmostEqual(total, 1)
        self.assertGreater(
            odds["market_home_probability"], odds["market_away_probability"]
        )

    def test_player_payload_creates_bounded_team_features(self):
        payload = [
            {
                "fixture": {"id": 92},
                "players": [
                    {
                        "team": {"id": 1, "name": "Mexico"},
                        "players": [
                            {
                                "player": {"id": 10, "name": "Forward"},
                                "statistics": [
                                    {
                                        "games": {
                                            "minutes": 90,
                                            "position": "F",
                                            "rating": "8.0",
                                        },
                                        "goals": {"total": 1, "assists": 1},
                                        "shots": {"total": 4, "on": 2},
                                        "passes": {"key": 3},
                                        "tackles": {"total": 1},
                                    }
                                ],
                            },
                            {
                                "player": {"id": 11, "name": "Keeper"},
                                "statistics": [
                                    {
                                        "games": {
                                            "minutes": 90,
                                            "position": "G",
                                            "rating": "7.0",
                                        },
                                        "goals": {"conceded": 1, "saves": 4},
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        ]

        players = flatten_player_stats(payload)
        features = aggregate_player_features(players).iloc[0]

        self.assertEqual(len(players), 2)
        self.assertGreater(features["expected_lineup_score"], 0)
        self.assertGreater(features["goalkeeper_score"], 0)
        self.assertTrue(
            -2 <= features["expected_lineup_score"] <= 2
        )
        self.assertTrue(-1.5 <= features["goalkeeper_score"] <= 1.5)


if __name__ == "__main__":
    unittest.main()
