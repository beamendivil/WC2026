from dataclasses import dataclass
from functools import lru_cache
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORICAL_RESULTS_PATH = DATA_DIR / "historical_results.csv"
FIFA_RANKINGS_PATH = DATA_DIR / "fifa_rankings.csv"
MODEL_START_DATE = pd.Timestamp("2005-01-01")
VALIDATION_START_DATE = pd.Timestamp("2025-01-01")
CALIBRATION_START_DATE = pd.Timestamp("2023-01-01")
FEATURE_COLUMNS = [
    "elo_difference",
    "form_difference",
    "attack_difference",
    "defense_difference",
    "goal_environment",
    "home_advantage",
]

TEAM_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cabo Verde",
    "Congo DR": "Congo DR",
    "Curaçao": "Curacao",
    "Czech Republic": "Czechia",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "DR Congo": "Congo DR",
    "Iran": "Iran",
    "IR Iran": "Iran",
    "Ivory Coast": "Cote d'Ivoire",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Türkiye": "Turkiye",
    "Turkey": "Turkiye",
    "USA": "United States",
}


def canonical_team_name(team):
    """Normalize provider-specific national-team names."""
    name = str(team).strip()
    return TEAM_ALIASES.get(name, name)


def default_team_state():
    return {
        "elo": 1500.0,
        "form": 0.5,
        "attack": 1.25,
        "defense": 1.25,
        "last_date": None,
        "matches": 0,
    }


def decayed_state(state, match_date):
    """Decay short-term indicators toward neutral between international windows."""
    if state["last_date"] is None:
        return state.copy()
    days = max((match_date - state["last_date"]).days, 0)
    retention = exp(-log(2) * days / 540)
    updated = state.copy()
    updated["form"] = 0.5 + (state["form"] - 0.5) * retention
    updated["attack"] = 1.25 + (state["attack"] - 1.25) * retention
    updated["defense"] = 1.25 + (state["defense"] - 1.25) * retention
    return updated


def tournament_weight(tournament):
    text = str(tournament).lower()
    if "fifa world cup" in text and "qualification" not in text:
        return 1.5
    if "qualification" in text or "qualifier" in text:
        return 1.2
    if "friendly" in text:
        return 0.65
    return 1.0


def update_team_state(state, opponent_elo, goals_for, goals_against, result, weight):
    """Update recent form and scoring rates with opponent-quality adjustment."""
    alpha = min(0.32, 0.16 * weight)
    quality_adjustment = np.clip((opponent_elo - 1500) / 1200, -0.2, 0.2)
    adjusted_result = np.clip(result + quality_adjustment, 0, 1)
    adjusted_goals_for = np.clip(goals_for + quality_adjustment, 0, 4)
    adjusted_goals_against = np.clip(goals_against - quality_adjustment, 0, 4)
    updated = state.copy()
    updated["form"] = (1 - alpha) * state["form"] + alpha * adjusted_result
    updated["attack"] = (
        (1 - alpha) * state["attack"] + alpha * adjusted_goals_for
    )
    updated["defense"] = (
        (1 - alpha) * state["defense"] + alpha * adjusted_goals_against
    )
    updated["matches"] = state["matches"] + 1
    return updated


def build_training_data(results):
    """Create leakage-free pre-match features and final team snapshots."""
    states = {}
    feature_rows = []

    completed = results.dropna(subset=["home_score", "away_score"]).copy()
    completed["date"] = pd.to_datetime(completed["date"], errors="coerce")
    completed = completed.dropna(subset=["date"]).sort_values("date")

    for match in completed.itertuples(index=False):
        home_name = canonical_team_name(match.home_team)
        away_name = canonical_team_name(match.away_team)
        home = decayed_state(states.get(home_name, default_team_state()), match.date)
        away = decayed_state(states.get(away_name, default_team_state()), match.date)
        neutral = str(match.neutral).strip().lower() in {"true", "1"}
        home_advantage = 0 if neutral else 1

        if match.date >= MODEL_START_DATE:
            result = (
                "home"
                if match.home_score > match.away_score
                else "away"
                if match.away_score > match.home_score
                else "draw"
            )
            feature_rows.append(
                {
                    "date": match.date,
                    "elo_difference": home["elo"] - away["elo"],
                    "form_difference": home["form"] - away["form"],
                    "attack_difference": home["attack"] - away["attack"],
                    "defense_difference": away["defense"] - home["defense"],
                    "goal_environment": (
                        home["attack"]
                        + away["attack"]
                        + home["defense"]
                        + away["defense"]
                    )
                    / 4,
                    "home_advantage": home_advantage,
                    "result": result,
                    "sample_weight": tournament_weight(match.tournament),
                }
            )

        expected_home = 1 / (
            1 + 10 ** (-(home["elo"] - away["elo"] + home_advantage * 70) / 400)
        )
        actual_home = (
            1.0
            if match.home_score > match.away_score
            else 0.0
            if match.home_score < match.away_score
            else 0.5
        )
        k_factor = 28 * tournament_weight(match.tournament)
        elo_change = k_factor * (actual_home - expected_home)
        home["elo"] += elo_change
        away["elo"] -= elo_change

        home = update_team_state(
            home,
            away["elo"],
            float(match.home_score),
            float(match.away_score),
            actual_home,
            tournament_weight(match.tournament),
        )
        away = update_team_state(
            away,
            home["elo"],
            float(match.away_score),
            float(match.home_score),
            1 - actual_home,
            tournament_weight(match.tournament),
        )
        home["last_date"] = match.date
        away["last_date"] = match.date
        states[home_name] = home
        states[away_name] = away

    return pd.DataFrame(feature_rows), states


@dataclass
class HistoricalModelBundle:
    model: object
    states: dict
    metrics: dict
    last_match_date: str
    calibration_temperature: float


def apply_temperature(probabilities, temperature):
    """Calibrate class probabilities with a fitted temperature."""
    logits = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


@lru_cache(maxsize=1)
def load_historical_model():
    """Train and cache the historical three-outcome probability model."""
    results = pd.read_csv(HISTORICAL_RESULTS_PATH)
    features, states = build_training_data(results)
    base_training = features.loc[features["date"] < CALIBRATION_START_DATE]
    calibration = features.loc[
        (features["date"] >= CALIBRATION_START_DATE)
        & (features["date"] < VALIDATION_START_DATE)
    ]
    validation = features.loc[features["date"] >= VALIDATION_START_DATE]

    calibration_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs"),
    )
    calibration_model.fit(
        base_training[FEATURE_COLUMNS],
        base_training["result"],
        logisticregression__sample_weight=base_training["sample_weight"],
    )
    raw_calibration = calibration_model.predict_proba(
        calibration[FEATURE_COLUMNS]
    )
    class_order = calibration_model.classes_
    calibration_targets = pd.Categorical(
        calibration["result"], categories=class_order
    ).codes

    def calibration_loss(temperature):
        probabilities = apply_temperature(raw_calibration, temperature)
        return -np.mean(
            np.log(
                probabilities[
                    np.arange(len(calibration_targets)), calibration_targets
                ]
            )
        )

    calibration_temperature = minimize_scalar(
        calibration_loss, bounds=(0.5, 2.5), method="bounded"
    ).x

    model = calibration_model

    metrics = {
        "training_matches": len(base_training),
        "calibration_matches": len(calibration),
        "validation_matches": len(validation),
        "calibration_temperature": calibration_temperature,
    }
    if not validation.empty:
        probabilities = apply_temperature(
            model.predict_proba(validation[FEATURE_COLUMNS]),
            calibration_temperature,
        )
        predictions = model.classes_[np.argmax(probabilities, axis=1)]
        training_class_rates = (
            base_training["result"].value_counts(normalize=True)
            .reindex(model.classes_)
            .values
        )
        baseline_probabilities = np.tile(training_class_rates, (len(validation), 1))
        metrics.update(
            {
                "validation_accuracy": accuracy_score(
                    validation["result"], predictions
                ),
                "validation_log_loss": log_loss(
                    validation["result"], probabilities, labels=model.classes_
                ),
                "baseline_log_loss": log_loss(
                    validation["result"],
                    baseline_probabilities,
                    labels=model.classes_,
                ),
            }
        )

    last_match_date = features["date"].max().date().isoformat()
    return HistoricalModelBundle(
        model,
        states,
        metrics,
        last_match_date,
        calibration_temperature,
    )


def load_fifa_rankings():
    rankings = pd.read_csv(FIFA_RANKINGS_PATH)
    rankings["team"] = rankings["team"].map(canonical_team_name)
    return rankings.drop_duplicates("team").set_index("team")


def enrich_with_historical_features(teams):
    """Attach current historical snapshots and official FIFA rankings."""
    enriched = teams.copy()
    bundle = load_historical_model()
    rankings = load_fifa_rankings()

    for index, row in enriched.iterrows():
        team = canonical_team_name(row["team"])
        state = bundle.states.get(team, default_team_state())
        enriched.loc[index, "historical_elo"] = state["elo"]
        enriched.loc[index, "historical_form"] = state["form"] * 10
        enriched.loc[index, "historical_attack"] = state["attack"]
        enriched.loc[index, "historical_defense"] = state["defense"]
        enriched.loc[index, "historical_matches"] = state["matches"]
        if team in rankings.index:
            enriched.loc[index, "fifa_rank"] = rankings.loc[team, "fifa_rank"]
            enriched.loc[index, "fifa_points"] = rankings.loc[team, "fifa_points"]
            enriched.loc[index, "ranking_date"] = rankings.loc[team, "ranking_date"]

    return enriched


def historical_match_probabilities(team_a, team_b):
    """Predict regulation home/draw/away probabilities for a neutral match."""
    bundle = load_historical_model()
    features = pd.DataFrame(
        [
            {
                "elo_difference": team_a.get("historical_elo", 1500)
                - team_b.get("historical_elo", 1500),
                "form_difference": team_a.get("historical_form", 5) / 10
                - team_b.get("historical_form", 5) / 10,
                "attack_difference": team_a.get("historical_attack", 1.25)
                - team_b.get("historical_attack", 1.25),
                "defense_difference": team_b.get("historical_defense", 1.25)
                - team_a.get("historical_defense", 1.25),
                "goal_environment": (
                    team_a.get("historical_attack", 1.25)
                    + team_b.get("historical_attack", 1.25)
                    + team_a.get("historical_defense", 1.25)
                    + team_b.get("historical_defense", 1.25)
                )
                / 4,
                "home_advantage": 0,
            }
        ]
    )
    calibrated = apply_temperature(
        bundle.model.predict_proba(features), bundle.calibration_temperature
    )[0]
    model_probabilities = dict(zip(bundle.model.classes_, calibrated))

    fifa_a = float(team_a.get("fifa_points", 1500))
    fifa_b = float(team_b.get("fifa_points", 1500))
    fifa_decisive_a = 1 / (1 + exp(-(fifa_a - fifa_b) / 170))
    draw_probability = model_probabilities["draw"]
    model_decisive_total = model_probabilities["home"] + model_probabilities["away"]
    model_decisive_a = (
        model_probabilities["home"] / model_decisive_total
        if model_decisive_total
        else 0.5
    )
    decisive_a = 0.72 * model_decisive_a + 0.28 * fifa_decisive_a
    return {
        "team_a": (1 - draw_probability) * decisive_a,
        "draw": draw_probability,
        "team_b": (1 - draw_probability) * (1 - decisive_a),
    }
