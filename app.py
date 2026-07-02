from datetime import datetime
import hashlib
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.api_config import LATEST_PAIRING_PREDICTIONS_CSV
from src.bracket import CONFIRMED_KNOCKOUT_WINNERS, CONFIRMED_ROUND_OF_32
from src.bracket import confirmed_knockout_pairings
from src.data_loader import add_safe_defaults, load_sample_data, validate_team_data
from src.explanations import find_biggest_factor
from src.features import add_strength_scores
from src.historical_model import load_historical_model
from src.model import advancement_probability, match_probabilities
from src.api_config import has_api_key
from src.simulator import (
    enforce_confirmed_round_of_32,
    run_pairing_simulations,
    run_simulations,
)
from src.update_data import load_available_fixtures, load_cached_model_data
from src.update_data import load_live_model_data


def cached_sample_data():
    """Load the small fallback CSV so file updates are reflected immediately."""
    return load_sample_data()


def render_soccer_loading_icon():
    """Replace Streamlit's activity-cycle loader with a soccer animation."""
    st.markdown(
        """
        <style>
        [data-testid="stStatusWidget"] > div {
            visibility: hidden;
        }
        [data-testid="stStatusWidget"]::after {
            animation: soccer-roll 0.9s ease-in-out infinite;
            color: currentColor;
            content: "sports_soccer";
            display: inline-block;
            font-family: "Material Symbols Rounded", "Material Symbols Outlined";
            font-size: 1.5rem;
            font-variation-settings:
                "FILL" 0,
                "wght" 500,
                "GRAD" 0,
                "opsz" 24;
            line-height: 1;
            transform-origin: center;
            visibility: visible;
            white-space: nowrap;
        }
        @keyframes soccer-roll {
            0% {
                transform: translateX(-4px) rotate(0deg);
            }
            50% {
                transform: translateX(4px) rotate(180deg);
            }
            100% {
                transform: translateX(-4px) rotate(360deg);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Create sidebar controls and return selected settings."""
    st.sidebar.header("Simulation Controls")
    number_of_simulations = st.sidebar.slider(
        "Number of simulations", min_value=100, max_value=5000, value=1000, step=100
    )
    refresh_api = st.sidebar.button(
        "Refresh API data",
        icon=":material/sync:",
        help="Contact the provider and replace the local API snapshot.",
    )

    return {
        "number_of_simulations": number_of_simulations,
        "refresh_api": refresh_api,
        "ranking_weight": 1.0,
        "form_weight": 1.0,
        "host_weight": 2.0,
        "elo_weight": 1.0,
        "xg_weight": 1.0,
        "player_weight": 0.5,
        "context_weight": 0.5,
        "market_weight": 0.5,
    }


def load_selected_data(refresh_api=False):
    """Load API data automatically, falling back to the sample CSV."""
    sample_teams = cached_sample_data()
    if not has_api_key():
        return sample_teams, "Sample CSV fallback", False

    live_teams, _ = (
        load_live_model_data(force=True)
        if refresh_api
        else load_cached_model_data()
    )
    if live_teams.empty:
        return sample_teams, "Sample CSV fallback", False

    live_teams = add_safe_defaults(live_teams)
    live_roster = set(live_teams["team"].astype(str))
    official_roster = set(sample_teams["team"].astype(str))
    if validate_team_data(live_teams) or live_roster != official_roster:
        return sample_teams, "API + Sample CSV fallback", True

    return live_teams, "Live API with sample defaults", True


def render_api_status(api_connected):
    """Show a strong connection indicator without exposing credentials."""
    state_class = "connected" if api_connected else "offline"
    state_label = "API CONNECTED" if api_connected else "API OFFLINE"
    st.markdown(
        f"""
        <div class="api-status {state_class}" role="status">
            <span class="api-light" aria-hidden="true"></span>
            <span>{state_label}</span>
        </div>
        <style>
        .api-status {{
            align-items: center;
            color: #8b949e;
            display: flex;
            font-size: 1rem;
            font-weight: 800;
            gap: 0.5rem;
            letter-spacing: 0;
            margin: 0.25rem 0 0.75rem;
        }}
        .api-light {{
            background: #5f6872;
            border: 1px solid #737d87;
            border-radius: 50%;
            display: inline-block;
            height: 0.7rem;
            width: 0.7rem;
        }}
        .api-status.connected {{
            color: #6fffb0;
        }}
        .api-status.connected .api-light {{
            background: #45f58a;
            border-color: #8affb7;
            box-shadow:
                0 0 4px rgba(69, 245, 138, 1),
                0 0 12px rgba(34, 197, 94, 0.9);
        }}
        .api-status.connected span:last-child {{
            text-shadow:
                0 0 4px rgba(111, 255, 176, 0.95),
                0 0 12px rgba(34, 197, 94, 0.8);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def prepare_team_strength(raw_teams, settings):
    """Apply defaults, validate, and calculate team strength."""
    raw_teams = add_safe_defaults(raw_teams)
    eliminated_teams = {
        team
        for match, pairing in CONFIRMED_ROUND_OF_32.items()
        if match in CONFIRMED_KNOCKOUT_WINNERS
        for team in pairing
        if team != CONFIRMED_KNOCKOUT_WINNERS[match]
    }
    raw_teams.loc[
        raw_teams["team"].isin(eliminated_teams), "eliminated"
    ] = True
    validation_errors = validate_team_data(raw_teams)
    if validation_errors:
        return raw_teams, None, validation_errors

    teams = add_strength_scores(
        raw_teams,
        settings["ranking_weight"],
        settings["form_weight"],
        settings["host_weight"],
        settings["elo_weight"],
        settings["xg_weight"],
        settings["player_weight"],
        settings["context_weight"],
        settings["market_weight"],
    )
    return raw_teams, teams, []


def render_model_quality(teams):
    """Render compact evidence and validation metrics for the trained model."""
    bundle = load_historical_model()
    metrics = bundle.metrics
    ranking_date = teams["ranking_date"].dropna().max()
    with st.expander("Model evidence"):
        columns = st.columns(4)
        columns[0].metric("Training matches", f"{metrics['training_matches']:,}")
        columns[1].metric(
            "Validation accuracy", f"{metrics['validation_accuracy']:.1%}"
        )
        columns[2].metric(
            "Validation log loss",
            f"{metrics['validation_log_loss']:.3f}",
            delta=(
                f"{metrics['baseline_log_loss'] - metrics['validation_log_loss']:.3f}"
                " better"
            ),
        )
        columns[3].metric("FIFA ranking date", str(ranking_date))
        st.caption(
            f"Validated on {metrics['validation_matches']:,} later matches. "
            f"Results data through {bundle.last_match_date}."
        )


def render_team_table(teams):
    """Render a compact view of the inputs that drive predictions."""
    st.subheader("Team Snapshot")
    boost_factors = {
        "rank_score": "FIFA ranking",
        "elo_score": "Historical Elo",
        "recent_form_component": "Recent form",
        "attack_score": "Attack",
        "defense_score": "Defense",
        "host_advantage_component": "Host advantage",
        "confirmed_group_finish_component": "Confirmed group finish",
    }
    columns = [
        "team",
        "group",
        "fifa_rank",
        "fifa_points",
        "historical_elo",
        "historical_form",
        "strength_score",
    ]
    snapshot = teams[[column for column in columns if column in teams.columns]].copy()
    snapshot["biggest_boost"] = teams.apply(
        lambda row: find_biggest_factor(row, boost_factors), axis=1
    )
    snapshot = snapshot.sort_values("strength_score", ascending=False)
    snapshot = snapshot.rename(
        columns={
            "team": "Team",
            "group": "Group",
            "fifa_rank": "Current FIFA ranking",
            "fifa_points": "FIFA points",
            "historical_elo": "Historical Elo",
            "historical_form": "Recent form",
            "strength_score": "Model strength",
            "biggest_boost": "Biggest boost",
        }
    )
    numeric_columns = snapshot.select_dtypes(include="number").columns
    snapshot[numeric_columns] = snapshot[numeric_columns].round(1)
    st.dataframe(
        snapshot,
        width="stretch",
        hide_index=True,
        column_config={
            "Historical Elo": st.column_config.NumberColumn(
                help=(
                    "A team-strength rating learned from historical international "
                    "results. It updates after each match and accounts for opponent "
                    "quality, result, and recency."
                ),
                format="%.1f",
            ),
            "Current FIFA ranking": st.column_config.NumberColumn(format="%d"),
        },
    )


def render_match_predictor(teams):
    """Render a simple head-to-head match predictor."""
    st.subheader("Match Predictor")
    col1, col2 = st.columns(2)
    eligible_teams = (
        teams.loc[~teams["eliminated"]]
        if "eliminated" in teams.columns
        else teams
    )
    team_names = eligible_teams["team"].sort_values().tolist()
    team_a_name = col1.selectbox("Team A", team_names, index=0)
    team_b_name = col2.selectbox("Team B", team_names, index=1)

    if team_a_name == team_b_name:
        st.info("Choose two different teams to predict a match.")
        return

    team_a = eligible_teams.loc[eligible_teams["team"] == team_a_name].iloc[0]
    team_b = eligible_teams.loc[eligible_teams["team"] == team_b_name].iloc[0]
    probabilities = match_probabilities(team_a, team_b)
    probability_a = advancement_probability(team_a, team_b)
    predicted_winner = team_a_name if probability_a >= 0.5 else team_b_name

    st.metric("Predicted winner", predicted_winner)
    st.write(
        f"Regulation: {team_a_name} {probabilities['team_a']:.1%} | "
        f"Draw {probabilities['draw']:.1%} | "
        f"{team_b_name} {probabilities['team_b']:.1%}"
    )
    st.caption(
        f"Knockout advancement: {team_a_name} {probability_a:.1%} | "
        f"{team_b_name} {(1 - probability_a):.1%}"
    )


def select_unique_pairings(round_probabilities, match_limit):
    """Select the highest-probability pairings without repeating a team."""
    selected_rows = []
    selected_teams = set()

    for _, pairing in round_probabilities.sort_values(
        "pairing_probability", ascending=False
    ).iterrows():
        team_a = pairing["team_a"]
        team_b = pairing["team_b"]
        if team_a in selected_teams or team_b in selected_teams:
            continue
        selected_rows.append(pairing)
        selected_teams.update([team_a, team_b])
        if len(selected_rows) == match_limit:
            break

    return pd.DataFrame(selected_rows, columns=round_probabilities.columns)


def load_pairing_snapshot(fixtures):
    """Load the latest saved matchup probabilities."""
    if not LATEST_PAIRING_PREDICTIONS_CSV.exists():
        return pd.DataFrame()
    try:
        snapshot = pd.read_csv(LATEST_PAIRING_PREDICTIONS_CSV)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if (
        "bracket_signature" not in snapshot
        or snapshot["bracket_signature"].iloc[0] != bracket_signature(fixtures)
    ):
        return pd.DataFrame()
    return snapshot


def bracket_signature(fixtures=None):
    """Fingerprint confirmed fixtures and completed scores."""
    confirmed = sorted(
        (match, team_a, team_b)
        for match, (team_a, team_b) in CONFIRMED_ROUND_OF_32.items()
    )
    winners = sorted(CONFIRMED_KNOCKOUT_WINNERS.items())
    completed = []
    if fixtures is not None and not fixtures.empty:
        required = {
            "team_home",
            "team_away",
            "match_status",
            "goals_home",
            "goals_away",
        }
        if required.issubset(fixtures.columns):
            completed_rows = fixtures.loc[
                fixtures["match_status"].isin({"FT", "AET", "PEN"})
            ]
            completed = sorted(
                (
                    str(row.team_home),
                    str(row.team_away),
                    str(row.goals_home),
                    str(row.goals_away),
                )
                for row in completed_rows.itertuples(index=False)
            )
    payload = json.dumps(
        {"confirmed": confirmed, "winners": winners, "completed": completed},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def save_pairing_snapshot(pairing_probabilities, number_of_simulations, fixtures=None):
    """Persist matchup probabilities for immediate display on future visits."""
    snapshot = pairing_probabilities.copy()
    snapshot["simulation_runs"] = number_of_simulations
    snapshot["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    snapshot["bracket_signature"] = bracket_signature(fixtures)
    LATEST_PAIRING_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(LATEST_PAIRING_PREDICTIONS_CSV, index=False)
    return snapshot


def render_projected_pairings(teams, fixtures, number_of_simulations):
    """Render an official bracket projection and pairing probabilities."""
    st.subheader("Knockout Pairings")
    teams_by_name = teams.set_index("team")

    current_signature = bracket_signature(fixtures)
    session_predictions = st.session_state.get("pairing_probabilities")
    session_is_current = (
        session_predictions is not None
        and not session_predictions.empty
        and "bracket_signature" in session_predictions
        and session_predictions["bracket_signature"].iloc[0] == current_signature
    )
    if not session_is_current:
        st.session_state["pairing_probabilities"] = load_pairing_snapshot(fixtures)

    if st.button("Refresh matchup predictions", icon=":material/sync:"):
        progress_bar = st.progress(0)
        pairing_probabilities = run_pairing_simulations(
            teams,
            number_of_simulations,
            fixtures=fixtures,
            progress_callback=progress_bar.progress,
        )
        progress_bar.empty()
        st.session_state["pairing_probabilities"] = save_pairing_snapshot(
            pairing_probabilities, number_of_simulations, fixtures
        )

    pairing_probabilities = st.session_state["pairing_probabilities"]
    if pairing_probabilities.empty:
        st.info("No saved matchup forecast is available yet.")
        return
    saved_runs = (
        int(pairing_probabilities["simulation_runs"].iloc[0])
        if "simulation_runs" in pairing_probabilities
        else number_of_simulations
    )
    pairing_probabilities = enforce_confirmed_round_of_32(
        pairing_probabilities, saved_runs
    )
    st.session_state["pairing_probabilities"] = pairing_probabilities

    ordered_rounds = [
        "Round of 32",
        "Round of 16",
        "Quarterfinals",
        "Semifinals",
        "Final",
    ]
    available_rounds = set(pairing_probabilities["round"])
    round_names = [name for name in ordered_rounds if name in available_rounds]
    match_limits = {
        "Round of 32": 16,
        "Round of 16": 8,
        "Quarterfinals": 4,
        "Semifinals": 2,
        "Final": 1,
    }
    tabs = st.tabs(round_names)
    for tab, round_name in zip(tabs, round_names):
        round_probabilities = pairing_probabilities.loc[
            pairing_probabilities["round"] == round_name
        ].copy()
        unique_pairings = select_unique_pairings(
            round_probabilities, match_limits[round_name]
        )
        round_matches = unique_pairings[
            [
                "team_a",
                "team_b",
                "pairing_probability",
                "simulations",
            ]
        ].copy()
        outcome_rows = []
        completed_winners = {
            frozenset(CONFIRMED_ROUND_OF_32[match]): winner
            for match, winner in CONFIRMED_KNOCKOUT_WINNERS.items()
            if match in CONFIRMED_ROUND_OF_32
        }
        confirmed_matchups = {
            pairing["teams"]
            for pairing in confirmed_knockout_pairings().values()
            if pairing["round"] == round_name
        }
        for pairing in round_matches.itertuples(index=False):
            team_a = teams_by_name.loc[pairing.team_a]
            team_b = teams_by_name.loc[pairing.team_b]
            completed_winner = completed_winners.get(
                frozenset((pairing.team_a, pairing.team_b))
            )
            if round_name == "Round of 32" and completed_winner:
                winner = completed_winner
                eliminated = (
                    pairing.team_b
                    if winner == pairing.team_a
                    else pairing.team_a
                )
                winner_probability = 1.0
                status = "Completed"
            else:
                probability_a = advancement_probability(team_a, team_b)
                if probability_a >= 0.5:
                    winner = pairing.team_a
                    eliminated = pairing.team_b
                    winner_probability = probability_a
                else:
                    winner = pairing.team_b
                    eliminated = pairing.team_a
                    winner_probability = 1 - probability_a
                status = (
                    "Confirmed matchup"
                    if frozenset((pairing.team_a, pairing.team_b))
                    in confirmed_matchups
                    else "Projected"
                )
            outcome_rows.append(
                {
                    "Status": status,
                    "Winner": winner,
                    "Eliminated": eliminated,
                    "Winner chance %": round(winner_probability * 100, 1),
                }
            )
        round_matches = pd.concat(
            [round_matches.reset_index(drop=True), pd.DataFrame(outcome_rows)],
            axis=1,
        )
        round_matches.columns = [
            "Team A",
            "Team B",
            "Pairing probability %",
            "Simulations",
            "Status",
            "Winner",
            "Eliminated",
            "Winner chance %",
        ]
        round_matches["Pairing probability %"] = round_matches[
            "Pairing probability %"
        ].round(1)
        tab.dataframe(round_matches, width="stretch", hide_index=True)

    simulation_runs = saved_runs
    generated_at = (
        pairing_probabilities["generated_at"].iloc[0]
        if "generated_at" in pairing_probabilities
        else "previously"
    )
    st.caption(
        f"Saved {generated_at} from {simulation_runs:,} simulations. "
        "Completed fixtures are held fixed."
    )


def render_simulation_results(teams, number_of_simulations, fixtures=None):
    """Run Monte Carlo simulations and render the result tables and charts."""
    st.subheader("Tournament Simulation")
    if not st.button("Run simulations", type="primary"):
        return

    progress_bar = st.progress(0)
    results = run_simulations(
        teams,
        number_of_simulations,
        fixtures=fixtures,
        progress_callback=progress_bar.progress,
    )
    progress_bar.empty()
    top_10 = results.head(10)

    winner_chart = px.bar(
        top_10.sort_values("champion_probability"),
        x="champion_probability",
        y="team",
        orientation="h",
        labels={
            "champion_probability": "Champion probability (%)",
            "team": "Team",
        },
        title="Top 10 Most Likely Winners",
    )
    winner_chart.update_layout(yaxis_title="", xaxis_ticksuffix="%")
    st.plotly_chart(winner_chart)
    with st.expander("Full tournament probabilities"):
        st.dataframe(results, width="stretch", hide_index=True)


def main():
    st.set_page_config(
        page_title="World Cup 2026 Winner Predictor",
        page_icon=":trophy:",
        layout="wide",
    )
    render_soccer_loading_icon()

    st.title("World Cup 2026 Winner Predictor")
    settings = render_sidebar()
    raw_teams, active_source, api_connected = load_selected_data(
        refresh_api=settings["refresh_api"]
    )
    render_api_status(api_connected)
    st.caption(f"Active data source: {active_source}")
    raw_teams, teams, validation_errors = prepare_team_strength(raw_teams, settings)

    if validation_errors:
        st.error("Please fix the team data before running simulations.")
        for error in validation_errors:
            st.warning(error)
        st.stop()

    fixtures = load_available_fixtures()
    completed_matches = (
        fixtures["match_status"].isin({"FT", "AET", "PEN"}).sum()
        if "match_status" in fixtures
        else 0
    )
    st.caption(f"Tournament state: {completed_matches} completed matches loaded")

    render_model_quality(teams)
    render_team_table(teams)
    render_match_predictor(teams)
    render_projected_pairings(teams, fixtures, settings["number_of_simulations"])
    render_simulation_results(
        teams,
        settings["number_of_simulations"],
        fixtures=fixtures,
    )


if __name__ == "__main__":
    main()
