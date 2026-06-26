import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import add_safe_defaults, load_sample_data, validate_team_data
from src.explanations import DIFFICULTY_COLUMNS, DIFFICULTY_EXPLANATION
from src.explanations import MATHESON_NOTE, MODEL_COLUMNS
from src.explanations import build_data_readiness_table, build_team_explanations
from src.features import add_strength_scores, calculate_tournament_difficulty_index
from src.model import win_probability
from src.simulator import run_simulations
from src.update_data import load_live_model_data


@st.cache_data
def cached_sample_data():
    """Cache the sample CSV for Streamlit reruns."""
    return load_sample_data()


def render_sidebar():
    """Create sidebar controls and return selected settings."""
    st.sidebar.header("Data")
    data_source = st.sidebar.radio("Data source", ["Sample CSV", "Live API"])

    st.sidebar.header("Simulation Controls")
    number_of_simulations = st.sidebar.slider(
        "Number of simulations", min_value=100, max_value=5000, value=1000, step=100
    )
    ranking_weight = st.sidebar.slider(
        "Ranking weight", min_value=0.0, max_value=3.0, value=1.0, step=0.1
    )
    form_weight = st.sidebar.slider(
        "Recent form weight", min_value=0.0, max_value=3.0, value=1.0, step=0.1
    )
    host_weight = st.sidebar.slider(
        "Host advantage weight", min_value=0.0, max_value=10.0, value=2.0, step=0.5
    )

    st.sidebar.header("Advanced Model Weights")
    elo_weight = st.sidebar.slider(
        "Elo weight", min_value=0.0, max_value=3.0, value=1.0, step=0.1
    )
    xg_weight = st.sidebar.slider(
        "xG and shots weight", min_value=0.0, max_value=3.0, value=1.0, step=0.1
    )
    player_weight = st.sidebar.slider(
        "Player availability weight", min_value=0.0, max_value=3.0, value=0.5, step=0.1
    )
    context_weight = st.sidebar.slider(
        "Context weight", min_value=0.0, max_value=3.0, value=0.5, step=0.1
    )
    market_weight = st.sidebar.slider(
        "Market intelligence weight", min_value=0.0, max_value=3.0, value=0.5, step=0.1
    )

    return {
        "data_source": data_source,
        "number_of_simulations": number_of_simulations,
        "ranking_weight": ranking_weight,
        "form_weight": form_weight,
        "host_weight": host_weight,
        "elo_weight": elo_weight,
        "xg_weight": xg_weight,
        "player_weight": player_weight,
        "context_weight": context_weight,
        "market_weight": market_weight,
    }


def load_selected_data(settings, uploaded_file):
    """Load sample/uploaded data or API-Football data with safe fallback."""
    if uploaded_file:
        return pd.read_csv(uploaded_file), "Uploaded CSV"

    if settings["data_source"] == "Live API":
        live_teams, message = load_live_model_data()
        live_teams = add_safe_defaults(live_teams) if not live_teams.empty else live_teams
        if live_teams.empty:
            st.warning(f"{message} Falling back to sample CSV data.")
            return cached_sample_data(), "Sample CSV fallback"

        live_errors = validate_team_data(live_teams)
        if live_errors:
            st.warning(
                "Live API data was fetched, but it is not yet a complete "
                "48-team tournament dataset. Falling back to sample CSV data."
            )
            st.caption("Live data issue: " + " ".join(live_errors))
            return cached_sample_data(), "Sample CSV fallback"

        st.success(message)
        return live_teams, "Live API"

    return cached_sample_data(), "Sample CSV"


def prepare_team_strength(raw_teams, settings):
    """Apply defaults, validate, and calculate team strength."""
    raw_teams = add_safe_defaults(raw_teams)
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


def render_data_readiness(raw_teams):
    """Render the optional data coverage table."""
    st.subheader("Live Data Readiness")
    st.write(
        "The simulator accepts richer optional columns for daily data feeds. "
        "Missing columns get neutral defaults, so the sample CSV still works."
    )
    st.dataframe(build_data_readiness_table(raw_teams), width="stretch", hide_index=True)


def render_team_table(teams):
    """Render the calculated team strength table."""
    st.subheader("Team Data")
    available_columns = [column for column in MODEL_COLUMNS if column in teams.columns]
    st.dataframe(
        teams[available_columns].sort_values("strength_score", ascending=False),
        width="stretch",
    )


def render_difficulty_index(difficulty_index):
    """Render the Tournament Difficulty Index table."""
    st.subheader("Tournament Difficulty Index")
    st.write(DIFFICULTY_EXPLANATION)
    st.caption(MATHESON_NOTE)
    st.dataframe(
        difficulty_index[DIFFICULTY_COLUMNS],
        width="stretch",
        hide_index=True,
    )


def render_team_explanations(teams, difficulty_index):
    """Render short plain-English explanation rows for each team."""
    st.subheader("Team Prediction Explanations")
    explanations = build_team_explanations(teams, difficulty_index)
    st.dataframe(
        explanations[
            [
                "team",
                "strength_score",
                "tournament_difficulty",
                "host_advantage",
                "recent_form",
                "attack_score",
                "defense_score",
                "biggest_positive_factor",
                "biggest_negative_factor",
                "explanation",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def render_match_predictor(teams):
    """Render a simple head-to-head match predictor."""
    st.subheader("Match Predictor")
    col1, col2 = st.columns(2)
    team_names = teams["team"].sort_values().tolist()
    team_a_name = col1.selectbox("Team A", team_names, index=0)
    team_b_name = col2.selectbox("Team B", team_names, index=1)

    if team_a_name == team_b_name:
        st.info("Choose two different teams to predict a match.")
        return

    team_a = teams.loc[teams["team"] == team_a_name].iloc[0]
    team_b = teams.loc[teams["team"] == team_b_name].iloc[0]
    probability_a = win_probability(team_a["strength_score"], team_b["strength_score"])
    predicted_winner = team_a_name if probability_a >= 0.5 else team_b_name

    st.metric("Predicted winner", predicted_winner)
    st.write(
        f"{team_a_name}: {probability_a:.1%} win probability | "
        f"{team_b_name}: {(1 - probability_a):.1%} win probability"
    )


def render_simulation_results(teams, difficulty_index, number_of_simulations):
    """Run Monte Carlo simulations and render the result tables and charts."""
    st.subheader("Tournament Simulation")
    if not st.button("Run simulations", type="primary"):
        return

    progress_bar = st.progress(0)
    results = run_simulations(
        teams,
        number_of_simulations,
        progress_callback=progress_bar.progress,
    )
    progress_bar.empty()
    results = results.merge(
        difficulty_index[["team", "tournament_difficulty_index"]],
        on="team",
        how="left",
    )
    top_10 = results.head(10)

    st.success(f"Completed {number_of_simulations:,} tournament simulations.")
    st.dataframe(results, width="stretch")

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

    difficulty_chart = px.bar(
        difficulty_index.head(10).sort_values("tournament_difficulty_index"),
        x="tournament_difficulty_index",
        y="team",
        orientation="h",
        labels={
            "tournament_difficulty_index": "Tournament Difficulty Index",
            "team": "Team",
        },
        title="Top 10 Toughest Projected Routes",
    )
    st.plotly_chart(difficulty_chart)

    winner = results.iloc[0]
    st.subheader("Most Likely Champion")
    st.write(
        f"{winner['team']} won {winner['championships']:,} out of "
        f"{number_of_simulations:,} simulations "
        f"({winner['champion_probability']:.2f}%)."
    )


def main():
    st.set_page_config(
        page_title="World Cup 2026 Winner Predictor",
        page_icon=":trophy:",
        layout="wide",
    )

    st.title("World Cup 2026 Winner Predictor")
    st.write(
        "Simulate the 2026 FIFA World Cup thousands of times to estimate each "
        "team's chance of lifting the trophy. Upload your own team data or use "
        "the included sample dataset."
    )

    settings = render_sidebar()
    uploaded_file = st.sidebar.file_uploader("Upload team CSV", type=["csv"])
    raw_teams, active_source = load_selected_data(settings, uploaded_file)
    st.caption(f"Active data source: {active_source}")
    raw_teams, teams, validation_errors = prepare_team_strength(raw_teams, settings)

    if validation_errors:
        st.error("Please fix the team data before running simulations.")
        for error in validation_errors:
            st.warning(error)
        st.stop()

    difficulty_index = calculate_tournament_difficulty_index(teams)

    render_data_readiness(raw_teams)
    render_team_table(teams)
    render_difficulty_index(difficulty_index)
    render_team_explanations(teams, difficulty_index)
    render_match_predictor(teams)
    render_simulation_results(
        teams, difficulty_index, settings["number_of_simulations"]
    )


if __name__ == "__main__":
    main()
