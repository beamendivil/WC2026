# World Cup 2026 Winner Predictor

A beginner-friendly Streamlit app that simulates the 2026 FIFA World Cup and estimates each team's chance of becoming champion.

## Features

- Loads a built-in sample dataset or an uploaded CSV file
- Calculates a strength score for every team
- Predicts head-to-head winners with a logistic win probability
- Simulates:
  - 12 groups of 4 teams
  - top 2 teams from each group
  - 8 best third-place teams
  - Round of 32, Round of 16, Quarterfinals, Semifinals, and Final
- Runs 1,000+ tournament simulations
- Shows champion probabilities and a top-10 winners chart
- Includes sidebar controls for simulation count and scoring weights

## Files

- `/home/runner/work/WC2026/WC2026/app.py` - Streamlit application
- `/home/runner/work/WC2026/WC2026/sample_teams.csv` - sample 48-team dataset
- `/home/runner/work/WC2026/WC2026/requirements.txt` - Python dependencies

## Required CSV columns

Your CSV must include:

- `team`
- `group`
- `fifa_rank`
- `recent_form_score`
- `goals_for`
- `goals_against`
- `host_advantage`

The app expects 48 teams split into 12 groups (`A` to `L`) with 4 teams per group.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Notes on the model

This first version is intentionally simple:

- lower FIFA rank numbers improve the rank score
- recent form and host advantage are weighted from the sidebar
- goals scored improve attack score
- fewer goals conceded improve defense score

It is designed to be functional and easy to understand first, with room to improve the model later.
