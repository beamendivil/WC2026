# World Cup 2026 Winner Predictor

A Streamlit app that predicts 2026 FIFA World Cup matches, likely Round-of-32 pairings, and tournament winners with a trained international-match model and Monte Carlo simulations.

The model trains on historical senior international results, applies recency and opponent-quality weighting, and blends its estimate with the official FIFA ranking points published on June 11, 2026. Confirmed tournament groups, completed results, eliminated teams, and confirmed knockout fixtures are held fixed instead of being resimulated.

## Project Structure

```text
app.py
src/data_loader.py
src/features.py
src/historical_model.py
src/model.py
src/simulator.py
src/explanations.py
src/api_config.py
src/api_client.py
src/update_data.py
data/sample_teams.csv
data/sample_matches.csv
data/historical_results.csv
data/fifa_rankings.csv
README.md
```

`app.py` is the Streamlit interface. The `src/` files hold the reusable logic for loading data, building features, predicting matches, simulating tournaments, explaining outputs, and fetching optional football-data.org data.

## Features

- Home page with project overview
- Automatic API-first loading with local CSV fallback
- Team table with calculated strength scores
- Calibrated regulation win/draw/loss and knockout advancement probabilities
- Separate regulation, extra-time, and penalty resolution
- Historical Elo, opponent-adjusted recent form, attack, and defense features
- Official FIFA ranking points blended into match probabilities
- FIFA Annex C Round-of-32 placement for all 495 third-place combinations
- Pairing probabilities aggregated across hundreds or thousands of simulations
- Completed cached fixture scores held fixed while unplayed matches are simulated
- Group stage simulation for 12 groups of 4 teams
- Advancement for top 2 teams in each group plus 8 best third-place teams
- Knockout simulation from Round of 32 through the Final
- Champion probability table
- Top 10 winner probability bar chart
- Model validation metrics displayed in the app
- Sidebar control for simulation count
- Optional advanced inputs for Elo, last-five form, xG, shots, player availability, context, betting markets, and live standings
- Live data readiness table showing which tiers are present in the loaded CSV
- Automatic host advantage for United States, Mexico, and Canada
- Travel burden, rest days, and venue impact placeholders
- Tournament Difficulty Index inspired by Matheson-style World Cup path difficulty analysis
- Average champion route difficulty from Monte Carlo knockout paths

## Data Format

The tournament snapshot must include these columns:

```text
team,group,fifa_rank,recent_form_score,goals_for,goals_against,host_advantage
```

Advanced optional columns:

```text
elo_rating,last_five_form,xg_for,xg_against,shots,shots_on_target,big_chances,corners,possession
injury_impact,suspension_impact,expected_lineup_score,goalkeeper_score
host_country,venue_country,venue_city,rest_days,travel_distance_km
climate_difference,altitude_difference,opponent_strength,venue_familiarity
temperature_f,altitude_m,venue_impact
market_implied_prob,current_points,current_goal_difference,current_goals_for
```

`last_five_form` can be entered like `W-W-D-L-W`. The app converts wins to 3 points, draws to 1, and losses to 0.

If `host_advantage` is missing or set to `0`, the app still automatically applies host advantage to `United States`, `Mexico`, and `Canada`.

Example:

```csv
team,group,fifa_rank,recent_form_score,goals_for,goals_against,host_advantage
United States,A,16,7.4,18,10,1
Canada,A,31,6.9,14,11,1
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## Live API Data

The app uses football-data.org v4 as its live-data provider. The adapter is isolated in `src/football_data_client.py`, `src/api_config.py`, and `src/update_data.py`.

1. Create a football-data.org account and get an API token:

```text
https://www.football-data.org/client/register
```

2. Create a `.env` file in the project root:

```text
FOOTBALL_DATA_API_KEY=your_token_here
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run with sample data:

```bash
streamlit run app.py
```

The app reads local snapshots on ordinary page loads and does not contact the provider during Streamlit widget reruns. Use **Refresh API data** in the sidebar when you explicitly want to replace the cached provider data. If `FOOTBALL_DATA_API_KEY` is missing, or if World Cup coverage is unavailable on the account plan, the app uses the bundled tournament snapshot.

Provider responses use endpoint-specific caches from two minutes for fixtures to one day for teams, while ordinary app loads use the local snapshot. If a refresh fails, the app records the attempt and waits 30 minutes before another non-forced refresh. During that cooldown it uses cached CSV files when available, otherwise it falls back to sample data.

The latest matchup probabilities are stored in `data/latest_pairing_predictions.csv` and appear immediately when the app opens. **Refresh matchup predictions** reruns the local Monte Carlo model and replaces that snapshot; it does not make an API request.

Each prediction snapshot fingerprints the bracket, venues, completed results, and model inputs. If any of those change, the app will not present the snapshot as current. Confirmed pairings always override simulated group placement, while host-nation context is applied to match outcome probabilities.

When the provider has no usable World Cup fixtures, completed 2026 tournament results from `data/historical_results.csv` are converted into the same fixture schema. Those scores are held fixed, contribute actual points, goals, and goal difference to simulated group tables, and are included in the forecast fingerprint. Cached provider fixtures take precedence when available.

Live API snapshots are saved locally:

```text
data/live_teams.csv
data/live_fixtures.csv
data/live_standings.csv
data/live_match_stats.csv
data/live_odds.csv
data/live_countries.csv
```

football-data.org fields such as fixtures, standings, squads, scorers, lineups, match status, venues, and available odds are real when the account plan returns them. Official FIFA rank and points come from `data/fifa_rankings.csv`; learned Elo, form, attack, and defense come from `data/historical_results.csv`. Advanced metrics that football-data.org does not provide remain neutral or come from the explicitly bundled tournament snapshot rather than being presented as live observations.

## Model and Validation

The training pipeline uses matches from 2005 through 2022 for fitting, 2023-2024 for probability calibration, and 2025 onward as an out-of-time validation set. Its current validation accuracy is about 61.3% with multiclass log loss of about 0.846, compared with about 1.044 for the class-frequency baseline. The app reports these metrics from the fitted model.

Each historical result updates a leakage-free Elo estimate and exponentially decayed form, attack, and defense state. Recent matches receive more weight, opponent quality adjusts the form and scoring signals, tournament matches carry more weight than friendlies, and single-match goal totals are capped to limit outlier influence. The draw model also uses the combined scoring environment, allowing it to distinguish low-scoring evenly matched teams from high-scoring evenly matched teams. Regulation probabilities are calibrated as home win, draw, and away win; knockout advancement then models extra time and penalties separately.

This is a predictive model, not certainty or betting advice. Forecasts are limited by their inputs. Current injuries, expected lineups, xG, and market odds can improve it further once reliable feeds are connected.

Knockout placement follows FIFA's official 2026 match slots and Annex C third-place allocation table. Confirmed fixtures and completed scores override simulations.

The Tournament Difficulty Index is a placeholder model feature, not a published reproduction of a specific paper. It is conceptually inspired by Victor Matheson's 2018 paper, *The Economics of the World Cup*, only for tournament logistics: host context, travel burden, rest days, climate differences, altitude differences, venue familiarity, and tournament complexity.

It is not an economic claim and does not use economic impact variables such as tourism, infrastructure spending, GDP, public finance, or local business activity as match prediction variables.

The current row-level formula is:

```text
0.30 * opponent_strength
0.20 * normalized travel_distance_km
0.15 * rest_disadvantage
0.15 * climate_difference
0.10 * altitude_difference
0.10 * venue_familiarity_penalty
```

Higher values mean a harder projected path.

## Model Roadmap

To keep improving the app, connect daily data feeds in this order:

1. FIFA rankings, Elo ratings, current standings, match results, goals, goal difference, and bracket state.
2. Last-five team form, expected goals, shots, shots on target, big chances, corners, and possession.
3. Injuries, suspensions, expected lineups, goalkeeper status, captain status, minutes, yellow cards, and red cards.
4. Advanced stats such as xGA, pressing, pass accuracy, conversion rate, clean sheets, save percentage, and set-piece goals.
5. Match context: host nation, travel distance, rest days, temperature, altitude, humidity, extra time, and penalty shootout fatigue.
6. Market intelligence: moneyline odds, implied probabilities, and odds movement.
7. Tournament-specific pathing: current group, points, goal difference, best-third-place ranking, round, potential opponent, and bracket difficulty.
