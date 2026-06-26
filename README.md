# World Cup 2026 Winner Predictor

A beginner-friendly Streamlit app that predicts the winner of the 2026 FIFA World Cup by running Monte Carlo tournament simulations.

The app can load `data/sample_teams.csv` or accept an uploaded CSV. It calculates team strength from ranking, Elo, recent form, attack, defense, xG, player availability, context, market intelligence, live tournament position, and host advantage when those columns are available.

## Project Structure

```text
app.py
src/data_loader.py
src/features.py
src/model.py
src/simulator.py
src/explanations.py
src/api_config.py
src/api_client.py
src/update_data.py
data/sample_teams.csv
data/sample_matches.csv
README.md
```

`app.py` is the Streamlit interface. The `src/` files hold the reusable logic for loading data, building features, predicting matches, simulating tournaments, explaining outputs, and fetching optional API-Football data.

## Features

- Home page with project overview
- CSV upload or built-in sample data
- Team table with calculated strength scores
- Match winner prediction with logistic win probabilities
- Group stage simulation for 12 groups of 4 teams
- Advancement for top 2 teams in each group plus 8 best third-place teams
- Knockout simulation from Round of 32 through the Final
- Champion probability table
- Top 10 winner probability bar chart
- Sidebar controls for simulation count and model weights
- Optional advanced inputs for Elo, last-five form, xG, shots, player availability, context, betting markets, and live standings
- Live data readiness table showing which tiers are present in the loaded CSV
- Automatic host advantage for United States, Mexico, and Canada
- Travel burden, rest days, and venue impact placeholders
- Tournament Difficulty Index inspired by Matheson-style World Cup path difficulty analysis
- Average champion route difficulty from Monte Carlo knockout paths

## Data Format

Your CSV must include these columns:

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

The app can use API-Football as a first live-data provider. The adapter is intentionally isolated in `src/api_client.py`, `src/api_config.py`, and `src/update_data.py` so another provider can be swapped in later.

1. Create an API-Football account and get an API key from API-SPORTS:

```text
https://www.api-football.com/
```

2. Create a `.env` file in the project root:

```text
API_FOOTBALL_KEY=your_api_key_here
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run with sample data:

```bash
streamlit run app.py
```

Then keep `Data source` set to `Sample CSV` in the sidebar.

5. Run with live API data:

```bash
streamlit run app.py
```

Then choose `Live API` in the sidebar. If `API_FOOTBALL_KEY` is missing, or if the API data is not yet a complete 48-team tournament dataset, the app falls back to `data/sample_teams.csv` and shows a warning.

To avoid excessive provider requests, live data is cached locally for 12 hours. If an API refresh fails, the app records the attempt and waits 30 minutes before trying again. During that cooldown it uses the cached CSV files when available, otherwise it falls back to sample data.

Live API snapshots are saved locally:

```text
data/live_teams.csv
data/live_fixtures.csv
data/live_standings.csv
data/live_match_stats.csv
data/live_odds.csv
data/live_countries.csv
```

When the World Cup league endpoint returns no teams, the app uses API-Football `GET /countries` as a country-directory fallback. That fallback keeps the simulator's required 48-team shape and includes Scotland when the provider returns it, but it should be treated as candidate-country data rather than confirmed tournament qualification data.

API-Football fields like fixtures, standings, teams, match status, venue city, statistics, and odds are real when the provider returns them. Model fields not directly provided by API-Football, such as FIFA rank, Elo, xG, xGA, travel distance, and rest days, use safe placeholders until a dedicated data source is connected.

## Notes

This is a functional first version, not a betting model. The sample data in `data/sample_teams.csv` is demo data, and the tournament bracket is simplified by seeding qualified teams by their simulated group-stage performance.

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

## Recent Changes

- Preserved the existing Streamlit app and sample CSV.
- Refactored the app into a small `src/` package while keeping `streamlit run app.py` as the entry point.
- Moved the sample team data to `data/sample_teams.csv`.
- Added `data/sample_matches.csv` as a placeholder for future fixture and venue work.
- Added safe defaults so partial CSVs do not break when optional model columns are missing.
- Added automatic 2026 host advantage for USA, Mexico, and Canada.
- Added travel, rest, and venue placeholders to the context score.
- Added a Tournament Difficulty Index table and toughest-route chart.
- Added average champion route difficulty to simulation results.
- Added logistics-specific difficulty inputs: host country, venue country, venue city, climate difference, altitude difference, opponent strength, and venue familiarity.

## Model Roadmap

To keep improving the app, connect daily data feeds in this order:

1. FIFA rankings, Elo ratings, current standings, match results, goals, goal difference, and bracket state.
2. Last-five team form, expected goals, shots, shots on target, big chances, corners, and possession.
3. Injuries, suspensions, expected lineups, goalkeeper status, captain status, minutes, yellow cards, and red cards.
4. Advanced stats such as xGA, pressing, pass accuracy, conversion rate, clean sheets, save percentage, and set-piece goals.
5. Match context: host nation, travel distance, rest days, temperature, altitude, humidity, extra time, and penalty shootout fatigue.
6. Market intelligence: moneyline odds, implied probabilities, and odds movement.
7. Tournament-specific pathing: current group, points, goal difference, best-third-place ranking, round, potential opponent, and bracket difficulty.
