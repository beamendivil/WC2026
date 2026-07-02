from pathlib import Path
from functools import lru_cache

import pandas as pd


THIRD_PLACE_MAPPING_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "fifa_third_place_mapping.csv"
)

ROUND_OF_32_SLOTS = {
    73: ("2A", "2B"),
    74: ("1E", "3"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "3"),
    78: ("2E", "2I"),
    79: ("1A", "3"),
    80: ("1L", "3"),
    81: ("1D", "3"),
    82: ("1G", "3"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "3"),
    86: ("1J", "2H"),
    87: ("1K", "3"),
    88: ("2D", "2G"),
}

THIRD_PLACE_MATCH_BY_WINNER = {
    "1E": 74,
    "1I": 77,
    "1A": 79,
    "1L": 80,
    "1D": 81,
    "1G": 82,
    "1B": 85,
    "1K": 87,
}

# Confirmed fixtures override simulated group positions as the bracket fills in.
CONFIRMED_ROUND_OF_32 = {
    73: ("South Africa", "Canada"),
    74: ("Germany", "Paraguay"),
    75: ("Netherlands", "Morocco"),
    76: ("Brazil", "Japan"),
    77: ("France", "Sweden"),
    78: ("Cote d'Ivoire", "Norway"),
    79: ("Mexico", "Ecuador"),
    80: ("England", "Congo DR"),
    81: ("United States", "Bosnia and Herzegovina"),
    82: ("Belgium", "Senegal"),
    83: ("Portugal", "Croatia"),
    84: ("Spain", "Austria"),
    85: ("Switzerland", "Algeria"),
    86: ("Argentina", "Cabo Verde"),
    87: ("Colombia", "Ghana"),
    88: ("Australia", "Egypt"),
}

# Completed knockout matches are observations, not probabilities. Their winners
# stay fixed in every simulation and feed the correct team into the next round.
CONFIRMED_KNOCKOUT_WINNERS = {
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

# Positions already settled by completed groups and confirmed knockout fixtures.
CONFIRMED_GROUP_POSITIONS = {
    "A": {1: "Mexico", 2: "South Africa"},
    "B": {2: "Canada", 3: "Bosnia and Herzegovina"},
    "C": {1: "Brazil", 2: "Morocco"},
    "D": {1: "United States", 2: "Australia", 3: "Paraguay"},
    "E": {1: "Germany", 2: "Cote d'Ivoire", 3: "Ecuador"},
    "F": {1: "Netherlands", 2: "Japan", 3: "Sweden"},
    "G": {1: "Belgium", 2: "Egypt", 3: "Iran"},
    "H": {1: "Spain", 2: "Cabo Verde"},
    "I": {1: "France", 2: "Norway"},
    "J": {1: "Argentina"},
    "K": {1: "Colombia", 2: "Portugal", 3: "Congo DR"},
    "L": {1: "England", 2: "Croatia", 3: "Ghana"},
}

CONFIRMED_THIRD_PLACE_QUALIFIERS = {
    "Bosnia and Herzegovina",
    "Ecuador",
    "Paraguay",
    "Sweden",
    "Congo DR",
    "Ghana",
}

KNOCKOUT_PATHS = {
    "Round of 16": {
        89: (74, 77),
        90: (73, 75),
        91: (76, 78),
        92: (79, 80),
        93: (83, 84),
        94: (81, 82),
        95: (86, 88),
        96: (85, 87),
    },
    "Quarterfinals": {
        97: (89, 90),
        98: (93, 94),
        99: (91, 92),
        100: (95, 96),
    },
    "Semifinals": {
        101: (97, 98),
        102: (99, 100),
    },
    "Final": {104: (101, 102)},
}


@lru_cache(maxsize=1)
def load_third_place_mapping():
    """Load FIFA Annex C's 495 third-place allocation combinations."""
    return pd.read_csv(THIRD_PLACE_MAPPING_PATH, dtype=str).set_index(
        "qualifying_groups"
    )


def resolve_third_place_assignments(third_place_teams):
    """Map the eight qualifying third-place groups to their group winners."""
    qualifying_groups = "".join(sorted(third_place_teams["group"].astype(str)))
    mapping = load_third_place_mapping()
    if qualifying_groups not in mapping.index:
        raise ValueError(
            "No FIFA Annex C mapping for third-place groups "
            f"{qualifying_groups or 'none'}."
        )

    teams_by_group = third_place_teams.set_index("group")["team"].to_dict()
    row = mapping.loc[qualifying_groups]
    return {
        THIRD_PLACE_MATCH_BY_WINNER[winner_slot]: teams_by_group[group]
        for winner_slot, group in row.items()
    }


def build_round_of_32(qualified_teams):
    """Place qualifiers into FIFA's official Round-of-32 match slots."""
    positions = {
        f"{int(row['group_position'])}{row['group']}": row
        for _, row in qualified_teams.iterrows()
        if int(row["group_position"]) in (1, 2)
    }
    third_place_teams = qualified_teams.loc[
        qualified_teams["group_position"] == 3
    ]
    third_place_assignments = resolve_third_place_assignments(third_place_teams)

    matches = []
    for match_number, (slot_a, slot_b) in ROUND_OF_32_SLOTS.items():
        team_a = positions[slot_a]
        team_b = (
            qualified_teams.loc[
                qualified_teams["team"] == third_place_assignments[match_number]
            ].iloc[0]
            if slot_b == "3"
            else positions[slot_b]
        )
        matches.append((match_number, team_a, team_b))
    return matches
