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
