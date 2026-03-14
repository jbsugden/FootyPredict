"""League zone definitions and probability computation.

Defines promotion, European qualification, playoff, and relegation zones per
league.  Zone probabilities are computed by summing the relevant slices of a
team's position distribution array (``pos_dist``).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Zone configurations keyed by league code
# ---------------------------------------------------------------------------
# Each zone is a dict with:
#   "label"     — display name
#   "positions" — 1-based positions (negative values are relative to n_teams,
#                 e.g. -1 = last place)
#   "css_class" — CSS class for styling

ZONE_CONFIGS: dict[str, list[dict]] = {
    "PL": [
        {"label": "Champions League", "positions": [1, 2, 3, 4], "css_class": "zone-ucl"},
        {"label": "Europa / Conference", "positions": [5, 6], "css_class": "zone-europa"},
        {"label": "Relegation", "positions": [-3, -2, -1], "css_class": "zone-relegation"},
    ],
    "NPL_W": [
        {"label": "Promotion", "positions": [1], "css_class": "zone-promotion"},
        {"label": "Relegation", "positions": [-3, -2, -1], "css_class": "zone-relegation"},
    ],
}

DEFAULT_ZONES: list[dict] = [
    {"label": "Champion", "positions": [1], "css_class": "zone-promotion"},
    {"label": "Relegation", "positions": [-3, -2, -1], "css_class": "zone-relegation"},
]


def _resolve_positions(positions: list[int], n_teams: int) -> list[int]:
    """Convert negative position indices to absolute 1-based positions."""
    resolved = []
    for p in positions:
        if p < 0:
            absolute = n_teams + 1 + p  # -1 → n_teams, -2 → n_teams-1, etc.
            if absolute >= 1:
                resolved.append(absolute)
        else:
            if p <= n_teams:
                resolved.append(p)
    return resolved


def compute_zone_probabilities(
    pos_dist: list[float],
    league_code: str,
    n_teams: int,
) -> list[dict]:
    """Compute probability of finishing in each defined zone.

    Args:
        pos_dist: Position probability distribution — ``pos_dist[i]`` is the
            probability of finishing in position ``i + 1``.
        league_code: League code (e.g. ``"PL"``, ``"NPL_W"``).
        n_teams: Total number of teams in the league.

    Returns:
        List of dicts with keys ``label``, ``probability``, ``css_class``.
    """
    zones = ZONE_CONFIGS.get(league_code, DEFAULT_ZONES)
    result = []

    for zone in zones:
        positions = _resolve_positions(zone["positions"], n_teams)
        prob = 0.0
        for p in positions:
            idx = p - 1  # pos_dist is 0-indexed
            if 0 <= idx < len(pos_dist):
                prob += pos_dist[idx]
        result.append({
            "label": zone["label"],
            "probability": prob,
            "css_class": zone["css_class"],
        })

    return result


def get_zone_for_position(
    league_code: str,
    position: int,
    n_teams: int,
) -> str | None:
    """Return the CSS class for a given league position, or ``None``.

    Used by the standings table template to highlight rows by zone.

    Args:
        league_code: League code.
        position: 1-based table position.
        n_teams: Total teams in the league.

    Returns:
        CSS class string (e.g. ``"zone-ucl"``) or ``None`` if position is not
        in any defined zone.
    """
    zones = ZONE_CONFIGS.get(league_code, DEFAULT_ZONES)
    for zone in zones:
        positions = _resolve_positions(zone["positions"], n_teams)
        if position in positions:
            return zone["css_class"]
    return None


def build_zone_class_map(league_code: str, n_teams: int) -> dict[int, str]:
    """Build a mapping of position → CSS class for all zoned positions.

    Args:
        league_code: League code.
        n_teams: Total teams in the league.

    Returns:
        Dict mapping 1-based position to CSS class string.
    """
    result: dict[int, str] = {}
    zones = ZONE_CONFIGS.get(league_code, DEFAULT_ZONES)
    for zone in zones:
        positions = _resolve_positions(zone["positions"], n_teams)
        for p in positions:
            result[p] = zone["css_class"]
    return result
