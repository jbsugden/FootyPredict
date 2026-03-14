"""Head-to-head history adjustment for match predictions.

Blends historical meeting results into the base Poisson lambdas so that
fixtures between teams with a meaningful head-to-head record are nudged
toward their historical scoring patterns.
"""

from __future__ import annotations

from datetime import datetime

from predictor.engine.poisson import MatchRecord, _get_weight

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

H2H_MAX_WEIGHT = 0.15
"""Maximum blending weight applied when h2h sample size is large."""

H2H_MIN_MATCHES = 2
"""Minimum number of h2h meetings required before any adjustment is made."""

H2H_K = 3.0
"""Shrinkage constant — controls how quickly the weight ramps up with more
meetings.  w = H2H_MAX_WEIGHT * (n / (n + H2H_K))."""


def adjust_lambdas_h2h(
    lambda_home: float,
    lambda_away: float,
    matches: list[MatchRecord],
    home_id: str,
    away_id: str,
    now: datetime | None = None,
) -> tuple[float, float]:
    """Blend head-to-head history into base expected-goal lambdas.

    Args:
        lambda_home: Base expected goals for the home side.
        lambda_away: Base expected goals for the away side.
        matches: Full list of historical match records.
        home_id: Team ID of the home side in the upcoming fixture.
        away_id: Team ID of the away side in the upcoming fixture.
        now: Reference datetime for time-decay (defaults to UTC now).

    Returns:
        Tuple ``(adjusted_lambda_home, adjusted_lambda_away)``, floored
        at 0.01.
    """
    # Filter to meetings between these two teams (in either direction)
    h2h = [
        m
        for m in matches
        if {m.home_team_id, m.away_team_id} == {home_id, away_id}
    ]

    if len(h2h) < H2H_MIN_MATCHES:
        return lambda_home, lambda_away

    # Compute time-weighted average goals from h2h meetings, mapped to the
    # perspective of the upcoming fixture's home/away assignment.
    weighted_home_goals = 0.0
    weighted_away_goals = 0.0
    total_weight = 0.0

    for m in h2h:
        w = _get_weight(m.played_at, now)
        if m.home_team_id == home_id:
            # Same orientation as the upcoming fixture
            weighted_home_goals += m.home_goals * w
            weighted_away_goals += m.away_goals * w
        else:
            # Reversed — swap perspectives
            weighted_home_goals += m.away_goals * w
            weighted_away_goals += m.home_goals * w
        total_weight += w

    if total_weight == 0:
        return lambda_home, lambda_away

    h2h_avg_home = weighted_home_goals / total_weight
    h2h_avg_away = weighted_away_goals / total_weight

    # Blending weight modulated by sample size (shrinkage pattern)
    n_h2h = len(h2h)
    blend_w = H2H_MAX_WEIGHT * (n_h2h / (n_h2h + H2H_K))

    adjusted_home = (1 - blend_w) * lambda_home + blend_w * h2h_avg_home
    adjusted_away = (1 - blend_w) * lambda_away + blend_w * h2h_avg_away

    return max(adjusted_home, 0.01), max(adjusted_away, 0.01)
