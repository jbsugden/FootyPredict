"""Fixture completeness analysis for a league season.

Detects missing fixtures by comparing expected home/away pairs (double
round-robin) against actual matches with FINISHED or SCHEDULED status.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any


def check_fixture_completeness(
    team_ids: list[str],
    matches: list[Any],
    team_names: dict[str, str],
) -> dict:
    """Analyse whether any expected fixtures are missing from the data source.

    Args:
        team_ids: All team IDs in the league.
        matches: Match objects with ``home_team_id``, ``away_team_id``, and
            ``status`` (string or enum with ``.value``).
        team_names: Mapping of team_id to display name.

    Returns:
        Dict with ``expected_total``, ``actual_total``, ``missing_count``,
        ``missing_fixtures`` list, and ``teams_affected`` dict.
    """
    expected_pairs: set[tuple[str, str]] = set(permutations(team_ids, 2))

    # A pair is covered if at least one FINISHED or SCHEDULED match exists
    covered_statuses = {"FINISHED", "SCHEDULED"}
    actual_pairs: set[tuple[str, str]] = set()
    for m in matches:
        status_val = m.status.value if hasattr(m.status, "value") else str(m.status)
        if status_val in covered_statuses:
            actual_pairs.add((m.home_team_id, m.away_team_id))

    missing_pairs = expected_pairs - actual_pairs

    missing_fixtures = [
        {
            "home_id": h,
            "away_id": a,
            "home_name": team_names.get(h, h),
            "away_name": team_names.get(a, a),
        }
        for h, a in sorted(missing_pairs, key=lambda p: (team_names.get(p[0], p[0]), team_names.get(p[1], p[1])))
    ]

    # Build teams_affected: count how many missing fixtures involve each team
    teams_affected: dict[str, dict] = {}
    for h, a in missing_pairs:
        for tid in (h, a):
            if tid not in teams_affected:
                teams_affected[tid] = {"team_name": team_names.get(tid, tid), "deficit": 0}
            teams_affected[tid]["deficit"] += 1

    return {
        "expected_total": len(expected_pairs),
        "actual_total": len(actual_pairs),
        "missing_count": len(missing_pairs),
        "missing_fixtures": missing_fixtures,
        "teams_affected": teams_affected,
    }
