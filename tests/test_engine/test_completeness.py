"""Tests for fixture completeness detection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from predictor.engine.completeness import check_fixture_completeness


@dataclass
class FakeMatch:
    home_team_id: str
    away_team_id: str
    status: str


TEAMS = ["A", "B", "C", "D"]
NAMES = {"A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta"}


def _all_pairs(teams: list[str]) -> list[tuple[str, str]]:
    return [(h, a) for h in teams for a in teams if h != a]


def _make_matches(pairs: list[tuple[str, str]], status: str = "FINISHED") -> list[FakeMatch]:
    return [FakeMatch(h, a, status) for h, a in pairs]


class TestCompleteLeague:
    def test_complete_league_has_no_missing(self):
        pairs = _all_pairs(TEAMS)
        matches = _make_matches(pairs)
        result = check_fixture_completeness(TEAMS, matches, NAMES)

        assert result["expected_total"] == 12  # 4 * 3
        assert result["actual_total"] == 12
        assert result["missing_count"] == 0
        assert result["missing_fixtures"] == []
        assert result["teams_affected"] == {}


class TestOneMissing:
    def test_detects_single_missing_fixture(self):
        pairs = _all_pairs(TEAMS)
        pairs.remove(("A", "B"))
        matches = _make_matches(pairs)
        result = check_fixture_completeness(TEAMS, matches, NAMES)

        assert result["missing_count"] == 1
        assert result["actual_total"] == 11
        assert len(result["missing_fixtures"]) == 1
        mf = result["missing_fixtures"][0]
        assert mf["home_id"] == "A"
        assert mf["away_id"] == "B"
        assert mf["home_name"] == "Alpha"
        assert mf["away_name"] == "Bravo"


class TestAbandonedRescheduled:
    def test_abandoned_with_replacement_not_missing(self):
        """Abandoned match + SCHEDULED replacement → pair is covered."""
        pairs = _all_pairs(TEAMS)
        matches = _make_matches(pairs)
        # Replace the FINISHED (A,B) with a CANCELLED + SCHEDULED pair
        matches = [m for m in matches if not (m.home_team_id == "A" and m.away_team_id == "B")]
        matches.append(FakeMatch("A", "B", "CANCELLED"))
        matches.append(FakeMatch("A", "B", "SCHEDULED"))

        result = check_fixture_completeness(TEAMS, matches, NAMES)
        assert result["missing_count"] == 0


class TestAbandonedNoReplacement:
    def test_abandoned_without_replacement_is_missing(self):
        """Cancelled match with no FINISHED/SCHEDULED replacement → missing."""
        pairs = _all_pairs(TEAMS)
        matches = _make_matches(pairs)
        matches = [m for m in matches if not (m.home_team_id == "A" and m.away_team_id == "B")]
        matches.append(FakeMatch("A", "B", "CANCELLED"))

        result = check_fixture_completeness(TEAMS, matches, NAMES)
        assert result["missing_count"] == 1
        assert result["missing_fixtures"][0]["home_name"] == "Alpha"


class TestTeamsAffected:
    def test_deficit_counts_are_correct(self):
        """Remove (A,B) and (A,C) — A should have deficit 2, B and C deficit 1 each."""
        pairs = _all_pairs(TEAMS)
        pairs.remove(("A", "B"))
        pairs.remove(("A", "C"))
        matches = _make_matches(pairs)
        result = check_fixture_completeness(TEAMS, matches, NAMES)

        assert result["missing_count"] == 2
        affected = result["teams_affected"]
        assert affected["A"]["deficit"] == 2
        assert affected["A"]["team_name"] == "Alpha"
        assert affected["B"]["deficit"] == 1
        assert affected["C"]["deficit"] == 1
        assert "D" not in affected


class TestScheduledCountsAsCovered:
    def test_scheduled_matches_are_not_missing(self):
        """SCHEDULED matches should count as covered, not just FINISHED."""
        pairs = _all_pairs(TEAMS)
        matches = []
        for h, a in pairs:
            status = "SCHEDULED" if h == "A" else "FINISHED"
            matches.append(FakeMatch(h, a, status))

        result = check_fixture_completeness(TEAMS, matches, NAMES)
        assert result["missing_count"] == 0


class TestPostponedIsMissing:
    def test_postponed_without_replacement_is_missing(self):
        """POSTPONED match with no replacement → counts as missing."""
        pairs = _all_pairs(TEAMS)
        matches = _make_matches(pairs)
        matches = [m for m in matches if not (m.home_team_id == "A" and m.away_team_id == "B")]
        matches.append(FakeMatch("A", "B", "POSTPONED"))

        result = check_fixture_completeness(TEAMS, matches, NAMES)
        assert result["missing_count"] == 1
