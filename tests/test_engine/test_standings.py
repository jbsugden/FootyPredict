"""Tests for the standings engine functions."""

from __future__ import annotations

from predictor.engine.standings import (
    TeamStanding,
    apply_result,
    initialise_standings,
    rank_standings,
)


class TestInitialiseStandings:
    def test_creates_entry_for_each_team(self) -> None:
        standings = initialise_standings(["A", "B", "C"])
        assert set(standings.keys()) == {"A", "B", "C"}

    def test_all_fields_are_zero(self) -> None:
        standings = initialise_standings(["X"])
        ts = standings["X"]
        assert ts.played == 0
        assert ts.won == 0
        assert ts.drawn == 0
        assert ts.lost == 0
        assert ts.goals_for == 0
        assert ts.goals_against == 0
        assert ts.points == 0

    def test_empty_list_returns_empty_dict(self) -> None:
        assert initialise_standings([]) == {}


class TestApplyResult:
    def test_home_win_awards_three_points(self) -> None:
        s = initialise_standings(["H", "A"])
        apply_result(s, "H", "A", 2, 0)
        assert s["H"].points == 3
        assert s["A"].points == 0

    def test_away_win_awards_three_points(self) -> None:
        s = initialise_standings(["H", "A"])
        apply_result(s, "H", "A", 0, 1)
        assert s["H"].points == 0
        assert s["A"].points == 3

    def test_draw_awards_one_point_each(self) -> None:
        s = initialise_standings(["H", "A"])
        apply_result(s, "H", "A", 1, 1)
        assert s["H"].points == 1
        assert s["A"].points == 1

    def test_goals_recorded_correctly(self) -> None:
        s = initialise_standings(["H", "A"])
        apply_result(s, "H", "A", 3, 1)
        assert s["H"].goals_for == 3
        assert s["H"].goals_against == 1
        assert s["A"].goals_for == 1
        assert s["A"].goals_against == 3

    def test_played_incremented(self) -> None:
        s = initialise_standings(["H", "A"])
        apply_result(s, "H", "A", 1, 0)
        assert s["H"].played == 1
        assert s["A"].played == 1

    def test_win_draw_loss_counts(self) -> None:
        s = initialise_standings(["H", "A"])
        apply_result(s, "H", "A", 2, 0)
        assert s["H"].won == 1 and s["H"].drawn == 0 and s["H"].lost == 0
        assert s["A"].won == 0 and s["A"].drawn == 0 and s["A"].lost == 1

    def test_creates_missing_teams(self) -> None:
        s = {}
        apply_result(s, "NEW_H", "NEW_A", 1, 0)
        assert "NEW_H" in s
        assert "NEW_A" in s

    def test_returns_same_dict(self) -> None:
        s = initialise_standings(["H", "A"])
        result = apply_result(s, "H", "A", 1, 0)
        assert result is s


class TestRankStandings:
    def test_sorts_by_points_descending(self) -> None:
        s = initialise_standings(["A", "B", "C"])
        s["A"].points = 10
        s["B"].points = 20
        s["C"].points = 15
        ranked = rank_standings(s)
        assert [r.team_id for r in ranked] == ["B", "C", "A"]

    def test_goal_difference_breaks_tie(self) -> None:
        s = initialise_standings(["A", "B"])
        s["A"].points = 10
        s["A"].goals_for = 15
        s["A"].goals_against = 10
        s["B"].points = 10
        s["B"].goals_for = 20
        s["B"].goals_against = 10
        ranked = rank_standings(s)
        assert ranked[0].team_id == "B"  # better GD

    def test_goals_for_breaks_further_tie(self) -> None:
        s = initialise_standings(["A", "B"])
        s["A"].points = 10
        s["A"].goals_for = 15
        s["A"].goals_against = 5  # GD = 10
        s["B"].points = 10
        s["B"].goals_for = 20
        s["B"].goals_against = 10  # GD = 10
        ranked = rank_standings(s)
        assert ranked[0].team_id == "B"  # more goals scored


class TestTeamStanding:
    def test_goal_difference_property(self) -> None:
        ts = TeamStanding(team_id="X", goals_for=10, goals_against=3)
        assert ts.goal_difference == 7

    def test_sort_key(self) -> None:
        ts = TeamStanding(team_id="X", points=10, goals_for=15, goals_against=5)
        assert ts.sort_key() == (-10, -10, -15)
