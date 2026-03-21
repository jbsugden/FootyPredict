"""Tests for the scenario (what-if) API endpoint.

Tests verify:
- 404 for unknown league and missing baseline prediction
- 422 for invalid request bodies (empty locked_results, negative goals)
- 200 with correct response shape for valid requests
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import AsyncClient

from predictor.db.models import (
    League,
    Match,
    MatchStatus,
    Prediction,
    Team,
    TeamSeason,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Tests for POST /api/predictions/{league_id}/scenario
# ---------------------------------------------------------------------------


class TestScenarioEndpoint:
    @pytest.mark.asyncio
    async def test_scenario_returns_404_for_unknown_league(
        self, async_client: AsyncClient
    ) -> None:
        """A non-existent league ID should return HTTP 404."""
        response = await async_client.post(
            f"/api/predictions/{_uuid()}/scenario",
            json={"locked_results": [
                {"home_team_id": _uuid(), "away_team_id": _uuid(),
                 "home_goals": 1, "away_goals": 0}
            ]},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_scenario_returns_404_without_baseline_prediction(
        self,
        async_client: AsyncClient,
        sample_league: League,
    ) -> None:
        """A league with no prediction should return 404 with clear message."""
        response = await async_client.post(
            f"/api/predictions/{sample_league.id}/scenario",
            json={"locked_results": [
                {"home_team_id": _uuid(), "away_team_id": _uuid(),
                 "home_goals": 1, "away_goals": 0}
            ]},
        )
        assert response.status_code == 404
        assert "No baseline prediction" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_scenario_returns_422_with_empty_locked_results(
        self,
        async_client: AsyncClient,
        sample_league: League,
    ) -> None:
        """Empty locked_results should fail Pydantic validation (min_length=1)."""
        response = await async_client.post(
            f"/api/predictions/{sample_league.id}/scenario",
            json={"locked_results": []},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_scenario_returns_200_with_valid_request(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
        sample_matches,
    ) -> None:
        """Valid scenario request should return 200 with correct response shape."""
        # Insert a baseline prediction
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": [
                    0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                    for j in range(len(sample_teams))
                ],
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)

        # Insert a scheduled match to lock
        scheduled = Match(
            id=_uuid(),
            league_id=sample_league.id,
            season="2024-25",
            matchday=10,
            home_team_id=sample_teams[0].id,
            away_team_id=sample_teams[1].id,
            status=MatchStatus.SCHEDULED,
            played_at=_now(),
        )
        db_session.add(scheduled)
        await db_session.flush()

        response = await async_client.post(
            f"/api/predictions/{sample_league.id}/scenario",
            json={"locked_results": [
                {
                    "home_team_id": sample_teams[0].id,
                    "away_team_id": sample_teams[1].id,
                    "home_goals": 2,
                    "away_goals": 1,
                }
            ]},
        )
        assert response.status_code == 200

        data = response.json()
        assert "teams" in data
        assert len(data["teams"]) == 4
        assert data["simulation_runs"] == 2000
        assert data["locked_count"] == 1

        # Each team entry should have the expected fields
        for team_entry in data["teams"]:
            assert "team_id" in team_entry
            assert "team_name" in team_entry
            assert "mean_pos" in team_entry
            assert "baseline_mean_pos" in team_entry
            assert "pos_change" in team_entry

    @pytest.mark.asyncio
    async def test_scenario_returns_422_for_invalid_goal_bounds(
        self,
        async_client: AsyncClient,
        sample_league: League,
    ) -> None:
        """Negative goal values should fail Pydantic validation (ge=0)."""
        response = await async_client.post(
            f"/api/predictions/{sample_league.id}/scenario",
            json={"locked_results": [
                {
                    "home_team_id": _uuid(),
                    "away_team_id": _uuid(),
                    "home_goals": -1,
                    "away_goals": 0,
                }
            ]},
        )
        assert response.status_code == 422
