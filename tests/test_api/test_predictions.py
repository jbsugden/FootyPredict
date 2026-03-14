"""Tests for the prediction API endpoints.

Tests verify:
- Correct response shapes and status codes
- 404 handling for unknown leagues
- Prediction data structure integrity
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import AsyncClient

from predictor.db.models import League, Prediction, DataSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Tests for GET /api/predictions/{league_id}
# ---------------------------------------------------------------------------


class TestGetLatestPrediction:
    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_league(
        self, async_client: AsyncClient
    ) -> None:
        """A non-existent league ID should return HTTP 404."""
        response = await async_client.get(f"/api/predictions/{_uuid()}")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_returns_404_when_no_prediction_exists(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
    ) -> None:
        """A league with no predictions yet should return 404."""
        response = await async_client.get(f"/api/predictions/{sample_league.id}")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_returns_200_with_correct_shape(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams,
    ) -> None:
        """When a prediction exists, the endpoint should return the correct structure."""
        # Insert a prediction directly into the DB
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": [1.0 if j == i else 0.0 for j in range(len(sample_teams))],
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
        await db_session.flush()

        response = await async_client.get(f"/api/predictions/{sample_league.id}")
        assert response.status_code == 200

        data = response.json()

        # Top-level structure
        assert "prediction_id" in data
        assert "league_id" in data
        assert "season" in data
        assert "generated_at" in data
        assert "simulation_runs" in data
        assert "teams" in data

        # Data values
        assert data["league_id"] == sample_league.id
        assert data["season"] == sample_league.current_season
        assert data["simulation_runs"] == 10_000

        # Teams list
        teams = data["teams"]
        assert isinstance(teams, list)
        assert len(teams) == len(sample_teams)

        # Individual team entry
        first = teams[0]
        assert "team_id" in first
        assert "mean_pos" in first
        assert "mean_points" in first
        assert "pos_dist" in first
        assert isinstance(first["pos_dist"], list)

    @pytest.mark.asyncio
    async def test_teams_are_sorted_by_mean_pos(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams,
    ) -> None:
        """Teams should be returned in ascending mean_pos order."""
        n = len(sample_teams)
        fake_results = {
            team.id: {
                "mean_pos": float(n - i),  # reverse order
                "mean_points": float(i * 5),
                "pos_dist": [1.0 / n] * n,
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=1000,
            results=fake_results,
        )
        db_session.add(prediction)
        await db_session.flush()

        response = await async_client.get(f"/api/predictions/{sample_league.id}")
        assert response.status_code == 200

        teams = response.json()["teams"]
        mean_positions = [t["mean_pos"] for t in teams]
        assert mean_positions == sorted(mean_positions)


# ---------------------------------------------------------------------------
# Tests for POST /api/predictions/{league_id}/run
# ---------------------------------------------------------------------------


class TestRunPrediction:
    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_league(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post(
            f"/api/predictions/{_uuid()}/run",
            headers={"X-Admin-Key": "change_me"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_401_without_admin_key(
        self,
        async_client: AsyncClient,
        sample_league: League,
    ) -> None:
        """Missing admin key should return 401."""
        response = await async_client.post(
            f"/api/predictions/{sample_league.id}/run"
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_returns_403_with_wrong_admin_key(
        self,
        async_client: AsyncClient,
        sample_league: League,
    ) -> None:
        """Incorrect admin key should return 403."""
        response = await async_client.post(
            f"/api/predictions/{sample_league.id}/run",
            headers={"X-Admin-Key": "wrong_key_entirely"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_returns_202_accepted_with_valid_key(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
    ) -> None:
        """Valid request should be accepted (202) and return a confirmation message."""
        response = await async_client.post(
            f"/api/predictions/{sample_league.id}/run",
            headers={"X-Admin-Key": "change_me"},
        )
        assert response.status_code == 202
        data = response.json()
        assert "message" in data
        assert "league_id" in data
        assert data["league_id"] == sample_league.id
