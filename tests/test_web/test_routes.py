"""Tests for Jinja2 web page routes.

Verifies template rendering, nav links, and regression for the
``| enumerate`` template bug that caused Internal Server Error.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import AsyncClient

from predictor.db.models import League, Prediction, Team


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class TestIndexPage:
    @pytest.mark.asyncio
    async def test_index_returns_200(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_index_contains_brand(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/")
        assert "FootyPredict" in response.text

    @pytest.mark.asyncio
    async def test_index_lists_leagues(
        self, async_client: AsyncClient, sample_league: League
    ) -> None:
        response = await async_client.get("/")
        assert response.status_code == 200
        assert sample_league.name in response.text


class TestLeagueDetailPage:
    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_league(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get(f"/league/{_uuid()}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_200_with_standings(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert sample_league.name in response.text
        # Standings table should be rendered
        assert "Arsenal FC" in response.text

    @pytest.mark.asyncio
    async def test_renders_prediction_table_without_error(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Regression test: prediction_table.html must render without crashing.

        Previously the template used ``| enumerate`` (not a valid Jinja2 filter)
        which caused an Internal Server Error.
        """
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
        await db_session.flush()

        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert "Predicted Final Table" in response.text
        # Check that top-positions labels rendered (regression for | enumerate bug)
        assert "P(" in response.text

    @pytest.mark.asyncio
    async def test_nav_contains_dynamic_league_links(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_team_seasons,
    ) -> None:
        """Nav should contain links with league UUID paths, not hardcoded slugs."""
        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert f"/league/{sample_league.id}" in response.text

    @pytest.mark.asyncio
    async def test_no_htmx_json_auto_load(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_team_seasons,
    ) -> None:
        """The standings div should NOT have hx-get pointing to the JSON API."""
        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert 'hx-get="/api/leagues/' not in response.text

    @pytest.mark.asyncio
    async def test_footer_references_thenpl(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_team_seasons,
    ) -> None:
        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert "thenpl.co.uk" in response.text
        assert "FA Full Time" not in response.text


class TestAdminImportPage:
    @pytest.mark.asyncio
    async def test_admin_import_returns_200(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get("/admin/import")
        assert response.status_code == 200
