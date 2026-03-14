"""Tests for the admin API endpoints."""

from __future__ import annotations

import io

import pytest
from httpx import AsyncClient

from predictor.db.models import League


VALID_ADMIN_KEY = "change_me"  # matches default in config.py


class TestAdminAuth:
    @pytest.mark.asyncio
    async def test_sync_returns_401_without_key(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post("/admin/sync")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_sync_returns_403_with_wrong_key(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post(
            "/admin/sync",
            headers={"X-Admin-Key": "wrong_key"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_import_csv_returns_401_without_key(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post("/admin/import-csv")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_import_csv_returns_403_with_wrong_key(
        self, async_client: AsyncClient
    ) -> None:
        csv_content = b"date,home_team,away_team,home_goals,away_goals,league_code,season\n"
        response = await async_client.post(
            "/admin/import-csv",
            headers={"X-Admin-Key": "wrong_key"},
            files={"file": ("results.csv", io.BytesIO(csv_content), "text/csv")},
        )
        assert response.status_code == 403


class TestImportCSV:
    @pytest.mark.asyncio
    async def test_rejects_non_csv_file(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post(
            "/admin/import-csv",
            headers={"X-Admin-Key": VALID_ADMIN_KEY},
            files={"file": ("data.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_csv_missing_columns(
        self, async_client: AsyncClient
    ) -> None:
        csv_content = b"date,home_team\n01/01/2025,Arsenal\n"
        response = await async_client.post(
            "/admin/import-csv",
            headers={"X-Admin-Key": VALID_ADMIN_KEY},
            files={"file": ("results.csv", io.BytesIO(csv_content), "text/csv")},
        )
        assert response.status_code == 400
        assert "missing" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_valid_csv_with_unknown_league_reports_errors(
        self,
        async_client: AsyncClient,
    ) -> None:
        csv_content = (
            b"date,home_team,away_team,home_goals,away_goals,league_code,season\n"
            b"01/01/2025,Team A,Team B,2,1,FAKE_LEAGUE,2024-25\n"
        )
        response = await async_client.post(
            "/admin/import-csv",
            headers={"X-Admin-Key": VALID_ADMIN_KEY},
            files={"file": ("results.csv", io.BytesIO(csv_content), "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rows_failed"] >= 1

    @pytest.mark.asyncio
    async def test_valid_csv_import_succeeds(
        self,
        async_client: AsyncClient,
        sample_league: League,
    ) -> None:
        csv_content = (
            f"date,home_team,away_team,home_goals,away_goals,league_code,season\n"
            f"01/01/2025,New Team A,New Team B,2,1,{sample_league.code},{sample_league.current_season}\n"
        ).encode()
        response = await async_client.post(
            "/admin/import-csv",
            headers={"X-Admin-Key": VALID_ADMIN_KEY},
            files={"file": ("results.csv", io.BytesIO(csv_content), "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rows_processed"] == 1
        assert data["rows_failed"] == 0
