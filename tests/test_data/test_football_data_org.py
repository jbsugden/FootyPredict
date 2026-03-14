"""Tests for the football-data.org API adapter.

Real HTTP calls are not made in unit tests. Use VCR cassettes (pytest-recording
or vcrpy) to record and replay actual API responses.

TODO: Install pytest-recording and record cassettes:
      ``pip install pytest-recording``
      ``pytest --record-mode=once tests/test_data/test_football_data_org.py``
      Cassette files will be saved to tests/cassettes/.

TODO: Add cassette files to .gitignore if they contain API keys in headers
      (use VCR's filter_headers option to strip X-Auth-Token).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from predictor.data.football_data_org import FootballDataOrgSource, LEAGUE_CODE_MAP
from predictor.data.base import MatchData, StandingRow
from predictor.db.models import MatchStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_API_KEY = "test_api_key_do_not_use"

FAKE_STANDINGS_RESPONSE = {
    "standings": [
        {
            "type": "TOTAL",
            "table": [
                {
                    "position": 1,
                    "team": {"id": 57, "name": "Arsenal FC", "shortName": "Arsenal"},
                    "playedGames": 20,
                    "won": 15,
                    "draw": 3,
                    "lost": 2,
                    "goalsFor": 48,
                    "goalsAgainst": 18,
                    "points": 48,
                    "form": "WWWDW",
                },
                {
                    "position": 2,
                    "team": {"id": 64, "name": "Liverpool FC", "shortName": "Liverpool"},
                    "playedGames": 20,
                    "won": 14,
                    "draw": 3,
                    "lost": 3,
                    "goalsFor": 42,
                    "goalsAgainst": 20,
                    "points": 45,
                    "form": "WWLDW",
                },
            ],
        }
    ]
}

FAKE_MATCHES_RESPONSE = {
    "matches": [
        {
            "id": 123456,
            "utcDate": "2024-10-01T15:00:00Z",
            "status": "FINISHED",
            "matchday": 7,
            "homeTeam": {"id": 57, "name": "Arsenal FC", "shortName": "Arsenal"},
            "awayTeam": {"id": 64, "name": "Liverpool FC", "shortName": "Liverpool"},
            "score": {
                "fullTime": {"home": 2, "away": 1},
            },
        }
    ]
}


# ---------------------------------------------------------------------------
# Unit tests (with mocked HTTP)
# ---------------------------------------------------------------------------


class TestFootballDataOrgSourceUnit:
    """Unit tests that mock the HTTP client."""

    @pytest.fixture
    def source(self) -> FootballDataOrgSource:
        import httpx

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        return FootballDataOrgSource(api_key=FAKE_API_KEY, client=mock_client)

    def test_league_code_map_contains_pl(self) -> None:
        """Sanity check that the PL code is mapped."""
        assert "PL" in LEAGUE_CODE_MAP
        assert LEAGUE_CODE_MAP["PL"] == "PL"

    @pytest.mark.asyncio
    async def test_fetch_standings_returns_standing_rows(
        self, source: FootballDataOrgSource
    ) -> None:
        """fetch_standings should parse the standings API response correctly."""
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = FAKE_STANDINGS_RESPONSE
        mock_response.raise_for_status = MagicMock()
        source._client.get.return_value = mock_response
        source._last_request_time = 0.0  # skip throttle

        with patch("predictor.data.football_data_org.asyncio.sleep", new_callable=AsyncMock):
            rows = await source.fetch_standings("PL", "2024")

        assert len(rows) == 2
        assert isinstance(rows[0], StandingRow)
        assert rows[0].position == 1
        assert rows[0].team.name == "Arsenal FC"
        assert rows[0].points == 48

    @pytest.mark.asyncio
    async def test_fetch_finished_matches_returns_match_data(
        self, source: FootballDataOrgSource
    ) -> None:
        """fetch_finished_matches should return a list of MatchData."""
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = FAKE_MATCHES_RESPONSE
        mock_response.raise_for_status = MagicMock()
        source._client.get.return_value = mock_response
        source._last_request_time = 0.0

        with patch("predictor.data.football_data_org.asyncio.sleep", new_callable=AsyncMock):
            matches = await source.fetch_finished_matches("PL", "2024")

        assert len(matches) == 1
        m = matches[0]
        assert isinstance(m, MatchData)
        assert m.home_team.name == "Arsenal FC"
        assert m.home_goals == 2
        assert m.away_goals == 1
        assert m.status == MatchStatus.FINISHED

    @pytest.mark.asyncio
    async def test_fetch_scheduled_fixtures_uses_scheduled_status(
        self, source: FootballDataOrgSource
    ) -> None:
        """fetch_scheduled_fixtures should pass status=SCHEDULED to the API."""
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"matches": []}
        mock_response.raise_for_status = MagicMock()
        source._client.get.return_value = mock_response
        source._last_request_time = 0.0

        with patch("predictor.data.football_data_org.asyncio.sleep", new_callable=AsyncMock):
            await source.fetch_scheduled_fixtures("PL", "2024")

        call_kwargs = source._client.get.call_args
        assert call_kwargs is not None
        params = call_kwargs.kwargs.get("params", {}) or call_kwargs.args[1] if len(call_kwargs.args) > 1 else {}
        # The params dict should include status=SCHEDULED
        # (exact key depends on call site)
        all_args = str(call_kwargs)
        assert "SCHEDULED" in all_args


# ---------------------------------------------------------------------------
# Integration tests (VCR cassettes — not yet implemented)
# ---------------------------------------------------------------------------


class TestFootballDataOrgIntegration:
    """Integration tests using recorded HTTP cassettes.

    TODO: Record cassettes with a real API key using::

        pytest --record-mode=once tests/test_data/test_football_data_org.py

    Then commit the cassette files (with X-Auth-Token filtered out).
    """

    @pytest.mark.skip(reason="TODO: Record VCR cassette first")
    @pytest.mark.asyncio
    async def test_fetch_pl_standings_live(self) -> None:
        """TODO: Use @pytest.mark.vcr decorator once cassette is recorded."""
        source = FootballDataOrgSource(api_key="REPLACE_WITH_REAL_KEY")
        rows = await source.fetch_standings("PL", "2024")
        assert len(rows) == 20  # Premier League has 20 teams
        await source.aclose()

    @pytest.mark.skip(reason="TODO: Record VCR cassette first")
    @pytest.mark.asyncio
    async def test_fetch_pl_finished_matches_live(self) -> None:
        """TODO: Verify real match data shape."""
        source = FootballDataOrgSource(api_key="REPLACE_WITH_REAL_KEY")
        matches = await source.fetch_finished_matches("PL", "2024")
        assert len(matches) > 0
        assert all(m.status == MatchStatus.FINISHED for m in matches)
        await source.aclose()
