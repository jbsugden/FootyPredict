"""Tests for the NPL API adapter (FAFullTimeScraper).

Real HTTP calls are not made in unit tests. Tests mock httpx responses
to test the parsing logic in isolation.

Note: The class is named FAFullTimeScraper for historical compatibility but
now uses the thenpl.co.uk JSON API.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from predictor.data.fa_fulltime_scraper import (
    COMPETITION_ID_MAP,
    FAFullTimeScraper,
    _STATUS_MAP,
)
from predictor.data.base import MatchData, StandingRow
from predictor.db.models import MatchStatus


# ---------------------------------------------------------------------------
# Sample API response data
# ---------------------------------------------------------------------------

SAMPLE_MATCH_FINISHED = {
    "_id": "match_001",
    "homeTeam": {"club": {"fullName": "Glossop North End"}},
    "awayTeam": {"club": {"fullName": "Mossley AFC"}},
    "score": {"current": {"home": 2, "away": 1}},
    "status": "FullTime",
    "date": "2025-01-15",
}

SAMPLE_MATCH_SCHEDULED = {
    "_id": "match_002",
    "homeTeam": {"club": {"fullName": "Colne FC"}},
    "awayTeam": {"club": {"fullName": "Squires Gate"}},
    "score": {"current": {}},
    "status": "NotKickedOff",
    "date": "2025-03-20",
}

SAMPLE_MATCH_POSTPONED = {
    "_id": "match_003",
    "homeTeam": {"club": {"fullName": "Avro FC"}},
    "awayTeam": {"club": {"fullName": "Irlam FC"}},
    "score": {"current": {}},
    "status": "Postponed",
    "date": "2025-02-01",
}


def _make_paginated_response(items: list[dict], has_next: bool = False) -> dict:
    return {
        "items": items,
        "pagination": {"hasNextPage": has_next},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseMatch:
    def test_finished_match_parsed_correctly(self) -> None:
        scraper = FAFullTimeScraper(client=AsyncMock(spec=httpx.AsyncClient))
        result = scraper._parse_match(SAMPLE_MATCH_FINISHED, "2024-25")

        assert isinstance(result, MatchData)
        assert result.home_team.name == "Glossop North End"
        assert result.away_team.name == "Mossley AFC"
        assert result.home_goals == 2
        assert result.away_goals == 1
        assert result.status == MatchStatus.FINISHED
        assert result.season == "2024-25"

    def test_scheduled_match_has_no_goals(self) -> None:
        scraper = FAFullTimeScraper(client=AsyncMock(spec=httpx.AsyncClient))
        result = scraper._parse_match(SAMPLE_MATCH_SCHEDULED, "2024-25")

        assert result.status == MatchStatus.SCHEDULED
        assert result.home_goals is None
        assert result.away_goals is None

    def test_postponed_match_status(self) -> None:
        scraper = FAFullTimeScraper(client=AsyncMock(spec=httpx.AsyncClient))
        result = scraper._parse_match(SAMPLE_MATCH_POSTPONED, "2024-25")

        assert result.status == MatchStatus.POSTPONED

    def test_date_parsed_as_utc(self) -> None:
        scraper = FAFullTimeScraper(client=AsyncMock(spec=httpx.AsyncClient))
        result = scraper._parse_match(SAMPLE_MATCH_FINISHED, "2024-25")

        assert result.played_at.year == 2025
        assert result.played_at.month == 1
        assert result.played_at.day == 15
        assert result.played_at.tzinfo is not None

    def test_external_id_captured(self) -> None:
        scraper = FAFullTimeScraper(client=AsyncMock(spec=httpx.AsyncClient))
        result = scraper._parse_match(SAMPLE_MATCH_FINISHED, "2024-25")
        assert result.external_id == "match_001"


class TestFetchStandings:
    @pytest.mark.asyncio
    async def test_computes_standings_from_finished_matches(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        # Two matches: A beats B 2-1, then B beats A 3-0
        matches = [
            {
                "_id": "m1",
                "homeTeam": {"club": {"fullName": "Team A"}},
                "awayTeam": {"club": {"fullName": "Team B"}},
                "score": {"current": {"home": 2, "away": 1}},
                "status": "FullTime",
                "date": "2025-01-01",
            },
            {
                "_id": "m2",
                "homeTeam": {"club": {"fullName": "Team B"}},
                "awayTeam": {"club": {"fullName": "Team A"}},
                "score": {"current": {"home": 3, "away": 0}},
                "status": "FullTime",
                "date": "2025-01-08",
            },
        ]
        response = MagicMock(spec=httpx.Response)
        response.json.return_value = _make_paginated_response(matches, has_next=False)
        response.raise_for_status = MagicMock()
        mock_client.get.return_value = response

        scraper = FAFullTimeScraper(client=mock_client)
        with patch("predictor.data.fa_fulltime_scraper.asyncio.sleep", new_callable=AsyncMock):
            rows = await scraper.fetch_standings("NPL_W", "2024-25")

        assert len(rows) == 2
        assert isinstance(rows[0], StandingRow)
        # Team B has 3 pts (1 win), Team A has 3 pts (1 win)
        # But Team B has better GD: +2 (4-2) vs Team A GD: +0 (2-4) => wait
        # Team A: GF=2+0=2, GA=1+3=4, GD=-2, Pts=3
        # Team B: GF=1+3=4, GA=2+0=2, GD=+2, Pts=3
        assert rows[0].team.name == "Team B"  # Better GD
        assert rows[0].points == 3

    @pytest.mark.asyncio
    async def test_scheduled_matches_excluded_from_standings(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        matches = [SAMPLE_MATCH_FINISHED, SAMPLE_MATCH_SCHEDULED]
        response = MagicMock(spec=httpx.Response)
        response.json.return_value = _make_paginated_response(matches, has_next=False)
        response.raise_for_status = MagicMock()
        mock_client.get.return_value = response

        scraper = FAFullTimeScraper(client=mock_client)
        with patch("predictor.data.fa_fulltime_scraper.asyncio.sleep", new_callable=AsyncMock):
            rows = await scraper.fetch_standings("NPL_W", "2024-25")

        # Only finished match teams should appear
        team_names = {r.team.name for r in rows}
        assert "Glossop North End" in team_names
        assert "Colne FC" not in team_names


class TestDeduplication:
    @pytest.mark.asyncio
    async def test_duplicate_matches_are_deduplicated(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        # Same match appears twice
        matches = [SAMPLE_MATCH_FINISHED, SAMPLE_MATCH_FINISHED]
        response = MagicMock(spec=httpx.Response)
        response.json.return_value = _make_paginated_response(matches, has_next=False)
        response.raise_for_status = MagicMock()
        mock_client.get.return_value = response

        scraper = FAFullTimeScraper(client=mock_client)
        with patch("predictor.data.fa_fulltime_scraper.asyncio.sleep", new_callable=AsyncMock):
            finished = await scraper.fetch_finished_matches("NPL_W", "2024-25")

        # Should be deduplicated to 1
        assert len(finished) == 1


class TestStatusMapping:
    def test_all_known_statuses_are_mapped(self) -> None:
        for status_str in ["FullTime", "NotKickedOff", "Postponed", "Cancelled", "Abandoned"]:
            assert status_str in _STATUS_MAP

    def test_competition_id_map_has_npl_west(self) -> None:
        assert "NPL_W" in COMPETITION_ID_MAP
