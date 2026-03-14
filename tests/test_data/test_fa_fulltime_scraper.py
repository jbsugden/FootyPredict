"""Tests for the FA Full Time HTML scraper.

Real HTTP calls are not made in unit tests. Tests use saved HTML fixture files
(stored under tests/fixtures/html/) to test the parsing logic in isolation.

TODO: Save real FA Full Time HTML pages to tests/fixtures/html/:
      - fa_fulltime_standings.html  — a real standings page HTML
      - fa_fulltime_results.html    — a real results page HTML
      - fa_fulltime_fixtures.html   — a real fixtures page HTML

TODO: Update the CSS selectors in FAFullTimeScraper once you have the real
      HTML and have identified the correct table IDs / class names.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from predictor.data.fa_fulltime_scraper import FAFullTimeScraper
from predictor.data.base import MatchData, StandingRow
from predictor.db.models import MatchStatus

# ---------------------------------------------------------------------------
# Path to saved HTML fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "html"


def _load_fixture(filename: str) -> str:
    """Load a saved HTML fixture file.

    TODO: Create the fixtures directory and save real FA Full Time HTML pages.
    """
    fixture_path = FIXTURES_DIR / filename
    if not fixture_path.exists():
        pytest.skip(f"HTML fixture not found: {fixture_path}. TODO: Save real HTML page.")
    return fixture_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit tests for parsing helpers
# ---------------------------------------------------------------------------


class TestParseScore:
    """Tests for FAFullTimeScraper._parse_score."""

    def test_standard_score_parses_correctly(self) -> None:
        assert FAFullTimeScraper._parse_score("2 - 1") == (2, 1)

    def test_score_without_spaces_parses_correctly(self) -> None:
        assert FAFullTimeScraper._parse_score("3-0") == (3, 0)

    def test_draw_parses_correctly(self) -> None:
        assert FAFullTimeScraper._parse_score("1 - 1") == (1, 1)

    def test_zero_zero_parses_correctly(self) -> None:
        assert FAFullTimeScraper._parse_score("0 - 0") == (0, 0)

    def test_invalid_score_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            FAFullTimeScraper._parse_score("not a score")

    def test_score_with_extra_spaces_parses_correctly(self) -> None:
        """Leading/trailing whitespace should not cause failure."""
        assert FAFullTimeScraper._parse_score("  2  -  1  ") == (2, 1)


class TestParseDate:
    """Tests for FAFullTimeScraper._parse_date."""

    def test_date_with_time_parses_correctly(self) -> None:
        result = FAFullTimeScraper._parse_date("12/10/2024 15:00")
        assert result.year == 2024
        assert result.month == 10
        assert result.day == 12
        assert result.hour == 15

    def test_date_only_parses_correctly(self) -> None:
        result = FAFullTimeScraper._parse_date("25/12/2024")
        assert result.year == 2024
        assert result.month == 12
        assert result.day == 25

    def test_result_is_timezone_aware(self) -> None:
        import datetime

        result = FAFullTimeScraper._parse_date("01/01/2025 12:00")
        assert result.tzinfo is not None


# ---------------------------------------------------------------------------
# Tests using saved HTML fixtures (skip if fixtures not present)
# ---------------------------------------------------------------------------


class TestParseStandingsTable:
    """Tests for FAFullTimeScraper._parse_standings_table using saved HTML.

    TODO: Save a real FA Full Time standings page to
          tests/fixtures/html/fa_fulltime_standings.html
          and update the CSS selector in the scraper to match.
    """

    @pytest.mark.skip(reason="TODO: Save HTML fixture and update CSS selectors")
    def test_real_standings_html_returns_correct_row_count(self) -> None:
        html = _load_fixture("fa_fulltime_standings.html")
        soup = BeautifulSoup(html, "lxml")
        scraper = FAFullTimeScraper()
        rows = scraper._parse_standings_table(soup, "2024-25")
        # TODO: Update expected count to match the real division size
        assert len(rows) > 0

    @pytest.mark.skip(reason="TODO: Save HTML fixture and update CSS selectors")
    def test_real_standings_html_first_row_has_correct_fields(self) -> None:
        html = _load_fixture("fa_fulltime_standings.html")
        soup = BeautifulSoup(html, "lxml")
        scraper = FAFullTimeScraper()
        rows = scraper._parse_standings_table(soup, "2024-25")
        assert isinstance(rows[0], StandingRow)
        assert rows[0].position == 1
        assert rows[0].points > 0


class TestParseResultsTable:
    """Tests for FAFullTimeScraper._parse_results_table using saved HTML.

    TODO: Save a real FA Full Time results page to
          tests/fixtures/html/fa_fulltime_results.html
    """

    @pytest.mark.skip(reason="TODO: Save HTML fixture and update CSS selectors")
    def test_real_results_html_returns_match_data(self) -> None:
        html = _load_fixture("fa_fulltime_results.html")
        soup = BeautifulSoup(html, "lxml")
        scraper = FAFullTimeScraper()
        matches = scraper._parse_results_table(soup, "2024-25")
        assert len(matches) > 0
        assert all(isinstance(m, MatchData) for m in matches)
        assert all(m.status == MatchStatus.FINISHED for m in matches)

    @pytest.mark.skip(reason="TODO: Save HTML fixture and update CSS selectors")
    def test_real_results_html_goals_are_non_negative(self) -> None:
        html = _load_fixture("fa_fulltime_results.html")
        soup = BeautifulSoup(html, "lxml")
        scraper = FAFullTimeScraper()
        matches = scraper._parse_results_table(soup, "2024-25")
        for m in matches:
            assert m.home_goals is not None and m.home_goals >= 0
            assert m.away_goals is not None and m.away_goals >= 0


class TestPoliteDelay:
    """Verify that the scraper enforces a polite delay between requests."""

    @pytest.mark.asyncio
    async def test_polite_get_calls_sleep(self) -> None:
        """The scraper should call asyncio.sleep before each request."""
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        scraper = FAFullTimeScraper(client=mock_client)

        with patch(
            "predictor.data.fa_fulltime_scraper.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await scraper._polite_get("/test")
            mock_sleep.assert_called_once()
            # Verify the delay is at least 1 second
            delay_arg = mock_sleep.call_args[0][0]
            assert delay_arg >= 1.0
