"""Data source adapter for FA Full Time (https://fulltime.thefa.com/).

Uses httpx for async HTTP and BeautifulSoup4 for HTML parsing.
Polite scraping defaults: 2-second delay between requests and a realistic
browser User-Agent header.

NOTE: Web scraping is subject to site terms of service. Verify that your use
case is permitted before deploying. The FA Full Time site structure may change
without notice — check the TODO comments and update selectors accordingly.

TODO: Obtain the correct base URL and form POST parameters by inspecting the
      FA Full Time site with browser dev tools.
TODO: Replace all CSS/XPath selectors with ones validated against live HTML.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import structlog
from bs4 import BeautifulSoup, Tag

from predictor.data.base import AbstractLeagueSource, MatchData, StandingRow, TeamData
from predictor.db.models import MatchStatus

logger = structlog.get_logger(__name__)

# Base URL for FA Full Time
# TODO: Confirm the correct base URL — it may be regional/league-specific
_BASE_URL = "https://fulltime.thefa.com"

# Polite scraping delay between requests (seconds)
_REQUEST_DELAY_SECONDS = 2.0

# Realistic browser User-Agent to avoid bot-blocking
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Mapping from our league codes to FA Full Time division IDs / URL slugs
# TODO: Find the correct division IDs from the FA Full Time site
DIVISION_ID_MAP: dict[str, str] = {
    "NPL_W": "NPL_WEST",  # Northern Premier League West — placeholder
}


class FAFullTimeScraper:
    """Scrapes league standings and results from FA Full Time.

    Implements :class:`~predictor.data.base.AbstractLeagueSource`.

    Args:
        client: Optional pre-configured :class:`httpx.AsyncClient`. If not
            provided a new one is created. Call :meth:`aclose` when done.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-GB,en;q=0.5",
            },
            timeout=30.0,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Polite delay
    # ------------------------------------------------------------------

    async def _polite_get(self, path: str, params: dict | None = None) -> BeautifulSoup:
        """GET a page and return a parsed :class:`BeautifulSoup` tree.

        Applies a polite delay before each request.

        Args:
            path: URL path relative to the base URL.
            params: Optional query string parameters.

        Returns:
            Parsed HTML document.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
        """
        await asyncio.sleep(_REQUEST_DELAY_SECONDS)
        logger.debug("fa_fulltime_request", path=path, params=params)
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")

    # ------------------------------------------------------------------
    # AbstractLeagueSource implementation
    # ------------------------------------------------------------------

    async def fetch_standings(
        self, league_code: str, season: str
    ) -> list[StandingRow]:
        """Scrape the standings table for a league/season.

        Args:
            league_code: Internal league code (mapped via :data:`DIVISION_ID_MAP`).
            season: Season string (format depends on FA Full Time URL scheme).

        Returns:
            List of :class:`StandingRow` ordered by position.

        TODO: Identify the correct URL path and query parameters.
        TODO: Update :meth:`_parse_standings_table` with real CSS selectors.
        """
        division_id = DIVISION_ID_MAP.get(league_code, league_code)

        # TODO: Replace with the real URL path once confirmed
        path = f"/LeagueTableResults.aspx"
        params: dict[str, str] = {
            "divisionseason": division_id,
            "selectedSeason": season,
        }

        soup = await self._polite_get(path, params=params)
        rows = self._parse_standings_table(soup, season)
        logger.info("fa_standings_fetched", league=league_code, count=len(rows))
        return rows

    async def fetch_finished_matches(
        self, league_code: str, season: str
    ) -> list[MatchData]:
        """Scrape completed results.

        Args:
            league_code: Internal league code.
            season: Season string.

        Returns:
            List of finished :class:`MatchData`.

        TODO: Identify the correct URL and pagination mechanism.
        TODO: FA Full Time may require multiple page loads to get all results.
        """
        division_id = DIVISION_ID_MAP.get(league_code, league_code)

        # TODO: Replace with the real URL path and params once confirmed
        path = "/Results.aspx"
        params: dict[str, str] = {
            "divisionseason": division_id,
            "selectedSeason": season,
        }

        soup = await self._polite_get(path, params=params)
        matches = self._parse_results_table(soup, season)
        logger.info(
            "fa_results_fetched", league=league_code, season=season, count=len(matches)
        )
        return matches

    async def fetch_scheduled_fixtures(
        self, league_code: str, season: str
    ) -> list[MatchData]:
        """Scrape upcoming fixtures.

        Args:
            league_code: Internal league code.
            season: Season string.

        Returns:
            List of scheduled :class:`MatchData`.

        TODO: Identify the correct URL and params for fixtures.
        TODO: Handle the case where fixtures are not yet published.
        """
        division_id = DIVISION_ID_MAP.get(league_code, league_code)

        # TODO: Replace with the real URL path once confirmed
        path = "/Fixtures.aspx"
        params: dict[str, str] = {
            "divisionseason": division_id,
            "selectedSeason": season,
        }

        soup = await self._polite_get(path, params=params)
        matches = self._parse_fixtures_table(soup, season)
        logger.info(
            "fa_fixtures_fetched", league=league_code, season=season, count=len(matches)
        )
        return matches

    # ------------------------------------------------------------------
    # HTML parsing helpers
    # ------------------------------------------------------------------

    def _parse_standings_table(
        self, soup: BeautifulSoup, season: str
    ) -> list[StandingRow]:
        """Parse the standings HTML table into :class:`StandingRow` objects.

        Args:
            soup: Parsed HTML document.
            season: Season string to attach to each row.

        Returns:
            List of :class:`StandingRow`.

        TODO: Identify the correct table ID/class from the live site HTML.
        TODO: Map column indices to the correct fields (position may vary).
        """
        rows: list[StandingRow] = []

        # TODO: Replace selector with the real table identifier
        table = soup.find("table", {"id": "ctl00_ctl00_Body_Body_leagueTable"})
        if not isinstance(table, Tag):
            logger.warning("standings_table_not_found")
            return rows

        for tr in table.find_all("tr")[1:]:  # skip header row
            cells = tr.find_all("td")
            if len(cells) < 9:
                continue
            try:
                rows.append(
                    StandingRow(
                        team=TeamData(
                            name=cells[1].get_text(strip=True),
                        ),
                        position=int(cells[0].get_text(strip=True) or 0),
                        played=int(cells[2].get_text(strip=True) or 0),
                        won=int(cells[3].get_text(strip=True) or 0),
                        drawn=int(cells[4].get_text(strip=True) or 0),
                        lost=int(cells[5].get_text(strip=True) or 0),
                        goals_for=int(cells[6].get_text(strip=True) or 0),
                        goals_against=int(cells[7].get_text(strip=True) or 0),
                        points=int(cells[8].get_text(strip=True) or 0),
                    )
                )
            except (ValueError, IndexError) as exc:
                logger.warning("standings_row_parse_error", error=str(exc))

        return rows

    def _parse_results_table(
        self, soup: BeautifulSoup, season: str
    ) -> list[MatchData]:
        """Parse the results HTML table into finished :class:`MatchData` objects.

        Args:
            soup: Parsed HTML document.
            season: Season string.

        Returns:
            List of finished :class:`MatchData`.

        TODO: Identify the correct table and column layout.
        TODO: Parse the date string format used by FA Full Time.
        """
        matches: list[MatchData] = []

        # TODO: Replace selector with the real results table identifier
        table = soup.find("table", {"id": "ctl00_ctl00_Body_Body_resultsTable"})
        if not isinstance(table, Tag):
            logger.warning("results_table_not_found")
            return matches

        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) < 5:
                continue
            try:
                # TODO: Update column indices to match real HTML structure
                date_str = cells[0].get_text(strip=True)
                home_name = cells[1].get_text(strip=True)
                score_str = cells[2].get_text(strip=True)  # e.g. "2 - 1"
                away_name = cells[3].get_text(strip=True)

                home_goals, away_goals = self._parse_score(score_str)
                played_at = self._parse_date(date_str)

                matches.append(
                    MatchData(
                        season=season,
                        home_team=TeamData(name=home_name),
                        away_team=TeamData(name=away_name),
                        played_at=played_at,
                        status=MatchStatus.FINISHED,
                        home_goals=home_goals,
                        away_goals=away_goals,
                    )
                )
            except (ValueError, IndexError) as exc:
                logger.warning("result_row_parse_error", error=str(exc))

        return matches

    def _parse_fixtures_table(
        self, soup: BeautifulSoup, season: str
    ) -> list[MatchData]:
        """Parse the fixtures HTML table into scheduled :class:`MatchData` objects.

        Args:
            soup: Parsed HTML document.
            season: Season string.

        Returns:
            List of scheduled :class:`MatchData`.

        TODO: Identify the correct table and column layout.
        TODO: Handle missing kickoff times (some fixtures only have a date).
        """
        matches: list[MatchData] = []

        # TODO: Replace selector with the real fixtures table identifier
        table = soup.find("table", {"id": "ctl00_ctl00_Body_Body_fixturesTable"})
        if not isinstance(table, Tag):
            logger.warning("fixtures_table_not_found")
            return matches

        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) < 3:
                continue
            try:
                # TODO: Update column indices to match real HTML structure
                date_str = cells[0].get_text(strip=True)
                home_name = cells[1].get_text(strip=True)
                away_name = cells[2].get_text(strip=True)

                played_at = self._parse_date(date_str)

                matches.append(
                    MatchData(
                        season=season,
                        home_team=TeamData(name=home_name),
                        away_team=TeamData(name=away_name),
                        played_at=played_at,
                        status=MatchStatus.SCHEDULED,
                    )
                )
            except (ValueError, IndexError) as exc:
                logger.warning("fixture_row_parse_error", error=str(exc))

        return matches

    # ------------------------------------------------------------------
    # Low-level parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_score(score_str: str) -> tuple[int, int]:
        """Parse a score string like ``'2 - 1'`` into ``(2, 1)``.

        Args:
            score_str: Raw text from the score cell.

        Returns:
            Tuple of ``(home_goals, away_goals)``.

        Raises:
            ValueError: If the string cannot be parsed.

        TODO: Verify the exact format used on the FA Full Time site.
        """
        parts = score_str.replace(" ", "").split("-")
        if len(parts) != 2:
            raise ValueError(f"Cannot parse score: {score_str!r}")
        return int(parts[0]), int(parts[1])

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Parse a date string from the FA Full Time site into a UTC datetime.

        Args:
            date_str: Raw text from a date cell, e.g. ``'12/10/2024 15:00'``.

        Returns:
            Timezone-aware UTC :class:`datetime`.

        TODO: Confirm the exact date format used on the FA Full Time site.
        TODO: Handle UK daylight saving time correctly (site likely uses local time).
        """
        # TODO: Replace with the real format string, e.g. "%d/%m/%Y %H:%M"
        try:
            naive = datetime.strptime(date_str.strip(), "%d/%m/%Y %H:%M")
        except ValueError:
            # Fallback: date only
            naive = datetime.strptime(date_str.strip(), "%d/%m/%Y")
        # Assume UK local time — for accurate conversion use zoneinfo
        # TODO: Use zoneinfo.ZoneInfo("Europe/London") for DST-correct conversion
        return naive.replace(tzinfo=timezone.utc)
