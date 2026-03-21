"""Data source adapter for https://www.football-data.org/ v4 API.

Free-tier accounts are limited to ~10 calls/minute; we enforce a 2-second
inter-request delay to stay well within limits.

API documentation: https://docs.football-data.org/general/v4/index.html

TODO: Verify exact endpoint paths and response shapes against the v4 spec.
      The free tier only exposes a subset of competitions — check that
      the league codes you need are available on your subscription tier.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import structlog

from predictor.data.base import AbstractLeagueSource, MatchData, StandingRow, TeamData
from predictor.db.models import MatchStatus

logger = structlog.get_logger(__name__)

# Base URL for all API calls
_BASE_URL = "https://api.football-data.org/v4"

# Polite rate-limiting: minimum seconds between consecutive requests
_REQUEST_DELAY_SECONDS = 2.0

# Mapping from our internal league codes to football-data.org competition codes
# TODO: Extend this mapping as more leagues are added
LEAGUE_CODE_MAP: dict[str, str] = {
    "PL": "PL",       # Premier League
    "ELC": "ELC",     # Championship
    "EL1": "EL1",     # League One
    "EL2": "EL2",     # League Two
    "EC": "EC",       # European Championship (example)
}


class FootballDataOrgSource:
    """Fetches standings and fixtures from the football-data.org v4 REST API.

    Implements :class:`~predictor.data.base.AbstractLeagueSource`.

    Args:
        api_key: Your football-data.org API token.
        client: Optional pre-configured :class:`httpx.AsyncClient`. If not
            provided, a new client is created and must be closed by calling
            :meth:`aclose`.
    """

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "X-Auth-Token": api_key,
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        self._last_request_time: float = 0.0

    async def aclose(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _throttle(self) -> None:
        """Sleep if necessary to respect the inter-request delay."""
        import time

        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _REQUEST_DELAY_SECONDS:
            await asyncio.sleep(_REQUEST_DELAY_SECONDS - elapsed)
        self._last_request_time = time.monotonic()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """Send a GET request with rate limiting and basic error handling.

        Args:
            path: URL path relative to the base URL (e.g. ``'/competitions/PL/standings'``).
            params: Optional query parameters.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
        """
        await self._throttle()
        logger.debug("football_data_org_request", path=path, params=params)
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # AbstractLeagueSource implementation
    # ------------------------------------------------------------------

    async def fetch_standings(
        self, league_code: str, season: str
    ) -> list[StandingRow]:
        """Fetch standings from ``/competitions/{code}/standings``.

        Args:
            league_code: Internal league code (mapped to API code via
                :data:`LEAGUE_CODE_MAP`).
            season: Season year string, e.g. ``'2024'``.

        Returns:
            List of :class:`StandingRow` ordered by position.

        TODO: Handle multiple standing types (TOTAL, HOME, AWAY).
        TODO: Map the API's ``season.startDate`` to our ``'YYYY-YY'`` format.
        """
        api_code = LEAGUE_CODE_MAP.get(league_code, league_code)
        # TODO: Confirm whether the season param is a year (2024) or a string
        data = await self._get(
            f"/competitions/{api_code}/standings",
            params={"season": season},
        )

        rows: list[StandingRow] = []
        # TODO: The response structure is:
        #   data["standings"][0]["table"] for TOTAL standings
        #   Each table entry has: position, team{id,name,shortName}, ...
        standings_list = data.get("standings", [])
        total_table = next(
            (s["table"] for s in standings_list if s.get("type") == "TOTAL"),
            standings_list[0]["table"] if standings_list else [],
        )

        for entry in total_table:
            team_data = entry.get("team", {})
            rows.append(
                StandingRow(
                    team=TeamData(
                        name=team_data.get("name", "Unknown"),
                        short_name=team_data.get("shortName"),
                        external_id=str(team_data.get("id", "")),
                        crest_url=team_data.get("crest"),
                        website_url=team_data.get("website"),
                    ),
                    position=entry.get("position", 0),
                    played=entry.get("playedGames", 0),
                    won=entry.get("won", 0),
                    drawn=entry.get("draw", 0),
                    lost=entry.get("lost", 0),
                    goals_for=entry.get("goalsFor", 0),
                    goals_against=entry.get("goalsAgainst", 0),
                    points=entry.get("points", 0),
                    form=entry.get("form", ""),
                )
            )

        logger.info(
            "standings_fetched",
            league=league_code,
            season=season,
            count=len(rows),
        )
        return rows

    async def fetch_finished_matches(
        self, league_code: str, season: str
    ) -> list[MatchData]:
        """Fetch completed results from ``/competitions/{code}/matches``.

        Args:
            league_code: Internal league code.
            season: Season year string.

        Returns:
            List of finished :class:`MatchData`.

        TODO: Handle pagination for leagues with many matches.
        TODO: Map ``status`` values: FINISHED, IN_PLAY, PAUSED, etc.
        """
        api_code = LEAGUE_CODE_MAP.get(league_code, league_code)
        data = await self._get(
            f"/competitions/{api_code}/matches",
            params={"season": season, "status": "FINISHED"},
        )
        return self._parse_matches(data.get("matches", []), season)

    async def fetch_scheduled_fixtures(
        self, league_code: str, season: str
    ) -> list[MatchData]:
        """Fetch upcoming fixtures from ``/competitions/{code}/matches``.

        Args:
            league_code: Internal league code.
            season: Season year string.

        Returns:
            List of scheduled :class:`MatchData`.

        TODO: Add date-range filtering to avoid fetching thousands of rows.
        """
        api_code = LEAGUE_CODE_MAP.get(league_code, league_code)
        data = await self._get(
            f"/competitions/{api_code}/matches",
            params={"season": season, "status": "SCHEDULED"},
        )
        return self._parse_matches(data.get("matches", []), season)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_matches(self, raw_matches: list[dict], season: str) -> list[MatchData]:
        """Convert a list of raw API match dicts to :class:`MatchData` objects.

        Args:
            raw_matches: ``matches`` array from the API response.
            season: Season string to attach to each record.

        Returns:
            List of :class:`MatchData`.

        TODO: Handle extra time / penalty scores in ``score.extraTime`` /
              ``score.penalties``.
        TODO: Map all API status strings to :class:`MatchStatus` enum values.
        """
        results: list[MatchData] = []
        status_map: dict[str, MatchStatus] = {
            "FINISHED": MatchStatus.FINISHED,
            "SCHEDULED": MatchStatus.SCHEDULED,
            "TIMED": MatchStatus.SCHEDULED,
            "POSTPONED": MatchStatus.POSTPONED,
            "CANCELLED": MatchStatus.CANCELLED,
            "SUSPENDED": MatchStatus.CANCELLED,
        }

        for m in raw_matches:
            home_raw = m.get("homeTeam", {})
            away_raw = m.get("awayTeam", {})
            score = m.get("score", {})
            full_time = score.get("fullTime", {})

            raw_status = m.get("status", "SCHEDULED")
            match_status = status_map.get(raw_status, MatchStatus.SCHEDULED)

            # Parse kickoff datetime
            utc_date_str: str = m.get("utcDate", "")
            try:
                played_at = datetime.fromisoformat(
                    utc_date_str.replace("Z", "+00:00")
                )
            except ValueError:
                logger.warning("unparseable_date", raw=utc_date_str)
                played_at = datetime.now(tz=timezone.utc)

            results.append(
                MatchData(
                    season=season,
                    home_team=TeamData(
                        name=home_raw.get("name", "Unknown"),
                        short_name=home_raw.get("shortName"),
                        external_id=str(home_raw.get("id", "")),
                        crest_url=home_raw.get("crest"),
                        website_url=home_raw.get("website"),
                    ),
                    away_team=TeamData(
                        name=away_raw.get("name", "Unknown"),
                        short_name=away_raw.get("shortName"),
                        external_id=str(away_raw.get("id", "")),
                        crest_url=away_raw.get("crest"),
                        website_url=away_raw.get("website"),
                    ),
                    played_at=played_at,
                    status=match_status,
                    home_goals=full_time.get("home"),
                    away_goals=full_time.get("away"),
                    matchday=m.get("matchday"),
                    external_id=str(m.get("id", "")),
                )
            )

        logger.info("matches_parsed", count=len(results))
        return results
