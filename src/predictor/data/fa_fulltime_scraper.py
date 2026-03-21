"""Data source adapter for the Northern Premier League (NPL) API.

The NPL website (thenpl.co.uk) exposes a JSON REST API at api.thenpl.co.uk.
All requests require the ``X-TENANT-ID: npl`` header.

This module is named fa_fulltime_scraper for historical compatibility but
now uses the official NPL API — no HTML scraping required.

API details discovered via browser inspection of thenpl.co.uk:
  - Base: https://api.thenpl.co.uk
  - Auth: X-TENANT-ID: npl  (no API key needed)
  - Matches: GET /matches?competition={id}&limit=100&page={n}
  - Status values: "FullTime" (played), "NotKickedOff" (scheduled),
                   "Postponed", "Cancelled", "Abandoned"

NPL West Division ID: 67d7c7bb74247876e6dae40d
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from collections import defaultdict

import httpx
import structlog

from predictor.data.base import AbstractLeagueSource, MatchData, StandingRow, TeamData
from predictor.db.models import MatchStatus

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.thenpl.co.uk"
_LOGO_BASE_URL = "https://www.thenpl.co.uk/img/"
_TENANT_ID = "npl"
_PAGE_SIZE = 100
_REQUEST_DELAY_SECONDS = 1.0

# Competition IDs from the NPL API (discovered 2026-03)
COMPETITION_ID_MAP: dict[str, str] = {
    "NPL_W": "67d7c7bb74247876e6dae40d",   # Northern Premier League West
    "NPL_P": "67d7c7bb74247876e6dae40a",   # Northern Premier League Premier
    "NPL_E": "67d7c7bb74247876e6dae40b",   # Northern Premier League East
    "NPL_M": "67d7c7bb74247876e6dae40c",   # Northern Premier League Midlands
}

_STATUS_MAP: dict[str, MatchStatus] = {
    "FullTime": MatchStatus.FINISHED,
    "NotKickedOff": MatchStatus.SCHEDULED,
    "Postponed": MatchStatus.POSTPONED,
    "Cancelled": MatchStatus.CANCELLED,
    "Abandoned": MatchStatus.CANCELLED,
    "HalfTime": MatchStatus.SCHEDULED,   # In-play — treat as not yet finished
    "InProgress": MatchStatus.SCHEDULED,
}


def _build_crest_url(club: dict) -> str | None:
    """Extract and build full crest URL from an NPL API club object."""
    logo = club.get("logo")
    if logo:
        return _LOGO_BASE_URL + logo
    return None


def _extract_website_url(club: dict) -> str | None:
    """Extract the club website URL from an NPL API club object."""
    return club.get("website") or club.get("url") or None


class FAFullTimeScraper:
    """Fetches NPL standings and fixtures from the thenpl.co.uk JSON API.

    Despite the class name (kept for import compatibility), this adapter
    uses a JSON REST API — not HTML scraping.

    Implements :class:`~predictor.data.base.AbstractLeagueSource`.

    Args:
        client: Optional pre-configured :class:`httpx.AsyncClient`. If not
            provided, a new client is created. Call :meth:`aclose` when done.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._match_cache: dict[str, list[dict]] = {}
        self._client = client or httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "X-TENANT-ID": _TENANT_ID,
                "Accept": "application/json",
                "Origin": "https://www.thenpl.co.uk",
                "Referer": "https://www.thenpl.co.uk/",
            },
            timeout=30.0,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def _get_all_matches(self, league_code: str) -> list[dict]:
        """Fetch all matches for a competition across all pages.

        Args:
            league_code: Internal league code mapped via :data:`COMPETITION_ID_MAP`.

        Returns:
            List of raw match dicts from the API.
        """
        if league_code in self._match_cache:
            return self._match_cache[league_code]

        competition_id = COMPETITION_ID_MAP.get(league_code, league_code)
        all_matches: list[dict] = []
        page = 1

        while True:
            await asyncio.sleep(_REQUEST_DELAY_SECONDS)
            response = await self._client.get(
                "/matches",
                params={
                    "competition": competition_id,
                    "limit": _PAGE_SIZE,
                    "page": page,
                },
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            all_matches.extend(items)

            pagination = data.get("pagination", {})
            if not pagination.get("hasNextPage") or len(items) < _PAGE_SIZE:
                # hasNextPage appears to be unreliable in this API —
                # stop when we get a partial page
                if len(items) < _PAGE_SIZE:
                    break
                page += 1
            else:
                page += 1

            if page > 50:  # safety limit
                break

        # Deduplicate by API match ID (the API sometimes returns duplicate entries)
        seen: set[str] = set()
        unique: list[dict] = []
        for m in all_matches:
            mid = m.get("_id") or m.get("id") or str(m)
            if mid not in seen:
                seen.add(mid)
                unique.append(m)

        logger.info("npl_matches_fetched", league=league_code, count=len(unique))
        self._match_cache[league_code] = unique
        return unique

    async def fetch_standings(
        self, league_code: str, season: str
    ) -> list[StandingRow]:
        """Compute standings from all finished matches.

        The NPL API does not expose a standalone standings endpoint, so we
        derive standings by aggregating all FINISHED match results.

        Args:
            league_code: Internal league code.
            season: Season string (used for logging only; API returns all
                matches for the active season).

        Returns:
            List of :class:`StandingRow` ordered by points descending.
        """
        all_matches = await self._get_all_matches(league_code)
        finished = [m for m in all_matches if m.get("status") == "FullTime"]

        # Accumulate standings
        stats: dict[str, dict] = defaultdict(lambda: {
            "played": 0, "won": 0, "drawn": 0, "lost": 0,
            "goals_for": 0, "goals_against": 0, "points": 0,
        })
        crest_map: dict[str, str | None] = {}
        website_map: dict[str, str | None] = {}

        for m in finished:
            home_name = m["homeTeam"]["club"]["fullName"]
            away_name = m["awayTeam"]["club"]["fullName"]
            if home_name not in crest_map:
                crest_map[home_name] = _build_crest_url(m["homeTeam"]["club"])
                website_map[home_name] = _extract_website_url(m["homeTeam"]["club"])
            if away_name not in crest_map:
                crest_map[away_name] = _build_crest_url(m["awayTeam"]["club"])
                website_map[away_name] = _extract_website_url(m["awayTeam"]["club"])
            score = m.get("score", {}).get("current", {})
            hg = score.get("home", 0) or 0
            ag = score.get("away", 0) or 0

            stats[home_name]["played"] += 1
            stats[home_name]["goals_for"] += hg
            stats[home_name]["goals_against"] += ag
            stats[away_name]["played"] += 1
            stats[away_name]["goals_for"] += ag
            stats[away_name]["goals_against"] += hg

            if hg > ag:
                stats[home_name]["won"] += 1
                stats[home_name]["points"] += 3
                stats[away_name]["lost"] += 1
            elif hg < ag:
                stats[away_name]["won"] += 1
                stats[away_name]["points"] += 3
                stats[home_name]["lost"] += 1
            else:
                stats[home_name]["drawn"] += 1
                stats[home_name]["points"] += 1
                stats[away_name]["drawn"] += 1
                stats[away_name]["points"] += 1

        # Sort by points desc, then goal difference desc, then goals for desc
        sorted_teams = sorted(
            stats.items(),
            key=lambda kv: (
                -kv[1]["points"],
                -(kv[1]["goals_for"] - kv[1]["goals_against"]),
                -kv[1]["goals_for"],
            ),
        )

        rows = [
            StandingRow(
                team=TeamData(name=name, crest_url=crest_map.get(name), website_url=website_map.get(name)),
                position=idx + 1,
                played=s["played"],
                won=s["won"],
                drawn=s["drawn"],
                lost=s["lost"],
                goals_for=s["goals_for"],
                goals_against=s["goals_against"],
                points=s["points"],
            )
            for idx, (name, s) in enumerate(sorted_teams)
        ]

        logger.info("npl_standings_computed", league=league_code, teams=len(rows))
        return rows

    async def fetch_finished_matches(
        self, league_code: str, season: str
    ) -> list[MatchData]:
        """Fetch all completed results.

        Args:
            league_code: Internal league code.
            season: Season string.

        Returns:
            List of finished :class:`MatchData`.
        """
        all_matches = await self._get_all_matches(league_code)
        return [
            self._parse_match(m, season)
            for m in all_matches
            if m.get("status") == "FullTime"
        ]

    async def fetch_scheduled_fixtures(
        self, league_code: str, season: str
    ) -> list[MatchData]:
        """Fetch upcoming fixtures.

        Args:
            league_code: Internal league code.
            season: Season string.

        Returns:
            List of scheduled :class:`MatchData`.
        """
        all_matches = await self._get_all_matches(league_code)
        scheduled_statuses = {"NotKickedOff", "Postponed"}
        return [
            self._parse_match(m, season)
            for m in all_matches
            if m.get("status") in scheduled_statuses
        ]

    def _parse_match(self, raw: dict, season: str) -> MatchData:
        """Convert a raw API match dict to :class:`MatchData`.

        Args:
            raw: Single match dict from the API.
            season: Season string to attach.

        Returns:
            :class:`MatchData` instance.
        """
        home_name = raw["homeTeam"]["club"]["fullName"]
        away_name = raw["awayTeam"]["club"]["fullName"]

        score = raw.get("score", {}).get("current", {})
        status_str = raw.get("status", "NotKickedOff")
        match_status = _STATUS_MAP.get(status_str, MatchStatus.SCHEDULED)

        home_goals: int | None = None
        away_goals: int | None = None
        if match_status == MatchStatus.FINISHED:
            home_goals = score.get("home")
            away_goals = score.get("away")

        # Parse date — API returns "YYYY-MM-DD"
        date_str: str = raw.get("date") or ""
        try:
            played_at = datetime.strptime(date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            played_at = datetime.now(tz=timezone.utc)
            logger.warning("npl_unparseable_date", raw=date_str, match_id=raw.get("_id"))

        return MatchData(
            season=season,
            home_team=TeamData(
                name=home_name,
                external_id=raw.get("_id"),
                crest_url=_build_crest_url(raw["homeTeam"]["club"]),
                website_url=_extract_website_url(raw["homeTeam"]["club"]),
            ),
            away_team=TeamData(
                name=away_name,
                crest_url=_build_crest_url(raw["awayTeam"]["club"]),
                website_url=_extract_website_url(raw["awayTeam"]["club"]),
            ),
            played_at=played_at,
            status=match_status,
            home_goals=home_goals,
            away_goals=away_goals,
            external_id=raw.get("_id"),
        )
