"""Data import orchestrator.

:class:`DataImporter` ties together a data source, the team and match
repositories, and a validation step into a single reusable fetch/upsert
pipeline.
"""

from __future__ import annotations

import structlog

from predictor.data.base import AbstractLeagueSource, MatchData
from predictor.db.models import League
from predictor.db.repos.match import MatchRepository
from predictor.db.repos.team import TeamRepository

logger = structlog.get_logger(__name__)


class DataImporter:
    """Orchestrates fetch -> validate -> upsert for a single league.

    Args:
        source: A data source implementing :class:`AbstractLeagueSource`.
        team_repo: Repository for team persistence.
        match_repo: Repository for match persistence.
    """

    def __init__(
        self,
        source: AbstractLeagueSource,
        team_repo: TeamRepository,
        match_repo: MatchRepository,
    ) -> None:
        self._source = source
        self._team_repo = team_repo
        self._match_repo = match_repo

    async def sync_league(self, league: League) -> dict[str, int]:
        """Full sync: fetch standings, results, and fixtures then upsert all.

        Steps:
        1. Fetch standings (to ensure all teams exist in the DB).
        2. Fetch finished matches and upsert results.
        3. Fetch scheduled fixtures and upsert upcoming matches.

        Args:
            league: The :class:`~predictor.db.models.League` to sync.

        Returns:
            A summary dict with keys ``teams_created``, ``matches_upserted``.
        """
        log = logger.bind(league=league.code, season=league.current_season)
        log.info("sync_started")

        stats = {"teams_created": 0, "matches_upserted": 0}

        # ------------------------------------------------------------------
        # 1. Standings — ensure all teams are present
        # ------------------------------------------------------------------
        standings = await self._source.fetch_standings(
            league.code, league.current_season
        )
        for row in standings:
            _, created = await self._team_repo.get_or_create(
                league_id=league.id,
                name=row.team.name,
                short_name=row.team.short_name,
                external_id=row.team.external_id,
                crest_url=row.team.crest_url,
                website_url=row.team.website_url,
            )
            if created:
                stats["teams_created"] += 1

        # ------------------------------------------------------------------
        # 2. Finished matches
        # ------------------------------------------------------------------
        finished = await self._source.fetch_finished_matches(
            league.code, league.current_season
        )
        for match_data in finished:
            await self._upsert_match(league, match_data)
            stats["matches_upserted"] += 1

        # ------------------------------------------------------------------
        # 3. Scheduled fixtures
        # ------------------------------------------------------------------
        scheduled = await self._source.fetch_scheduled_fixtures(
            league.code, league.current_season
        )
        for match_data in scheduled:
            await self._upsert_match(league, match_data)
            stats["matches_upserted"] += 1

        log.info("sync_completed", **stats)
        return stats

    async def _upsert_match(self, league: League, match_data: MatchData) -> None:
        """Resolve team IDs then upsert a single match record.

        Args:
            league: Parent league (provides ``id`` for FK and ``current_season``).
            match_data: Parsed match from the data source.
        """
        home_team, _ = await self._team_repo.get_or_create(
            league_id=league.id,
            name=match_data.home_team.name,
            short_name=match_data.home_team.short_name,
            external_id=match_data.home_team.external_id,
            crest_url=match_data.home_team.crest_url,
            website_url=match_data.home_team.website_url,
        )
        away_team, _ = await self._team_repo.get_or_create(
            league_id=league.id,
            name=match_data.away_team.name,
            short_name=match_data.away_team.short_name,
            external_id=match_data.away_team.external_id,
            crest_url=match_data.away_team.crest_url,
            website_url=match_data.away_team.website_url,
        )

        upsert_data = {
            "league_id": league.id,
            "season": match_data.season,
            "home_team_id": home_team.id,
            "away_team_id": away_team.id,
            "played_at": match_data.played_at,
            "status": match_data.status,
            "home_goals": match_data.home_goals,
            "away_goals": match_data.away_goals,
            "matchday": match_data.matchday,
        }
        if match_data.external_id:
            upsert_data["external_id"] = match_data.external_id

        await self._match_repo.upsert(upsert_data)

    @staticmethod
    def _validate_match(match_data: MatchData) -> list[str]:
        """Return a list of validation error strings (empty = valid).

        Args:
            match_data: The match to validate.

        Returns:
            List of human-readable error strings. An empty list means the
            record is valid and can be safely upserted.

        TODO: Add more validation rules as edge cases are discovered:
              - played_at in the future for FINISHED matches
              - negative goal counts
              - home_team == away_team
        """
        errors: list[str] = []
        if not match_data.home_team.name.strip():
            errors.append("home_team name is empty")
        if not match_data.away_team.name.strip():
            errors.append("away_team name is empty")
        if match_data.played_at is None:
            errors.append("played_at is None")
        return errors
