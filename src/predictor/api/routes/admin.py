"""Admin API routes for FootyPredict.

All routes in this module require a valid ``X-Admin-Key`` header.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select

from predictor.api.deps import AdminKeyVerified, DbSession
from predictor.config import get_settings
from predictor.db.models import DataSource, League, MatchStatus
from predictor.db.repos.match import MatchRepository
from predictor.db.repos.team import TeamRepository

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class SyncResponse(BaseModel):
    """Result summary for a data sync operation."""

    message: str
    leagues_synced: int
    matches_upserted: int
    teams_created: int


class ImportCSVResponse(BaseModel):
    """Result of a CSV import operation."""

    message: str
    rows_processed: int
    rows_failed: int
    errors: list[str]


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post("/sync", response_model=SyncResponse)
async def sync_all_leagues(
    db: DbSession,
    _: AdminKeyVerified,
) -> SyncResponse:
    """Trigger an immediate data sync from all configured data sources.

    Fetches standings, finished results, and upcoming fixtures for every
    league in the database, then rebuilds TeamSeason aggregates and ELO.

    Returns:
        :class:`SyncResponse` with a summary of what was synced.
    """
    from predictor.data.football_data_org import FootballDataOrgSource
    from predictor.data.fa_fulltime_scraper import FAFullTimeScraper
    from predictor.data.importer import DataImporter
    from predictor.engine.team_season import rebuild_team_seasons

    logger.info("admin_sync_triggered")
    settings = get_settings()

    result = await db.execute(select(League).order_by(League.tier))
    leagues = list(result.scalars().all())

    total_matches = 0
    total_teams = 0
    leagues_synced = 0

    for league in leagues:
        try:
            if league.data_source == DataSource.API_FOOTBALL_DATA:
                source = FootballDataOrgSource(api_key=settings.FOOTBALL_DATA_API_KEY)
            else:
                source = FAFullTimeScraper()

            team_repo = TeamRepository(db)
            match_repo = MatchRepository(db)
            importer = DataImporter(source=source, team_repo=team_repo, match_repo=match_repo)

            stats = await importer.sync_league(league)
            total_teams += stats["teams_created"]
            total_matches += stats["matches_upserted"]

            # Flush pending inserts so rebuild_team_seasons can query them
            await db.flush()
            # Rebuild TeamSeason aggregates + ELO from match data
            await rebuild_team_seasons(db, league)
            await db.commit()

            await source.aclose()
            leagues_synced += 1
            logger.info("league_sync_complete", league=league.code, **stats)

        except Exception as exc:
            logger.error("league_sync_failed", league=league.code, error=str(exc))
            await db.rollback()

    return SyncResponse(
        message=f"Sync complete. {leagues_synced}/{len(leagues)} leagues synced.",
        leagues_synced=leagues_synced,
        matches_upserted=total_matches,
        teams_created=total_teams,
    )


@router.post("/import-csv", response_model=ImportCSVResponse)
async def import_csv(
    file: UploadFile,
    db: DbSession,
    _: AdminKeyVerified,
) -> ImportCSVResponse:
    """Import match results from an uploaded CSV file.

    Expected CSV columns (header row required):
    ``date,home_team,away_team,home_goals,away_goals,league_code,season``

    Date format: ``DD/MM/YYYY``

    Args:
        file: Uploaded CSV file (``multipart/form-data``).

    Returns:
        :class:`ImportCSVResponse` with processing summary.

    Raises:
        HTTPException 400: If the file is not a valid CSV or has wrong headers.
    """
    from predictor.engine.team_season import rebuild_team_seasons

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are accepted.",
        )

    content = await file.read()
    try:
        text = content.decode("utf-8-sig")  # handle BOM from Excel exports
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    required_headers = {
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "league_code",
        "season",
    }
    fieldnames = set(reader.fieldnames or [])
    missing = required_headers - fieldnames
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV is missing required columns: {sorted(missing)}",
        )

    # Cache leagues by code to avoid repeated DB queries
    league_cache: dict[str, League | None] = {}
    affected_leagues: set[str] = set()

    team_repo = TeamRepository(db)
    match_repo = MatchRepository(db)

    rows_processed = 0
    rows_failed = 0
    errors: list[str] = []

    for line_num, row in enumerate(reader, start=2):
        try:
            league_code = row["league_code"].strip()
            season = row["season"].strip()

            # Resolve league (cached)
            if league_code not in league_cache:
                lg_result = await db.execute(
                    select(League).where(League.code == league_code)
                )
                league_cache[league_code] = lg_result.scalar_one_or_none()

            league = league_cache[league_code]
            if league is None:
                raise ValueError(f"Unknown league code {league_code!r}")

            # Parse date
            date_str = row["date"].strip()
            try:
                played_at = datetime.strptime(date_str, "%d/%m/%Y").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                played_at = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )

            home_team, _ = await team_repo.get_or_create(
                league_id=league.id,
                name=row["home_team"].strip(),
            )
            away_team, _ = await team_repo.get_or_create(
                league_id=league.id,
                name=row["away_team"].strip(),
            )

            await match_repo.upsert({
                "league_id": league.id,
                "season": season,
                "home_team_id": home_team.id,
                "away_team_id": away_team.id,
                "played_at": played_at,
                "status": MatchStatus.FINISHED,
                "home_goals": int(row["home_goals"]),
                "away_goals": int(row["away_goals"]),
                "matchday": None,
            })

            affected_leagues.add(league.id)
            rows_processed += 1

        except Exception as exc:
            rows_failed += 1
            errors.append(f"Row {line_num}: {exc}")
            if len(errors) >= 50:
                errors.append("(too many errors — truncated)")
                break

    # Rebuild TeamSeason for each affected league
    for league_id in affected_leagues:
        lg_result = await db.execute(select(League).where(League.id == league_id))
        league = lg_result.scalar_one_or_none()
        if league:
            await rebuild_team_seasons(db, league)

    await db.commit()

    logger.info(
        "csv_import_completed",
        processed=rows_processed,
        failed=rows_failed,
    )

    return ImportCSVResponse(
        message=f"Imported {rows_processed} rows ({rows_failed} failed).",
        rows_processed=rows_processed,
        rows_failed=rows_failed,
        errors=errors,
    )
