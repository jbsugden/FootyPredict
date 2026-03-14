"""Admin API routes for FootyPredict.

All routes in this module require a valid ``X-Admin-Key`` header.
"""

from __future__ import annotations

import csv
import io

import structlog
from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

from predictor.api.deps import AdminKeyVerified, DbSession

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

    This is the same operation that runs nightly via the scheduler.  Useful
    for manually forcing a refresh without waiting for the cron job.

    Returns:
        :class:`SyncResponse` with a summary of what was synced.

    TODO: Load all active leagues from the DB.
    TODO: Dispatch DataImporter.sync_league() per league.
    TODO: Return accurate counts from the import stats.
    """
    logger.info("admin_sync_triggered")

    # TODO: Implement real sync logic
    # from sqlalchemy import select
    # from predictor.db.models import League
    # from predictor.data.importer import DataImporter
    # ...

    return SyncResponse(
        message="Sync complete (placeholder — real sync not yet implemented).",
        leagues_synced=0,
        matches_upserted=0,
        teams_created=0,
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

    TODO: Validate against the DB's known league codes.
    TODO: Look up / create teams and upsert matches.
    TODO: Trigger ELO recalculation after bulk import.
    """
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

    rows_processed = 0
    rows_failed = 0
    errors: list[str] = []

    for line_num, row in enumerate(reader, start=2):
        try:
            # TODO: Parse date, look up league, upsert match record
            # from datetime import datetime
            # played_at = datetime.strptime(row["date"], "%d/%m/%Y")
            # ...
            rows_processed += 1
        except Exception as exc:
            rows_failed += 1
            errors.append(f"Row {line_num}: {exc}")
            if len(errors) >= 50:
                errors.append("(too many errors — truncated)")
                break

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
