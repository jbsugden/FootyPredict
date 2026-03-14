"""TeamSeason rebuild utility.

After a data sync, this module recomputes the aggregated season statistics
(P/W/D/L/GF/GA/Pts), recent form string, and ELO ratings for every team
from the raw Match records, then upserts the TeamSeason table.

Call :func:`rebuild_team_seasons` after any data sync or CSV import.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from predictor.db.models import League, Match, MatchStatus, TeamSeason
from predictor.engine.elo import DEFAULT_INITIAL_RATING, EloCalculator, K_NON_LEAGUE, K_PREMIER_LEAGUE

logger = structlog.get_logger(__name__)


async def rebuild_team_seasons(
    session: AsyncSession,
    league: League,
) -> int:
    """Recompute and upsert TeamSeason records from raw Match data.

    Reads all FINISHED matches for the league/season, computes per-team
    aggregates (P/W/D/L/GF/GA/Pts), derives a form string (last 5 results),
    and updates ELO ratings chronologically.

    Args:
        session: Active async database session.
        league: The league to rebuild stats for.

    Returns:
        Number of TeamSeason records upserted.
    """
    log = logger.bind(league=league.code, season=league.current_season)
    log.info("rebuild_team_seasons_started")

    # Load all finished matches ordered by date
    stmt = (
        select(Match)
        .where(
            Match.league_id == league.id,
            Match.season == league.current_season,
            Match.status == MatchStatus.FINISHED,
        )
        .order_by(Match.played_at)
    )
    result = await session.execute(stmt)
    matches: list[Match] = list(result.scalars().all())

    if not matches:
        log.info("no_finished_matches_found")
        return 0

    # Choose K-factor based on tier
    k_factor = K_PREMIER_LEAGUE if league.tier == 1 else K_NON_LEAGUE
    elo_calc = EloCalculator(k_factor=k_factor)

    # Accumulators keyed by team_id
    stats: dict[str, dict] = defaultdict(lambda: {
        "played": 0, "won": 0, "drawn": 0, "lost": 0,
        "goals_for": 0, "goals_against": 0, "points": 0,
        "results": [],  # chronological list of 'W'/'D'/'L' for this team
    })
    elo_ratings: dict[str, float] = {}

    for match in matches:
        hid = match.home_team_id
        aid = match.away_team_id
        hg = match.home_goals or 0
        ag = match.away_goals or 0

        # ELO update
        home_elo = elo_ratings.get(hid, DEFAULT_INITIAL_RATING)
        away_elo = elo_ratings.get(aid, DEFAULT_INITIAL_RATING)
        new_home_elo, new_away_elo = elo_calc.update_ratings(home_elo, away_elo, hg, ag)
        elo_ratings[hid] = new_home_elo
        elo_ratings[aid] = new_away_elo

        # Home stats
        stats[hid]["played"] += 1
        stats[hid]["goals_for"] += hg
        stats[hid]["goals_against"] += ag
        # Away stats
        stats[aid]["played"] += 1
        stats[aid]["goals_for"] += ag
        stats[aid]["goals_against"] += hg

        if hg > ag:
            stats[hid]["won"] += 1
            stats[hid]["points"] += 3
            stats[hid]["results"].append("W")
            stats[aid]["lost"] += 1
            stats[aid]["results"].append("L")
        elif hg < ag:
            stats[aid]["won"] += 1
            stats[aid]["points"] += 3
            stats[aid]["results"].append("W")
            stats[hid]["lost"] += 1
            stats[hid]["results"].append("L")
        else:
            stats[hid]["drawn"] += 1
            stats[hid]["points"] += 1
            stats[hid]["results"].append("D")
            stats[aid]["drawn"] += 1
            stats[aid]["points"] += 1
            stats[aid]["results"].append("D")

    # Upsert TeamSeason records
    upserted = 0
    now = datetime.now(tz=timezone.utc)

    for team_id, s in stats.items():
        form = "".join(s["results"][-5:])  # last 5 results
        elo = elo_ratings.get(team_id, DEFAULT_INITIAL_RATING)

        # Try to find existing record
        existing_result = await session.execute(
            select(TeamSeason).where(
                TeamSeason.team_id == team_id,
                TeamSeason.league_id == league.id,
                TeamSeason.season == league.current_season,
            )
        )
        ts = existing_result.scalar_one_or_none()

        if ts is None:
            ts = TeamSeason(
                team_id=team_id,
                league_id=league.id,
                season=league.current_season,
            )
            session.add(ts)

        ts.played = s["played"]
        ts.won = s["won"]
        ts.drawn = s["drawn"]
        ts.lost = s["lost"]
        ts.goals_for = s["goals_for"]
        ts.goals_against = s["goals_against"]
        ts.points = s["points"]
        ts.form = form
        ts.elo_rating = round(elo, 2)
        ts.last_synced = now

        upserted += 1

    await session.flush()
    log.info("rebuild_team_seasons_completed", upserted=upserted)
    return upserted
