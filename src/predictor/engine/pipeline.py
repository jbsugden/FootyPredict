"""Shared simulation setup pipeline.

Extracts the common logic for building a :class:`SimulationInput` from the
database so that it can be reused by the prediction API, the nightly
scheduler, and the scenario explorer.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from predictor.db.models import League
from predictor.db.repos.match import MatchRepository, previous_season
from predictor.db.repos.team import TeamRepository
from predictor.engine.poisson import MatchRecord, StrengthCalculator
from predictor.engine.simulator import Fixture, SimulationInput
from predictor.engine.standings import apply_result, initialise_standings


async def build_simulation_input(
    session: AsyncSession,
    league_id: str,
    current_season: str,
) -> SimulationInput | None:
    """Load all data needed for a simulation from the database.

    Args:
        session: Active async database session.
        league_id: UUID of the league.
        current_season: Season string (e.g. ``"2025"``).

    Returns:
        A fully populated :class:`SimulationInput`, or ``None`` if there are
        no finished matches to base the simulation on.
    """
    match_repo = MatchRepository(session)
    finished = await match_repo.get_finished(league_id, current_season)
    scheduled = await match_repo.get_scheduled(league_id, current_season)

    if not finished:
        return None

    # Include previous-season matches for cross-season strength prior
    prev = previous_season(current_season)
    prev_finished = await match_repo.get_finished_multi_season(league_id, [prev])

    # Build StrengthCalculator from current + previous season
    match_records = [
        MatchRecord(
            home_team_id=m.home_team_id,
            away_team_id=m.away_team_id,
            home_goals=m.home_goals or 0,
            away_goals=m.away_goals or 0,
            played_at=m.played_at,
        )
        for m in prev_finished + finished
    ]
    strength_calc = StrengthCalculator(match_records)
    strength_calc.compute_strengths()

    # Compute league average goals
    total_goals = sum((m.home_goals or 0) + (m.away_goals or 0) for m in finished)
    league_avg = total_goals / (len(finished) * 2) if finished else 1.4

    # Build current standings from finished matches
    all_team_ids = list(
        {m.home_team_id for m in finished} | {m.away_team_id for m in finished}
    )
    current_standings = initialise_standings(all_team_ids)
    for m in finished:
        apply_result(
            current_standings,
            m.home_team_id,
            m.away_team_id,
            m.home_goals or 0,
            m.away_goals or 0,
        )

    # Build remaining fixtures
    remaining_fixtures = [
        Fixture(home_id=m.home_team_id, away_id=m.away_team_id)
        for m in scheduled
    ]

    return SimulationInput(
        team_ids=all_team_ids,
        current_standings=current_standings,
        remaining_fixtures=remaining_fixtures,
        strength_calculator=strength_calc,
        league_avg_goals=league_avg,
    )


async def build_team_name_map(
    session: AsyncSession,
    league_id: str,
) -> dict[str, str]:
    """Build a mapping of team_id -> team_name for a league.

    Args:
        session: Active async database session.
        league_id: UUID of the league.

    Returns:
        Dict mapping team UUID to team name.
    """
    team_repo = TeamRepository(session)
    teams = await team_repo.get_by_league(league_id)
    return {t.id: t.name for t in teams}
