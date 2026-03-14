"""Per-fixture prediction service for the team deep-dive page.

Given a team and the league's match history, computes expected goals,
win/draw/loss probabilities, score heatmaps, and aggregate outlook for
each remaining fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from predictor.engine.dixon_coles import DixonColesCorrection
from predictor.engine.head_to_head import adjust_lambdas_h2h
from predictor.engine.poisson import MatchRecord, StrengthCalculator


@dataclass
class FixturePrediction:
    """Prediction breakdown for a single upcoming fixture."""

    opponent_id: str
    opponent_name: str
    is_home: bool
    played_at: datetime
    lambda_team: float
    lambda_opponent: float
    p_win: float
    p_draw: float
    p_loss: float
    score_matrix: list[list[float]]
    top_scores: list[tuple[int, int, float]]


@dataclass
class TeamStrengthProfile:
    """Attack and defence strength relative to league average (1.0)."""

    attack: float
    defence: float


@dataclass
class FixturePredictionResult:
    """Aggregate result of per-fixture predictions for one team."""

    team_strength: TeamStrengthProfile
    league_avg_goals: float
    fixtures: list[FixturePrediction]
    expected_remaining_points: float


def predict_fixtures(
    finished_matches: list,
    scheduled_matches: list,
    team_id: str,
    team_map: dict,
) -> FixturePredictionResult:
    """Compute per-fixture predictions for a single team.

    Args:
        finished_matches: List of Match ORM objects (status=FINISHED).
        scheduled_matches: List of Match ORM objects (status=SCHEDULED).
        team_id: UUID of the team to analyse.
        team_map: Dict mapping team_id -> Team ORM object (for names).

    Returns:
        A :class:`FixturePredictionResult` with strength profile,
        per-fixture breakdowns, and expected remaining points.
    """
    # Build MatchRecord list from finished matches
    records = [
        MatchRecord(
            home_team_id=m.home_team_id,
            away_team_id=m.away_team_id,
            home_goals=m.home_goals,
            away_goals=m.away_goals,
            played_at=m.played_at,
        )
        for m in finished_matches
    ]

    calc = StrengthCalculator(records)
    strengths = calc.get_strengths()

    # Compute league average goals from finished matches
    if finished_matches:
        total_goals = sum(m.home_goals + m.away_goals for m in finished_matches)
        total_team_games = len(finished_matches) * 2
        league_avg_goals = total_goals / total_team_games
    else:
        league_avg_goals = 1.4

    # Guard against degenerate league averages
    league_avg_goals = max(league_avg_goals, 0.1)

    # Team strength profile
    ts = strengths.get(team_id)
    if ts is not None:
        team_strength = TeamStrengthProfile(attack=ts.attack, defence=ts.defence)
    else:
        team_strength = TeamStrengthProfile(attack=1.0, defence=1.0)

    dc = DixonColesCorrection()
    fixture_preds: list[FixturePrediction] = []
    expected_points = 0.0

    # Filter scheduled matches to those involving this team
    team_scheduled = [
        m
        for m in scheduled_matches
        if m.home_team_id == team_id or m.away_team_id == team_id
    ]

    for match in team_scheduled:
        is_home = match.home_team_id == team_id
        opponent_id = match.away_team_id if is_home else match.home_team_id
        opponent_name = team_map[opponent_id].name if opponent_id in team_map else opponent_id

        lambda_home, lambda_away = calc.compute_lambda(
            match.home_team_id, match.away_team_id, league_avg_goals
        )
        lambda_home, lambda_away = adjust_lambdas_h2h(
            lambda_home, lambda_away, records, match.home_team_id, match.away_team_id
        )
        matrix = calc.score_probability_matrix(lambda_home, lambda_away)
        matrix = dc.apply(matrix, lambda_home, lambda_away)
        p_home_win, p_draw, p_away_win = dc.outcome_probabilities(matrix)

        if is_home:
            lambda_team = lambda_home
            lambda_opp = lambda_away
            p_win = p_home_win
            p_loss = p_away_win
        else:
            lambda_team = lambda_away
            lambda_opp = lambda_home
            p_win = p_away_win
            p_loss = p_home_win

        # Truncate matrix to 6x6 and convert to plain lists
        truncated = matrix[:6, :6]
        score_matrix = [[float(truncated[i, j]) for j in range(6)] for i in range(6)]

        # Extract top 5 most likely scorelines from the full matrix
        flat_indices = []
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(cols):
                flat_indices.append((i, j, float(matrix[i, j])))
        flat_indices.sort(key=lambda x: x[2], reverse=True)
        top_scores = [(h, a, p) for h, a, p in flat_indices[:5]]

        fixture_preds.append(
            FixturePrediction(
                opponent_id=opponent_id,
                opponent_name=opponent_name,
                is_home=is_home,
                played_at=match.played_at,
                lambda_team=lambda_team,
                lambda_opponent=lambda_opp,
                p_win=p_win,
                p_draw=p_draw,
                p_loss=p_loss,
                score_matrix=score_matrix,
                top_scores=top_scores,
            )
        )

        expected_points += 3 * p_win + 1 * p_draw

    return FixturePredictionResult(
        team_strength=team_strength,
        league_avg_goals=league_avg_goals,
        fixtures=fixture_preds,
        expected_remaining_points=expected_points,
    )
