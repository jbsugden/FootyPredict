"""SQLAlchemy 2.0 ORM models for FootyPredict.

All primary keys are UUIDs generated on the Python side for portability across
databases. Timestamps use timezone-aware UTC datetimes.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _new_uuid() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all models."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataSource(str, enum.Enum):
    """Supported data ingestion sources."""

    API_FOOTBALL_DATA = "API_FOOTBALL_DATA"
    SCRAPE_FA_FULLTIME = "SCRAPE_FA_FULLTIME"


class MatchStatus(str, enum.Enum):
    """Lifecycle states for a fixture."""

    SCHEDULED = "SCHEDULED"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    CANCELLED = "CANCELLED"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class League(Base):
    """A football league or competition tracked by the application."""

    __tablename__ = "leagues"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    code: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    """Short machine-readable code, e.g. 'PL', 'NPL_W'."""
    tier: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    """English football pyramid tier (1=Premier League, 6=NPL etc.)."""
    data_source: Mapped[DataSource] = mapped_column(
        Enum(DataSource), nullable=False, default=DataSource.API_FOOTBALL_DATA
    )
    current_season: Mapped[str] = mapped_column(String(20), nullable=False)
    """E.g. '2024-25'."""

    # Relationships
    teams: Mapped[list[Team]] = relationship("Team", back_populates="league")
    matches: Mapped[list[Match]] = relationship("Match", back_populates="league")
    team_seasons: Mapped[list[TeamSeason]] = relationship(
        "TeamSeason", back_populates="league"
    )
    predictions: Mapped[list[Prediction]] = relationship(
        "Prediction", back_populates="league"
    )

    def __repr__(self) -> str:
        return f"<League {self.code!r} tier={self.tier}>"


class Team(Base):
    """A football club belonging to a league."""

    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("league_id", "name", name="uq_team_league_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    league_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    short_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    """ID used by the upstream data source (e.g. football-data.org team ID)."""
    crest_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    """URL to the team's badge/crest image."""
    website_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    """URL to the club's official website."""

    # Relationships
    league: Mapped[League] = relationship("League", back_populates="teams")
    home_matches: Mapped[list[Match]] = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches: Mapped[list[Match]] = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )
    team_seasons: Mapped[list[TeamSeason]] = relationship(
        "TeamSeason", back_populates="team"
    )

    def __repr__(self) -> str:
        return f"<Team {self.name!r}>"


class Match(Base):
    """A single fixture between two teams."""

    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    league_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False
    )
    season: Mapped[str] = mapped_column(String(20), nullable=False)
    matchday: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_team_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False
    )
    away_team_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False
    )
    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[MatchStatus] = mapped_column(
        Enum(MatchStatus), nullable=False, default=MatchStatus.SCHEDULED
    )
    external_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    """Upstream fixture identifier for deduplication (e.g. NPL API ``_id``)."""
    played_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc, onupdate=_now_utc
    )

    # Relationships
    league: Mapped[League] = relationship("League", back_populates="matches")
    home_team: Mapped[Team] = relationship(
        "Team", foreign_keys=[home_team_id], back_populates="home_matches"
    )
    away_team: Mapped[Team] = relationship(
        "Team", foreign_keys=[away_team_id], back_populates="away_matches"
    )

    def __repr__(self) -> str:
        return (
            f"<Match {self.home_team_id} vs {self.away_team_id} "
            f"({self.home_goals}-{self.away_goals}) {self.status.value}>"
        )


class TeamSeason(Base):
    """Aggregated season statistics and model parameters for a team in a season."""

    __tablename__ = "team_seasons"
    __table_args__ = (
        UniqueConstraint("team_id", "league_id", "season", name="uq_team_season"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    team_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    league_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False
    )
    season: Mapped[str] = mapped_column(String(20), nullable=False)

    # Season record
    played: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    won: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    drawn: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lost: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    goals_for: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    goals_against: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    points: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Recent form string, e.g. "WWDLW" (most recent last)
    form: Mapped[str] = mapped_column(String(10), nullable=False, default="")

    # Model parameters
    elo_rating: Mapped[float] = mapped_column(Float, nullable=False, default=1500.0)

    last_synced: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )

    # Relationships
    team: Mapped[Team] = relationship("Team", back_populates="team_seasons")
    league: Mapped[League] = relationship("League", back_populates="team_seasons")

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    def __repr__(self) -> str:
        return f"<TeamSeason team={self.team_id} season={self.season} pts={self.points}>"


class Prediction(Base):
    """A stored Monte Carlo simulation result for a league season."""

    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    league_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False
    )
    season: Mapped[str] = mapped_column(String(20), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    simulation_runs: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=10_000
    )
    results: Mapped[dict] = mapped_column(JSON, nullable=False)
    """Serialised simulation output.

    Structure::

        {
            "<team_id>": {
                "mean_pos": 3.4,
                "mean_points": 71.2,
                "pos_dist": [0, 0, 0.12, 0.45, ...]  # index=position-1
            },
            ...
        }
    """

    # Relationships
    league: Mapped[League] = relationship("League", back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"<Prediction league={self.league_id} season={self.season} "
            f"runs={self.simulation_runs}>"
        )
