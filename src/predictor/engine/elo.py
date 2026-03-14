"""Elo rating system for football teams.

The Elo system assigns each team a numeric rating. After each match the
winning team gains points and the losing team loses the same number.
The amount exchanged depends on the *expected* outcome — beating a much
stronger side earns more than beating a weak one.

Reference: https://en.wikipedia.org/wiki/Elo_rating_system
"""

from __future__ import annotations


# Default K-factors by competition tier
K_PREMIER_LEAGUE = 20
"""Smaller K for top-flight (ratings are more stable)."""

K_NON_LEAGUE = 32
"""Larger K for non-league (smaller sample sizes, more volatility)."""

# Starting Elo for all new teams
DEFAULT_INITIAL_RATING = 1500.0

# Home advantage in Elo points (added to the home team's effective rating)
HOME_ADVANTAGE_ELO = 100.0


class EloCalculator:
    """Computes Elo rating updates after football matches.

    Args:
        k_factor: The K-factor controlling how many points change hands per
            match. Use :data:`K_PREMIER_LEAGUE` for tier-1 and
            :data:`K_NON_LEAGUE` for lower tiers.
        home_advantage: Elo points added to the home team's effective rating
            when computing expected scores.
    """

    def __init__(
        self,
        k_factor: float = K_NON_LEAGUE,
        home_advantage: float = HOME_ADVANTAGE_ELO,
    ) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage

    # ------------------------------------------------------------------
    # Core maths
    # ------------------------------------------------------------------

    def expected_score(
        self,
        rating_a: float,
        rating_b: float,
        a_is_home: bool = False,
    ) -> float:
        """Return the expected score (win probability) for team A vs team B.

        The expected score is the Elo probability that team A outperforms
        team B (i.e. wins), accounting for home advantage.

        Args:
            rating_a: Current Elo rating of team A.
            rating_b: Current Elo rating of team B.
            a_is_home: Whether team A is the home side.

        Returns:
            Float in [0, 1] — the probability that team A wins.

        Example::

            calc = EloCalculator()
            p = calc.expected_score(1600, 1500, a_is_home=True)
            # Returns ~0.679
        """
        effective_a = rating_a + (self.home_advantage if a_is_home else 0.0)
        return 1.0 / (1.0 + 10.0 ** ((rating_b - effective_a) / 400.0))

    def actual_score(self, home_goals: int, away_goals: int) -> tuple[float, float]:
        """Convert a match result to Elo scores (W=1, D=0.5, L=0).

        Args:
            home_goals: Goals scored by the home team.
            away_goals: Goals scored by the away team.

        Returns:
            Tuple ``(home_score, away_score)`` where scores sum to 1.0.
        """
        if home_goals > away_goals:
            return 1.0, 0.0
        if home_goals < away_goals:
            return 0.0, 1.0
        return 0.5, 0.5

    def update_ratings(
        self,
        home_rating: float,
        away_rating: float,
        home_goals: int,
        away_goals: int,
    ) -> tuple[float, float]:
        """Apply Elo update after a match and return new ratings.

        Args:
            home_rating: Pre-match Elo of the home team.
            away_rating: Pre-match Elo of the away team.
            home_goals: Final goals scored by the home team.
            away_goals: Final goals scored by the away team.

        Returns:
            Tuple ``(new_home_rating, new_away_rating)``.

        Example::

            calc = EloCalculator(k_factor=32)
            new_home, new_away = calc.update_ratings(1500, 1500, 2, 1)
            # Home win: new_home > 1500, new_away < 1500
        """
        expected_home = self.expected_score(home_rating, away_rating, a_is_home=True)
        expected_away = 1.0 - expected_home

        actual_home, actual_away = self.actual_score(home_goals, away_goals)

        new_home = home_rating + self.k_factor * (actual_home - expected_home)
        new_away = away_rating + self.k_factor * (actual_away - expected_away)

        return new_home, new_away

    def bulk_update(
        self,
        ratings: dict[str, float],
        results: list[tuple[str, str, int, int]],
    ) -> dict[str, float]:
        """Apply a sequence of match results to a ratings dictionary.

        Args:
            ratings: Mapping of ``team_id -> current_elo``. Missing teams
                are initialised to :data:`DEFAULT_INITIAL_RATING`.
            results: List of ``(home_id, away_id, home_goals, away_goals)``
                tuples, ordered chronologically.

        Returns:
            Updated ``team_id -> elo`` mapping (new dict, original unchanged).
        """
        updated = dict(ratings)
        for home_id, away_id, home_goals, away_goals in results:
            home_elo = updated.get(home_id, DEFAULT_INITIAL_RATING)
            away_elo = updated.get(away_id, DEFAULT_INITIAL_RATING)
            new_home, new_away = self.update_ratings(
                home_elo, away_elo, home_goals, away_goals
            )
            updated[home_id] = new_home
            updated[away_id] = new_away
        return updated
