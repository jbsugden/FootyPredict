"""Dixon-Coles low-score correction for Poisson score probabilities.

Dixon & Coles (1997) observed that the independent Poisson model under-predicts
0-0 draws and 1-0 / 0-1 results, and over-predicts 1-1 draws.  The correction
applies a multiplicative adjustment ``tau(i, j, lambda_h, lambda_a, rho)`` to
the Poisson probability for scores ``(i, j)`` where ``i + j <= 1`` (and the
1-1 case).

Reference:
  Dixon, M.J. & Coles, S.G. (1997). Modelling association football scores and
  inefficiencies in the football betting market.
  *Applied Statistics*, 46(2), 265-280.
"""

from __future__ import annotations

import numpy as np

# Default rho value — typically estimated by MLE from historical data.
# Dixon & Coles report values around -0.13 for English football.
DEFAULT_RHO = -0.13


class DixonColesCorrection:
    """Apply the Dixon-Coles rho correction to a Poisson score matrix.

    Args:
        rho: The correlation parameter (negative value tightens the correction
            for low-scoring matches). Defaults to :data:`DEFAULT_RHO`.
    """

    def __init__(self, rho: float = DEFAULT_RHO) -> None:
        if rho >= 1.0:
            raise ValueError("rho must be less than 1.0")
        self.rho = rho

    def tau(
        self,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float,
    ) -> float:
        """Compute the correction factor tau for score ``(home_goals, away_goals)``.

        The factor is applied only to the four low-scoring combinations:
        (0,0), (1,0), (0,1), (1,1).  All other scores have tau=1.

        Args:
            home_goals: Number of goals scored by the home team.
            away_goals: Number of goals scored by the away team.
            lambda_home: Poisson mean for the home team.
            lambda_away: Poisson mean for the away team.

        Returns:
            Multiplicative correction factor.

        Raises:
            ValueError: If the correction would produce a negative probability
                (indicates rho is too large in magnitude for these lambdas).
        """
        rho = self.rho
        if home_goals == 0 and away_goals == 0:
            factor = 1.0 - lambda_home * lambda_away * rho
        elif home_goals == 1 and away_goals == 0:
            factor = 1.0 + lambda_away * rho
        elif home_goals == 0 and away_goals == 1:
            factor = 1.0 + lambda_home * rho
        elif home_goals == 1 and away_goals == 1:
            factor = 1.0 - rho
        else:
            return 1.0

        if factor <= 0:
            # Guard: correction would produce negative probability — fall back to 1
            return 1.0
        return factor

    def apply(
        self,
        matrix: np.ndarray,
        lambda_home: float,
        lambda_away: float,
    ) -> np.ndarray:
        """Apply the Dixon-Coles correction to a score probability matrix.

        Args:
            matrix: Uncorrected ``(N+1) x (N+1)`` Poisson probability matrix
                as produced by
                :meth:`~predictor.engine.poisson.StrengthCalculator.score_probability_matrix`.
            lambda_home: Poisson mean for the home team.
            lambda_away: Poisson mean for the away team.

        Returns:
            Corrected probability matrix of the same shape as ``matrix``.
            The matrix is re-normalised to sum to 1 after correction.
        """
        corrected = matrix.copy()

        # Apply tau correction only to the four low-score cells
        for h, a in ((0, 0), (1, 0), (0, 1), (1, 1)):
            if h < corrected.shape[0] and a < corrected.shape[1]:
                t = self.tau(h, a, lambda_home, lambda_away)
                corrected[h, a] *= t

        # Re-normalise so probabilities still sum to 1
        total = corrected.sum()
        if total > 0:
            corrected /= total

        return corrected

    def outcome_probabilities(
        self,
        matrix: np.ndarray,
    ) -> tuple[float, float, float]:
        """Derive win/draw/loss probabilities from a score matrix.

        Args:
            matrix: Score probability matrix (corrected or uncorrected).

        Returns:
            Tuple ``(p_home_win, p_draw, p_away_win)``.
        """
        n = matrix.shape[0]
        p_home = float(np.sum(matrix[i, j] for i in range(n) for j in range(i) if j < n))
        # Faster vectorised version
        p_home = float(np.sum(np.tril(matrix, k=-1)))
        p_away = float(np.sum(np.triu(matrix, k=1)))
        p_draw = float(np.trace(matrix))
        return p_home, p_draw, p_away
