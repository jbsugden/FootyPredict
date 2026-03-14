"""Tests for the Dixon-Coles correction model."""

from __future__ import annotations

import numpy as np
import pytest

from predictor.engine.dixon_coles import DEFAULT_RHO, DixonColesCorrection


class TestTauCorrection:
    def test_high_scores_return_one(self) -> None:
        """Scores above (1,1) should have tau = 1 (no correction)."""
        dc = DixonColesCorrection()
        assert dc.tau(2, 0, 1.5, 1.5) == 1.0
        assert dc.tau(3, 2, 1.5, 1.5) == 1.0
        assert dc.tau(0, 2, 1.5, 1.5) == 1.0

    def test_zero_zero_correction(self) -> None:
        """0-0 with negative rho should increase probability (tau > 1)."""
        dc = DixonColesCorrection(rho=-0.13)
        tau = dc.tau(0, 0, 1.5, 1.2)
        # tau = 1 - lambda_h * lambda_a * rho = 1 - 1.5*1.2*(-0.13) = 1 + 0.234 = 1.234
        assert tau > 1.0
        assert tau == pytest.approx(1.0 - 1.5 * 1.2 * (-0.13), abs=1e-9)

    def test_one_one_correction(self) -> None:
        """1-1 with negative rho should decrease probability (tau < 1)."""
        dc = DixonColesCorrection(rho=-0.13)
        tau = dc.tau(1, 1, 1.5, 1.2)
        # tau = 1 - rho = 1 - (-0.13) = 1.13
        assert tau == pytest.approx(1.13, abs=1e-9)

    def test_one_zero_correction(self) -> None:
        dc = DixonColesCorrection(rho=-0.13)
        tau = dc.tau(1, 0, 1.5, 1.2)
        # tau = 1 + lambda_away * rho = 1 + 1.2 * (-0.13)
        expected = 1.0 + 1.2 * (-0.13)
        assert tau == pytest.approx(expected, abs=1e-9)
        assert tau < 1.0

    def test_zero_one_correction(self) -> None:
        dc = DixonColesCorrection(rho=-0.13)
        tau = dc.tau(0, 1, 1.5, 1.2)
        # tau = 1 + lambda_home * rho = 1 + 1.5 * (-0.13)
        expected = 1.0 + 1.5 * (-0.13)
        assert tau == pytest.approx(expected, abs=1e-9)

    def test_zero_rho_means_no_correction(self) -> None:
        """With rho=0, all tau values should be 1."""
        dc = DixonColesCorrection(rho=0.0)
        for h, a in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 3)]:
            assert dc.tau(h, a, 1.5, 1.2) == 1.0

    def test_negative_factor_falls_back_to_one(self) -> None:
        """If correction would produce negative probability, return 1."""
        dc = DixonColesCorrection(rho=0.99)
        # tau(1, 0, ...) = 1 + lambda_away * 0.99 — should be > 0 for reasonable lambdas
        # but for 0-1: tau = 1 + lambda_home * rho, if lambda_home large enough, could be < 0
        # Let's use a huge rho that makes tau negative for 1-0
        dc2 = DixonColesCorrection(rho=-5.0)
        tau = dc2.tau(1, 0, 1.5, 1.2)
        # 1 + 1.2 * (-5) = 1 - 6 = -5 → should fallback to 1.0
        assert tau == 1.0

    def test_rho_must_be_less_than_one(self) -> None:
        with pytest.raises(ValueError):
            DixonColesCorrection(rho=1.0)


class TestApplyCorrection:
    def test_corrected_matrix_sums_to_one(self) -> None:
        dc = DixonColesCorrection()
        from predictor.engine.poisson import StrengthCalculator
        from predictor.engine.poisson import MatchRecord
        from datetime import datetime, timezone

        # Create a simple probability matrix
        from scipy.stats import poisson

        lh, la = 1.5, 1.2
        home_probs = poisson.pmf(np.arange(11), lh)
        away_probs = poisson.pmf(np.arange(11), la)
        matrix = np.outer(home_probs, away_probs)

        corrected = dc.apply(matrix, lh, la)
        assert corrected.sum() == pytest.approx(1.0, abs=1e-6)

    def test_corrected_matrix_shape_unchanged(self) -> None:
        dc = DixonColesCorrection()
        matrix = np.ones((11, 11)) / 121
        corrected = dc.apply(matrix, 1.5, 1.2)
        assert corrected.shape == (11, 11)

    def test_correction_modifies_low_score_cells(self) -> None:
        dc = DixonColesCorrection(rho=-0.13)
        from scipy.stats import poisson

        lh, la = 1.5, 1.2
        home_probs = poisson.pmf(np.arange(11), lh)
        away_probs = poisson.pmf(np.arange(11), la)
        matrix = np.outer(home_probs, away_probs)

        corrected = dc.apply(matrix, lh, la)
        # The 0-0 cell should be different after correction
        # (we can't compare directly due to renormalisation, but the ratio should differ)
        ratio_00 = corrected[0, 0] / corrected[5, 5]
        orig_ratio_00 = matrix[0, 0] / matrix[5, 5]
        assert ratio_00 != pytest.approx(orig_ratio_00, rel=0.01)


class TestOutcomeProbabilities:
    def test_probabilities_sum_to_one(self) -> None:
        dc = DixonColesCorrection()
        from scipy.stats import poisson

        lh, la = 1.5, 1.2
        home_probs = poisson.pmf(np.arange(11), lh)
        away_probs = poisson.pmf(np.arange(11), la)
        matrix = np.outer(home_probs, away_probs)
        corrected = dc.apply(matrix, lh, la)

        p_home, p_draw, p_away = dc.outcome_probabilities(corrected)
        assert p_home + p_draw + p_away == pytest.approx(1.0, abs=1e-6)

    def test_home_win_probability_positive(self) -> None:
        dc = DixonColesCorrection()
        from scipy.stats import poisson

        lh, la = 2.0, 0.8
        home_probs = poisson.pmf(np.arange(11), lh)
        away_probs = poisson.pmf(np.arange(11), la)
        matrix = np.outer(home_probs, away_probs)

        p_home, p_draw, p_away = dc.outcome_probabilities(matrix)
        assert p_home > p_away  # Strong home team should have higher win prob
