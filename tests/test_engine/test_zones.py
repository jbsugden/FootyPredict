"""Tests for league zone definitions and probability computation."""

from __future__ import annotations

from predictor.engine.zones import (
    build_zone_class_map,
    compute_zone_probabilities,
    get_zone_for_position,
)


class TestComputeZoneProbabilities:
    """Zone probability computation from pos_dist arrays."""

    def test_pl_zones_sum_correct_positions(self) -> None:
        """PL: UCL=1-4, Europa=5-6, Relegation=18-20."""
        # 20 teams, uniform distribution → each position has 5% probability
        pos_dist = [0.05] * 20
        zones = compute_zone_probabilities(pos_dist, "PL", 20)

        labels = {z["label"]: z["probability"] for z in zones}
        assert abs(labels["Champions League"] - 0.20) < 1e-9
        assert abs(labels["Europa / Conference"] - 0.10) < 1e-9
        assert abs(labels["Relegation"] - 0.15) < 1e-9

    def test_npl_w_zones(self) -> None:
        """NPL_W: Promotion=1st, Relegation=bottom 3."""
        pos_dist = [0.05] * 20
        zones = compute_zone_probabilities(pos_dist, "NPL_W", 20)

        labels = {z["label"]: z["probability"] for z in zones}
        assert abs(labels["Promotion"] - 0.05) < 1e-9
        assert abs(labels["Relegation"] - 0.15) < 1e-9

    def test_unknown_league_uses_defaults(self) -> None:
        """Unknown league code falls back to default zones."""
        pos_dist = [0.1] * 10
        zones = compute_zone_probabilities(pos_dist, "UNKNOWN", 10)

        labels = {z["label"] for z in zones}
        assert "Champion" in labels
        assert "Relegation" in labels

    def test_negative_positions_resolved_to_end(self) -> None:
        """Negative positions resolve to end of table."""
        # 10 teams, all probability on last position
        pos_dist = [0.0] * 9 + [1.0]
        zones = compute_zone_probabilities(pos_dist, "NPL_W", 10)

        relegation = next(z for z in zones if z["label"] == "Relegation")
        assert relegation["probability"] == 1.0

    def test_probabilities_in_valid_range(self) -> None:
        """All zone probabilities should be between 0 and 1."""
        pos_dist = [0.05] * 20
        zones = compute_zone_probabilities(pos_dist, "PL", 20)

        for z in zones:
            assert 0.0 <= z["probability"] <= 1.0

    def test_empty_pos_dist_returns_zero_probabilities(self) -> None:
        zones = compute_zone_probabilities([], "PL", 20)
        for z in zones:
            assert z["probability"] == 0.0

    def test_concentrated_distribution(self) -> None:
        """100% chance of 1st place → 100% UCL, 0% relegation."""
        pos_dist = [1.0] + [0.0] * 19
        zones = compute_zone_probabilities(pos_dist, "PL", 20)

        labels = {z["label"]: z["probability"] for z in zones}
        assert labels["Champions League"] == 1.0
        assert labels["Relegation"] == 0.0


class TestGetZoneForPosition:
    """Position-to-zone CSS class lookup."""

    def test_pl_position_1_is_ucl(self) -> None:
        assert get_zone_for_position("PL", 1, 20) == "zone-ucl"

    def test_pl_position_5_is_europa(self) -> None:
        assert get_zone_for_position("PL", 5, 20) == "zone-europa"

    def test_pl_position_20_is_relegation(self) -> None:
        assert get_zone_for_position("PL", 20, 20) == "zone-relegation"

    def test_pl_midtable_is_none(self) -> None:
        assert get_zone_for_position("PL", 10, 20) is None

    def test_npl_w_position_1_is_promotion(self) -> None:
        assert get_zone_for_position("NPL_W", 1, 20) == "zone-promotion"


class TestBuildZoneClassMap:
    """Position → CSS class mapping."""

    def test_pl_map_contains_expected_positions(self) -> None:
        zone_map = build_zone_class_map("PL", 20)
        assert zone_map[1] == "zone-ucl"
        assert zone_map[4] == "zone-ucl"
        assert zone_map[5] == "zone-europa"
        assert zone_map[6] == "zone-europa"
        assert zone_map[18] == "zone-relegation"
        assert zone_map[20] == "zone-relegation"
        assert 10 not in zone_map

    def test_npl_w_map_with_different_team_count(self) -> None:
        """Zone map adjusts negative positions for team count."""
        zone_map = build_zone_class_map("NPL_W", 18)
        assert zone_map[1] == "zone-promotion"
        assert zone_map[16] == "zone-relegation"
        assert zone_map[17] == "zone-relegation"
        assert zone_map[18] == "zone-relegation"
