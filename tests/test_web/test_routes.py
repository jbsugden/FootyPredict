"""Tests for Jinja2 web page routes.

Verifies template rendering, nav links, and regression for the
``| enumerate`` template bug that caused Internal Server Error.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import AsyncClient

from predictor.db.models import League, Match, MatchStatus, Prediction, Team, TeamSeason


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class TestIndexPage:
    @pytest.mark.asyncio
    async def test_index_returns_200(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_index_contains_brand(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/")
        assert "FootyPredict" in response.text

    @pytest.mark.asyncio
    async def test_index_lists_leagues(
        self, async_client: AsyncClient, sample_league: League
    ) -> None:
        response = await async_client.get("/")
        assert response.status_code == 200
        assert sample_league.name in response.text


class TestLeagueDetailPage:
    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_league(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get(f"/league/{_uuid()}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_200_with_standings(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert sample_league.name in response.text
        # Standings table should be rendered
        assert "Arsenal FC" in response.text

    @pytest.mark.asyncio
    async def test_renders_prediction_table_without_error(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Regression test: prediction_table.html must render without crashing.

        Previously the template used ``| enumerate`` (not a valid Jinja2 filter)
        which caused an Internal Server Error.
        """
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": [
                    0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                    for j in range(len(sample_teams))
                ],
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)
        await db_session.flush()

        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert "Predicted Final Table" in response.text
        # Check that top-positions labels rendered (regression for | enumerate bug)
        assert "P(" in response.text

    @pytest.mark.asyncio
    async def test_nav_contains_dynamic_league_links(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_team_seasons,
    ) -> None:
        """Nav should contain links with league UUID paths, not hardcoded slugs."""
        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert f"/league/{sample_league.id}" in response.text

    @pytest.mark.asyncio
    async def test_no_htmx_json_auto_load(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_team_seasons,
    ) -> None:
        """The standings div should NOT have hx-get pointing to the JSON API."""
        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert 'hx-get="/api/leagues/' not in response.text

    @pytest.mark.asyncio
    async def test_footer_references_thenpl(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_team_seasons,
    ) -> None:
        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert "thenpl.co.uk" in response.text
        assert "FA Full Time" not in response.text


class TestTeamDetailPage:
    @pytest.mark.asyncio
    async def test_returns_200_with_team_name(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
        sample_matches,
    ) -> None:
        team = sample_teams[0]
        response = await async_client.get(
            f"/league/{sample_league.id}/team/{team.id}"
        )
        assert response.status_code == 200
        assert team.name in response.text

    @pytest.mark.asyncio
    async def test_returns_404_for_invalid_team(
        self, async_client: AsyncClient, sample_league: League
    ) -> None:
        response = await async_client.get(
            f"/league/{sample_league.id}/team/{_uuid()}"
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_404_for_invalid_league(
        self, async_client: AsyncClient, sample_teams: list[Team]
    ) -> None:
        response = await async_client.get(
            f"/league/{_uuid()}/team/{sample_teams[0].id}"
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_renders_fixture_cards_with_scheduled_matches(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
        sample_matches,
    ) -> None:
        """Add a scheduled match and verify fixture card renders."""
        team_a, team_b = sample_teams[0], sample_teams[1]
        scheduled = Match(
            id=_uuid(),
            league_id=sample_league.id,
            season="2024-25",
            matchday=10,
            home_team_id=team_a.id,
            away_team_id=team_b.id,
            status=MatchStatus.SCHEDULED,
            played_at=_now(),
        )
        db_session.add(scheduled)
        await db_session.flush()

        response = await async_client.get(
            f"/league/{sample_league.id}/team/{team_a.id}"
        )
        assert response.status_code == 200
        # Should contain the opponent name in a fixture card
        assert team_b.name in response.text
        # Should contain W/D/L bar
        assert "wdl-bar" in response.text

    @pytest.mark.asyncio
    async def test_renders_position_distribution_with_prediction(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
        sample_matches,
    ) -> None:
        """Verify position distribution section appears when prediction exists."""
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": [
                    0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                    for j in range(len(sample_teams))
                ],
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)
        await db_session.flush()

        team = sample_teams[0]
        response = await async_client.get(
            f"/league/{sample_league.id}/team/{team.id}"
        )
        assert response.status_code == 200
        assert "Prediction Summary" in response.text
        assert "Mean Position" in response.text


class TestZonePillsOnLeaguePage:
    """Feature 1: Zone probability pills on the predicted table."""

    @pytest.mark.asyncio
    async def test_zone_pills_render_on_prediction_table(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Zone pills should appear when a prediction with pos_dist exists."""
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": (
                    [0.85, 0.10, 0.03, 0.02] if i == 0
                    else [
                        0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                        for j in range(len(sample_teams))
                    ]
                ),
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)
        await db_session.flush()

        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        text = response.text
        assert "zone-pills" in text
        assert "zone-pill" in text
        assert "Champion" in text
        assert "Relegation" in text
        assert "85.0%" in text

    @pytest.mark.asyncio
    async def test_zone_pills_hidden_without_prediction(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Without a prediction, zone pills should not appear."""
        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert "zone-pills" not in response.text
        assert "zone-pill" not in response.text


class TestZoneCardsOnTeamPage:
    """Feature 1: Zone probability cards on the team detail page."""

    @pytest.mark.asyncio
    async def test_zone_cards_render_on_team_page(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
        sample_matches,
    ) -> None:
        """Zone cards should render when prediction data is available."""
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": (
                    [0.85, 0.10, 0.03, 0.02] if i == 0
                    else [
                        0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                        for j in range(len(sample_teams))
                    ]
                ),
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)
        await db_session.flush()

        team = sample_teams[0]
        response = await async_client.get(
            f"/league/{sample_league.id}/team/{team.id}"
        )
        assert response.status_code == 200
        assert "Champion" in response.text
        assert "85.0%" in response.text

    @pytest.mark.asyncio
    async def test_zone_cards_hidden_without_prediction(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
        sample_matches,
    ) -> None:
        """Without a prediction, zone labels should not appear on team page."""
        team = sample_teams[0]
        response = await async_client.get(
            f"/league/{sample_league.id}/team/{team.id}"
        )
        assert response.status_code == 200
        # "Champion" label only appears in the zone cards section
        assert "Champion" not in response.text


class TestTimelineOnLeaguePage:
    """Feature 2: Prediction timeline on the league detail page."""

    @pytest.mark.asyncio
    async def test_timeline_renders_with_multiple_predictions(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Timeline chart should render when 2+ predictions exist."""
        from datetime import timedelta

        base_time = _now()
        for day_offset in range(2):
            fake_results = {
                team.id: {
                    "mean_pos": float(i + 1),
                    "mean_points": float(60 - i * 5),
                    "pos_dist": [
                        0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                        for j in range(len(sample_teams))
                    ],
                }
                for i, team in enumerate(sample_teams)
            }
            prediction = Prediction(
                id=_uuid(),
                league_id=sample_league.id,
                season=sample_league.current_season,
                generated_at=base_time - timedelta(days=day_offset),
                simulation_runs=10_000,
                results=fake_results,
            )
            db_session.add(prediction)
        await db_session.flush()

        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        text = response.text
        assert "Prediction Timeline" in text
        assert "timelineChart" in text
        assert "renderTimelineChart" in text

    @pytest.mark.asyncio
    async def test_timeline_hidden_with_single_prediction(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Timeline should NOT render with only 1 prediction."""
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": [
                    0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                    for j in range(len(sample_teams))
                ],
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)
        await db_session.flush()

        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert "Prediction Timeline" not in response.text
        assert "timelineChart" not in response.text

    @pytest.mark.asyncio
    async def test_timeline_hidden_without_prediction(
        self,
        async_client: AsyncClient,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Timeline should NOT render without any predictions."""
        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert "Prediction Timeline" not in response.text


class TestTimelineOnTeamPage:
    """Feature 2: Prediction timeline on the team detail page."""

    @pytest.mark.asyncio
    async def test_team_timeline_renders_with_multiple_predictions(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
        sample_matches,
    ) -> None:
        """Team timeline should render when 2+ predictions exist."""
        from datetime import timedelta

        base_time = _now()
        for day_offset in range(2):
            fake_results = {
                team.id: {
                    "mean_pos": float(i + 1),
                    "mean_points": float(60 - i * 5),
                    "pos_dist": [
                        0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                        for j in range(len(sample_teams))
                    ],
                }
                for i, team in enumerate(sample_teams)
            }
            prediction = Prediction(
                id=_uuid(),
                league_id=sample_league.id,
                season=sample_league.current_season,
                generated_at=base_time - timedelta(days=day_offset),
                simulation_runs=10_000,
                results=fake_results,
            )
            db_session.add(prediction)
        await db_session.flush()

        team = sample_teams[0]
        response = await async_client.get(
            f"/league/{sample_league.id}/team/{team.id}"
        )
        assert response.status_code == 200
        text = response.text
        assert "Position Over Time" in text
        assert "teamTimelineChart" in text
        assert "renderTeamTimelineChart" in text

    @pytest.mark.asyncio
    async def test_team_timeline_hidden_with_single_prediction(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
        sample_matches,
    ) -> None:
        """Team timeline should NOT render with only 1 prediction."""
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": [
                    0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                    for j in range(len(sample_teams))
                ],
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)
        await db_session.flush()

        team = sample_teams[0]
        response = await async_client.get(
            f"/league/{sample_league.id}/team/{team.id}"
        )
        assert response.status_code == 200
        assert "Position Over Time" not in response.text


class TestScenarioExplorerOnLeaguePage:
    """Feature 3: Scenario explorer on the league detail page."""

    @pytest.mark.asyncio
    async def test_scenario_explorer_renders_with_prediction_and_scheduled_matches(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Scenario explorer should render when both prediction and scheduled matches exist."""
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": [
                    0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                    for j in range(len(sample_teams))
                ],
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)

        scheduled = Match(
            id=_uuid(),
            league_id=sample_league.id,
            season="2024-25",
            matchday=10,
            home_team_id=sample_teams[0].id,
            away_team_id=sample_teams[1].id,
            status=MatchStatus.SCHEDULED,
            played_at=_now(),
        )
        db_session.add(scheduled)
        await db_session.flush()

        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        text = response.text
        assert "Scenario Explorer" in text
        assert "scenario-table" in text
        assert "Run Scenario" in text
        assert "Lock" in text
        assert sample_teams[0].name in text
        assert sample_teams[1].name in text

    @pytest.mark.asyncio
    async def test_scenario_explorer_hidden_without_prediction(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Scenario explorer should NOT render without a prediction."""
        scheduled = Match(
            id=_uuid(),
            league_id=sample_league.id,
            season="2024-25",
            matchday=10,
            home_team_id=sample_teams[0].id,
            away_team_id=sample_teams[1].id,
            status=MatchStatus.SCHEDULED,
            played_at=_now(),
        )
        db_session.add(scheduled)
        await db_session.flush()

        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert "scenario-table" not in response.text
        assert "Run Scenario" not in response.text

    @pytest.mark.asyncio
    async def test_scenario_explorer_hidden_without_scheduled_matches(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
        sample_matches,
    ) -> None:
        """Scenario explorer should NOT render without scheduled matches."""
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": [
                    0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                    for j in range(len(sample_teams))
                ],
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)
        await db_session.flush()

        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        assert "scenario-table" not in response.text
        assert "Run Scenario" not in response.text


class TestKeyMatchesOnLeaguePage:
    """Feature 4: Key matches (match significance) on the league detail page."""

    @pytest.mark.asyncio
    async def test_key_matches_render_with_meta_data(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Key matches section should render when __meta__.key_matches exists."""
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": [
                    0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                    for j in range(len(sample_teams))
                ],
            }
            for i, team in enumerate(sample_teams)
        }
        fake_results["__meta__"] = {
            "key_matches": [
                {
                    "home_id": sample_teams[0].id,
                    "away_id": sample_teams[1].id,
                    "home_name": sample_teams[0].name,
                    "away_name": sample_teams[1].name,
                    "significance_score": 5.2,
                    "shift_home_win": 0.8,
                    "shift_draw": 0.3,
                    "shift_away_win": -0.5,
                },
                {
                    "home_id": sample_teams[2].id,
                    "away_id": sample_teams[3].id,
                    "home_name": sample_teams[2].name,
                    "away_name": sample_teams[3].name,
                    "significance_score": 3.1,
                    "shift_home_win": 0.4,
                    "shift_draw": 0.1,
                    "shift_away_win": -0.3,
                },
            ]
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)
        await db_session.flush()

        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        text = response.text
        assert "Key Matches" in text
        assert "significance-badge" in text
        assert "Significance" in text
        assert sample_teams[0].name in text
        assert sample_teams[1].name in text
        assert "If Home Win" in text
        assert "If Draw" in text
        assert "If Away Win" in text

    @pytest.mark.asyncio
    async def test_key_matches_hidden_without_meta(
        self,
        async_client: AsyncClient,
        db_session,
        sample_league: League,
        sample_teams: list[Team],
        sample_team_seasons,
    ) -> None:
        """Key matches section should NOT render when __meta__ is absent."""
        fake_results = {
            team.id: {
                "mean_pos": float(i + 1),
                "mean_points": float(60 - i * 5),
                "pos_dist": [
                    0.7 if j == i else 0.1 / max(len(sample_teams) - 1, 1)
                    for j in range(len(sample_teams))
                ],
            }
            for i, team in enumerate(sample_teams)
        }
        prediction = Prediction(
            id=_uuid(),
            league_id=sample_league.id,
            season=sample_league.current_season,
            generated_at=_now(),
            simulation_runs=10_000,
            results=fake_results,
        )
        db_session.add(prediction)
        await db_session.flush()

        response = await async_client.get(f"/league/{sample_league.id}")
        assert response.status_code == 200
        # "Key Matches" also appears in HTML comments outside the conditional,
        # so check for elements that only render inside {% if key_matches %}
        assert "significance-badge" not in response.text
        assert "If Home Win" not in response.text


class TestAdminImportPage:
    @pytest.mark.asyncio
    async def test_admin_import_returns_200(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get("/admin/import")
        assert response.status_code == 200
