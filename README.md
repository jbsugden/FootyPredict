# FootyPredict

Predicts final football league tables using a Dixon-Coles Poisson model with Monte Carlo simulation (10,000 runs). Covers two leagues:

- **Premier League** — Manchester United et al., via the football-data.org API
- **Northern Premier League West** — Bury FC et al., via the official NPL JSON API (thenpl.co.uk)

## How it works

1. **Data sync** — Pulls all finished results and upcoming fixtures for each league
2. **Team strengths** — Computes attack/defence ratings from form-weighted historical results
3. **Monte Carlo** — Simulates the remaining fixtures 10,000 times using Dixon-Coles corrected Poisson distributions
4. **Predicted table** — Aggregates simulations into a final-position probability distribution per team

## Stack

| Layer | Tech |
|---|---|
| Backend | Python 3.12, FastAPI, async SQLAlchemy + SQLite |
| Prediction engine | NumPy, SciPy (Poisson), Dixon-Coles correction, ELO ratings |
| Frontend | Jinja2, HTMX, Alpine.js, Chart.js, Pico CSS |
| Scheduling | APScheduler (nightly sync at 02:00 UTC, predict at 03:00 UTC) |

## Data sources

| League | Source | Auth |
|---|---|---|
| Premier League | [football-data.org](https://football-data.org) v4 API | Free API key (X-Auth-Token header) |
| NPL West | [thenpl.co.uk](https://thenpl.co.uk) JSON API | None (X-TENANT-ID: npl) |

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/jbsugden/FootyPredict.git
cd FootyPredict
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env — add your football-data.org API key

# 3. Bootstrap the database
uv run python -m predictor.seed

# 4. Run the server
uv run python -m predictor.api.app
# → http://localhost:8000
```

## Initial data load

After first run, trigger a data sync and prediction via the admin API:

```bash
# Sync both leagues (fetches results + fixtures, rebuilds standings + ELO)
curl -X POST http://localhost:8000/admin/sync \
  -H "X-Admin-Key: <your-admin-key>"

# Run predictions for Premier League
curl -X POST http://localhost:8000/api/predictions/<league-id>/run \
  -H "X-Admin-Key: <your-admin-key>"
```

Or visit the web UI at `http://localhost:8000` and use the Re-run button on each league page.

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `FOOTBALL_DATA_API_KEY` | football-data.org API key | (required) |
| `DATABASE_URL` | Async SQLAlchemy URL | `sqlite+aiosqlite:///./footypredict.db` |
| `ADMIN_API_KEY` | Key for protected admin endpoints | `change_me` |
| `DEBUG` | Enable verbose logging + auto-reload | `false` |

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/leagues` | List all tracked leagues |
| GET | `/api/leagues/{id}/table` | Current standings for a league |
| GET | `/api/predictions/{id}` | Latest Monte Carlo prediction |
| POST | `/api/predictions/{id}/run` | Trigger new simulation (admin) |
| POST | `/admin/sync` | Force data refresh from all sources (admin) |
| POST | `/admin/import-csv` | Import results from CSV file (admin) |
| GET | `/api/docs` | Interactive API docs (Swagger UI) |

## Project structure

```
src/predictor/
├── api/           FastAPI routes (leagues, predictions, admin)
├── data/          Data adapters (football-data.org API, NPL API)
├── db/            SQLAlchemy models, session, repositories
├── engine/        ELO, Poisson strengths, Dixon-Coles, Monte Carlo
├── web/           Jinja2 routes + HTML templates (HTMX + Alpine.js)
├── static/        CSS + Chart.js helpers
├── scheduler.py   APScheduler nightly jobs
├── seed.py        DB bootstrap script
└── config.py      Pydantic settings
```

## Sample predictions (March 2026)

**Premier League** (10,000 simulations, 8 games remaining)

| Predicted pos | Team | Mean pts |
|---|---|---|
| 1 | Arsenal FC | 87.2 |
| 2 | Manchester City FC | 74.4 |
| 3 | Manchester United FC | 66.8 |

**NPL West** (10,000 simulations, ~8 games remaining)

| Predicted pos | Team | Mean pts |
|---|---|---|
| 1 | Bury FC | 87.2 |
| 2 | Avro | 86.4 |
| 3 | Lower Breck | 80.7 |
