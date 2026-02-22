# AstroSignals

AstroSignals is a web application for exploring TESS and Kepler light-curve data and supporting workflows for identifying potential astronomical anomalies in time-series photometry.

Current focus:

- Ingest TESS/Kepler light curves
- Visualize normalized flux over time
- Support anomaly discovery workflows (for example, potential exoplanet candidates)

## What This App Is For

This project is intended as a lightweight interface for fetching and inspecting space-telescope photometry before model-based analysis. The goal is to make it easy to:

- query a target from TESS/Kepler sources,
- inspect the returned light curve, and
- search for anomalous behavior in the light curve (including potential exoplanet-candidate signals).

## Data Manipulation Transparency (TESS/Kepler)

Scientific transparency matters, so this README documents how the `POST /api/ingest` pipeline preprocesses downloaded TESS/Kepler light curves:

- Filters to finite values for `time` and `flux`
- Filters `flux_err` to finite values when uncertainty is present
- Applies quality masking when a `quality` column is available (keeps samples with `quality == 0`)
- Continues with finite-only filtering if quality flags are unavailable, and logs that condition
- Normalizes flux by the median flux value:
  - `flux_norm = flux / median_flux`
- Normalizes uncertainty using the same median-flux scale (when `flux_err` exists):
  - `flux_err_norm = flux_err / median_flux`
- Returns time-system metadata in the API response `meta` (for example `time_format`, `time_scale`, `time_zero_point`, `time_system_source`)
- Logs cadence sanity statistics from filtered time values (including median cadence in seconds)

These details are included so users can interpret plots and downstream results with the ingestion/preprocessing steps clearly documented.

## Run Locally

```bash
docker compose up --build
```

Open:

- Web: `http://localhost:5173`
- API health: `http://localhost:8000/api/health`

## Example Usage

Use the web form with a TESS target such as:

- target: `TIC 25155310`
- mission: `TESS`
- author: `SPOC`
- sector: blank

The app sends `POST /api/ingest` and plots normalized flux vs. time.
The first request may take longer because the light curve must be downloaded and cached.

## Local Development Notes

### Python Environment (API)

Use the API virtual environment at `apps/api/.venv` for local Python commands.

```bash
cd apps/api
make bootstrap
make which-python
make test
make run
```

Expected interpreter:

```bash
<project-root>/apps/api/.venv/bin/python
```

### API Base URL (Web)

- Docker Compose sets `VITE_API_BASE_URL=http://api:8000` for the web service.
- Vite proxies `/api` requests to `VITE_API_BASE_URL`.
- If running the web app outside Docker, set:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

### Cache

- Disk cache is stored at `./data/cache`
- In Docker Compose, `./data` is mounted into the API container at `/app/data`
- Clear cache by deleting `./data/cache`

## Architecture

- See `docs/architecture.md` for the component diagram, API callouts, and external system dependencies.
