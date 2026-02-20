# AstroSignals
## Goals:
##      Feed time-series photometry into a model
##      Detect: Transit events, Variable stars, Possible supernova signatures

## Architecture

- See `docs/architecture.md` for the component diagram, API callouts, and external system dependencies.

## Run

```bash
docker compose up --build
```

## Open

- Web: http://localhost:5173
- API health: http://localhost:8000/api/health

## API Base URL Config

- Docker Compose sets `VITE_API_BASE_URL=http://api:8000` for the web service.
- The web app requests `/api/health` from Vite, and Vite proxies `/api` to `VITE_API_BASE_URL`.
- If you run web locally outside Docker, set:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Expected Web Status

- `Checking API...` while loading
- `API connected ✅` when `/api/health` returns `{ "ok": true }`
- `API not reachable ❌` if the request fails or `ok` is false


## Cache (Story 1.3)

- Disk cache is stored at `./data/cache` on the host.
- In Docker Compose, `./data` is mounted into the API container at `/app/data`.
- Clear cache by deleting `./data/cache`.

## Story 1.1 Usage

Use the web form and submit:

- target: `TIC 25155310`
- mission: `TESS`
- author: `SPOC`
- sector: blank

The app sends `POST /api/ingest`, then plots normalized flux vs time.
The first ingestion may take longer because the light curve has to be downloaded.
