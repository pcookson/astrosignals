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

## Source Selector (Story S0.1)

- The web app includes a top-level source selector:
  - `TESS/Kepler` uses `POST /api/ingest` (existing behavior).
  - `ZTF` uses `POST /api/ingest/ztf`.
- Source selection is stored in the URL query string (`?source=tess` or `?source=ztf`) so refresh keeps state.
- ZTF ingestion is currently a backend stub and returns `501 Not Implemented` until Story Z1.1.

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

## TESS Data Manipulation Transparency

`POST /api/ingest` applies these transformations to downloaded TESS/Kepler light curves:

- Filters to finite samples for `time` and `flux`, and for `flux_err` when present.
- Applies TESS quality masking when a `quality` column is available: keeps only `quality == 0`.
- If quality is unavailable, continues with finite-only filtering and logs that quality was unavailable.
- Normalizes by median flux: `flux_norm = flux / median_flux`.
- Normalizes uncertainty with the same scale: `flux_err_norm = flux_err / median_flux` (when `flux_err` exists).
- Returns time-system metadata in API response `meta` (`time_format`, `time_scale`, `time_zero_point`, `time_system_source`).
- Logs cadence sanity statistics from filtered time values, including median cadence in seconds.

## ZTF Usage (Story Z1.1)

- ZTF ingestion endpoint: `POST /api/ingest/ztf`
- Z1.1 currently supports either:
  - `ra` + `dec` (+ optional `radiusArcsec`), or
  - numeric `objectId` (digits-only)
- `ZTF18...` style object names are not supported yet.
- Example positional query values to try:
  - `ra=298.0025`
  - `dec=29.87147`
  - `radiusArcsec=5`
  - `band=r`
- The IRSA ZTF API is public and may rate limit. Keep requests reasonable.
