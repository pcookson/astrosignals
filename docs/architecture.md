# AstroSignals Architecture

```mermaid
flowchart LR
    User["User Browser<br/>localhost:5173"] -->|Open UI| Web["Web App<br/>Vue 3 + Vite<br/>apps/web"]
    Web -->|GET /api/health| API["API Service<br/>FastAPI + Uvicorn<br/>apps/api"]
    Web -->|POST /api/ingest<br/>target, mission, author, sector| API

    API -->|search lightcurve| LK["lightkurve Library"]
    LK -->|Query mission archives| Archives["External Mission Data Systems<br/>MAST / TESS / Kepler catalogs"]
    LK -->|download first result| API

    API -->|JSON time flux flux_err n_points| Web
    Web -->|Render interactive chart| Plotly["Plotly.js (frontend)"]

    subgraph Compose["Docker Compose Network"]
      Web
      API
    end
```

## API Callouts

- `GET /api/health`
  - Purpose: connectivity check from web app.
  - Response: `{ "ok": true }`
- `POST /api/ingest`
  - Input: `{ "target", "mission", "author", "sector" }`
  - Behavior: searches and downloads one light curve via `lightkurve`, cleans NaNs, normalizes flux by median.
  - Response: metadata + arrays (`time`, `flux`, optional `flux_err`).

## External Systems

- `lightkurve` Python package for discovery/download of light curves.
- Mission archives queried through lightkurve (for TESS/Kepler products).
