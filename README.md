# AstroSignals

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
