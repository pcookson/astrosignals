import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import CACHE_DIR
from tess import router as tess_router
from ztf import router as ztf_router

logger = logging.getLogger("astrosignals.api")

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(tess_router)
app.include_router(ztf_router)


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}
