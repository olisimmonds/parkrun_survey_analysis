"""
parkrun Insights Platform — FastAPI application entry point.

Start for development:
  cd backend && uvicorn app.main:app --reload --port 8000

With background worker (local dev only):
  cd backend && uvicorn app.main:app --reload --port 8000 &
  python -m app.worker.pipeline
"""
from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import chat, datasets, ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("parkrun Insights API starting — embedding provider: %s", settings.embedding_provider)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="parkrun Insights API",
    description=(
        "AI-powered survey insights platform for parkrun Digital Ambassadors. "
        "Ingests survey exports, builds a compounding knowledge base, and answers "
        "natural-language questions about the parkrun community."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

_origins = settings.cors_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # credentials=True requires explicit origins (not "*")
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(datasets.router)
app.include_router(chat.router)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "service": "parkrun-insights-api"}


