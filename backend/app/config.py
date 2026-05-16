from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve to the repo root regardless of the working directory when the backend starts.
_ROOT_ENV = str(Path(__file__).parent.parent.parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ROOT_ENV,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str

    # LLM — Groq (router, classifier, wiki maintainer)
    groq_api_key: str

    # LLM — Together AI (optional; used for hosted nomic-embed-text)
    together_api_key: str = ""

    # Embedding provider: "local" (sentence-transformers) or "together"
    embedding_provider: str = "local"

    # Embedding model name (used for both local and Together AI paths)
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dim: int = 768

    # Groq model names
    groq_fast_model: str = "llama-3.1-8b-instant"       # classifier, router, map batches
    groq_capable_model: str = "llama-3.3-70b-versatile"  # wiki maintainer, synthesiser

    # Worker
    worker_poll_interval_seconds: int = 10
    max_job_retries: int = 3

    # CORS: comma-separated list of allowed origins
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
