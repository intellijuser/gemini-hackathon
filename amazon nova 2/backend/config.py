"""
Nova DueDiligence SaaS — Centralized Configuration
Gemini edition — all AWS/Bedrock settings replaced with Google Cloud settings.
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/nova_saas"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@localhost:5432/nova_saas"

    # ── Redis / Celery ────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Auth ──────────────────────────────────────────────────────────────────
    JWT_SECRET: str = "CHANGE_ME_IN_PRODUCTION"
    JWT_LIFETIME_SECONDS: int = 86400

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # ── Google Gemini ─────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"         # swap to gemini-1.5-pro for max quality
    GEMINI_EMBED_MODEL: str = "models/text-embedding-004"
    GEMINI_EMBED_DIMENSIONS: int = 768              # text-embedding-004 native dimension

    # ── Document Processing ───────────────────────────────────────────────────
    MAX_UPLOAD_MB: int = 30
    MAX_CHUNKS_PER_DOC: int = 150
    CHUNK_SIZE: int = 3000
    CHUNK_OVERLAP: int = 200
    CONTEXT_WINDOW_CHARS: int = 22000
    SEMANTIC_TOP_K: int = 6
    SEMANTIC_THRESHOLD: float = 0.1

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()