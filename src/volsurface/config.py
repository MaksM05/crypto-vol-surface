"""Application settings loaded from environment via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the volsurface package.

    Defaults match ``docker-compose.yml`` so a clean clone followed by
    ``docker compose up -d`` works without any extra environment.

    Override via env vars prefixed with ``VOLSURFACE_``
    (e.g. ``VOLSURFACE_DB_HOST=db.example.com``) or via a ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="VOLSURFACE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "volsurface"
    db_password: str = "volsurface_dev"
    db_name: str = "volsurface"
    db_pool_min: int = 1
    db_pool_max: int = 10

    # Deribit ingestion
    deribit_rest_url: str = "https://www.deribit.com/api/v2"
    deribit_ws_url: str = "wss://www.deribit.com/ws/api/v2"
    deribit_http_timeout_s: float = 10.0
    poll_interval_s: int = 300
    ws_backoff_initial_s: float = 1.0
    ws_backoff_max_s: float = 60.0

    @property
    def database_url(self) -> str:
        """Return an asyncpg-compatible Postgres DSN."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
