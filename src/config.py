"""PRISM Analyst Services — Configuration Module.

Centralized configuration using Pydantic Settings.
All environment variables are defined here with defaults.
Supports AWS RDS, Google Cloud SQL, or any PostgreSQL instance.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Server ──
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # ── Database (PostgreSQL — provider-agnostic) ──
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "postgres"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_SSL_MODE: str = "prefer"
    DB_SSL_ROOT_CERT: str = ""

    # ── LLM: Gemini API Keys ──
    GEMINI_API_KEY: str = ""
    GEMINI_API_KEY_1: str = ""
    GEMINI_API_KEY_2: str = ""
    GEMINI_API_KEY_3: str = ""
    GEMINI_API_KEY_4: str = ""

    # ── LLM: OpenRouter (fallback) ──
    OPENROUTER_API_KEY: str = ""

    # ── Web Search ──
    TAVILY_API_KEY: str = ""

    # ── Feature Flags ──
    AUTH_ENABLED: bool = False

    @property
    def database_url(self) -> str:
        """Build async PostgreSQL connection string."""
        ssl = f"?ssl={self.DB_SSL_MODE}" if self.DB_SSL_MODE != "disable" else ""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}{ssl}"

    @property
    def gemini_api_keys(self) -> list[str]:
        """Collect all non-empty Gemini API keys for round-robin."""
        keys = [self.GEMINI_API_KEY, self.GEMINI_API_KEY_1, self.GEMINI_API_KEY_2, self.GEMINI_API_KEY_3, self.GEMINI_API_KEY_4]
        return [k for k in keys if k]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
