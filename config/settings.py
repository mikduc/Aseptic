"""Application configuration and environment settings."""
import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment."""

    # LLM Configuration
    LLM_PROVIDER: Literal["openai", "ollama"] = os.getenv(
        "LLM_PROVIDER", "openai"
    )
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Local LLM (Ollama)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    SAMPLE_ROWS: int = int(os.getenv("SAMPLE_ROWS", "5"))


settings = Settings()
