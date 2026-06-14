# src/agent_runtime/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class AppConfig(BaseSettings):
    # Use the fake provider by default; production can switch to OpenAI or another provider.
    PROVIDER_TYPE: str = "fake"
    MAX_GLOBAL_STEPS: int = 8
    DEFAULT_CONCURRENCY: int = 2
    model_config = SettingsConfigDict(env_file=".env")

settings = AppConfig()
