# src/agent_runtime/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class AppConfig(BaseSettings):
    # 默认使用 Fake Provider，真实生产可切换为 openai 等
    PROVIDER_TYPE: str = "fake"
    MAX_GLOBAL_STEPS: int = 8
    DEFAULT_CONCURRENCY: int = 2
    
    class Config:
        env_file = ".env"

settings = AppConfig()