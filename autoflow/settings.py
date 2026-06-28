from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    database_url: str = "sqlite:///./autoflow.db"
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "autoflow"
    redis_ttl_seconds: int = 604800
    checkpoint_backend: str = "auto"
    checkpoint_ttl_seconds: int = 604800
    kali_host: str = "127.0.0.1"
    kali_port: int = 22
    kali_username: str = "kali"
    kali_password: str = ""
    kali_key_path: str = ""
    vector_store_provider: str = "chroma"
    vector_store_path: str = "./data/vectorstore"
    llm_model: str = "gpt-5.4"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.freemodel.dev/v1"
    llm_streaming: bool = True
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 1
    llm_disable_thinking: bool = False
    llm_disable_thinking_for_json: bool = True


settings = Settings()
