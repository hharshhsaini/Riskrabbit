from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, read from environment variables or backend/.env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str
    GITHUB_TOKEN: str
    OPENAI_API_KEY: str
    MODEL_PATH: str = "ml/model.json"
    DEFAULT_USER_EMAIL: str = "dev@example.com"
    ENV: str = "development"


settings = Settings()
