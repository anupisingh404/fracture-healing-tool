from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    tavily_api_key: str = ""
    chroma_persist_dir: str = "./chroma_db"
    model_save_dir: str = "./backend/ml/saved_models"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    best_ml_model: str = "XGBoost"
    log_level: str = "INFO"
    anonymized_telemetry: str = "false"
    retrain_every_n: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
