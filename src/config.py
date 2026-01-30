"""
Configuration management for the fraud detection system
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    api_workers: int = Field(default=4, validation_alias="API_WORKERS")
    api_reload: bool = Field(default=True, validation_alias="API_RELOAD")

    # Environment
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")

    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4", validation_alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, validation_alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=500, validation_alias="LLM_MAX_TOKENS")

    # Vector Database
    vector_db_type: str = Field(default="chromadb", validation_alias="VECTOR_DB_TYPE")
    vector_db_path: str = Field(default="./data/vector_db", validation_alias="VECTOR_DB_PATH")
    pinecone_api_key: Optional[str] = Field(default=None, validation_alias="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, validation_alias="PINECONE_ENVIRONMENT")

    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/fraud_db",
        validation_alias="DATABASE_URL"
    )
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")

    # Model Configuration
    model_type: str = Field(default="xgboost", validation_alias="MODEL_TYPE")
    model_path: str = Field(default="./models/fraud_model.pkl", validation_alias="MODEL_PATH")
    threshold: float = Field(default=0.5, validation_alias="THRESHOLD")
    batch_size: int = Field(default=32, validation_alias="BATCH_SIZE")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", validation_alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="fraud-detection", validation_alias="MLFLOW_EXPERIMENT_NAME")

    # AWS
    aws_access_key_id: Optional[str] = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, validation_alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    s3_bucket: Optional[str] = Field(default=None, validation_alias="S3_BUCKET")

    # Security
    secret_key: str = Field(default="change-this-secret-key", validation_alias="SECRET_KEY")
    algorithm: str = Field(default="HS256", validation_alias="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Monitoring
    prometheus_port: int = Field(default=9090, validation_alias="PROMETHEUS_PORT")
    enable_metrics: bool = Field(default=True, validation_alias="ENABLE_METRICS")

    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: str = Field(default="json", validation_alias="LOG_FORMAT")

    # Performance
    cache_ttl: int = Field(default=3600, validation_alias="CACHE_TTL")
    max_concurrent_requests: int = Field(default=100, validation_alias="MAX_CONCURRENT_REQUESTS")

    # Feature Flags
    enable_shap_explanation: bool = Field(default=True, validation_alias="ENABLE_SHAP_EXPLANATION")
    enable_rag: bool = Field(default=True, validation_alias="ENABLE_RAG")
    enable_audit_logging: bool = Field(default=True, validation_alias="ENABLE_AUDIT_LOGGING")


# Global settings instance
settings = Settings()
