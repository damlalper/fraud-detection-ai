"""
Configuration management for the fraud detection system
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    api_reload: bool = Field(default=True, env="API_RELOAD")

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")

    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=500, env="LLM_MAX_TOKENS")

    # Vector Database
    vector_db_type: str = Field(default="chromadb", env="VECTOR_DB_TYPE")
    vector_db_path: str = Field(default="./data/vector_db", env="VECTOR_DB_PATH")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")

    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/fraud_db",
        env="DATABASE_URL"
    )
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Model Configuration
    model_type: str = Field(default="xgboost", env="MODEL_TYPE")
    model_path: str = Field(default="./models/fraud_model.pkl", env="MODEL_PATH")
    threshold: float = Field(default=0.5, env="THRESHOLD")
    batch_size: int = Field(default=32, env="BATCH_SIZE")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="fraud-detection", env="MLFLOW_EXPERIMENT_NAME")

    # AWS
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    s3_bucket: Optional[str] = Field(default=None, env="S3_BUCKET")

    # Security
    secret_key: str = Field(default="change-this-secret-key", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Monitoring
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")

    # Performance
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")

    # Feature Flags
    enable_shap_explanation: bool = Field(default=True, env="ENABLE_SHAP_EXPLANATION")
    enable_rag: bool = Field(default=True, env="ENABLE_RAG")
    enable_audit_logging: bool = Field(default=True, env="ENABLE_AUDIT_LOGGING")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
