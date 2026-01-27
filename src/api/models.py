"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime


class TransactionRequest(BaseModel):
    """Request model for fraud prediction"""

    transaction_id: Optional[str] = Field(None, description="Optional transaction ID")
    features: Dict[str, float] = Field(..., description="Transaction features (V1-V28, Time, Amount, etc.)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN_12345",
                "features": {
                    "Time": 13547.0,
                    "Amount": 1234.56,
                    "V1": -1.359807,
                    "V2": -0.072781,
                    "V3": 2.536347,
                    "V4": 1.378155,
                    "V14": -19.214325
                }
            }
        }
    )


class RiskFactor(BaseModel):
    """Individual risk factor from SHAP analysis"""

    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value")
    shap_value: float = Field(..., description="SHAP contribution")
    impact: str = Field(..., description="Impact direction")
    magnitude: float = Field(..., description="Absolute impact")


class PredictionResponse(BaseModel):
    """Response model for fraud prediction"""

    transaction_id: str = Field(..., description="Transaction ID")
    fraud_probability: float = Field(..., description="Fraud probability (0-1)")
    is_fraud: bool = Field(..., description="Binary fraud classification")
    confidence: float = Field(..., description="Model confidence (0-1)")
    threshold: float = Field(..., description="Classification threshold")
    model_type: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN_12345",
                "fraud_probability": 0.87,
                "is_fraud": True,
                "confidence": 0.74,
                "threshold": 0.5,
                "model_type": "xgboost",
                "timestamp": "2026-01-25T01:30:00Z"
            }
        }
    )


class ExplanationResponse(BaseModel):
    """Response model for fraud explanation"""

    transaction_id: str
    prediction: PredictionResponse
    top_risk_factors: List[RiskFactor] = Field(..., description="Top contributing features")
    llm_explanation: Optional[str] = Field(None, description="AI-generated explanation")
    policy_context: Optional[str] = Field(None, description="Relevant fraud policies")
    timestamp: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN_12345",
                "prediction": {
                    "transaction_id": "TXN_12345",
                    "fraud_probability": 0.87,
                    "is_fraud": True,
                    "confidence": 0.74,
                    "threshold": 0.5,
                    "model_type": "xgboost",
                    "timestamp": "2026-01-25T01:30:00Z"
                },
                "top_risk_factors": [
                    {
                        "feature": "V14",
                        "value": -19.45,
                        "shap_value": 0.35,
                        "impact": "increases fraud risk",
                        "magnitude": 0.35
                    }
                ],
                "llm_explanation": "Transaction flagged as fraudulent...",
                "policy_context": "V14 values below -15 are suspicious...",
                "timestamp": "2026-01-25T01:30:00Z"
            }
        }
    )


class BatchTransactionRequest(BaseModel):
    """Request model for batch predictions"""

    transactions: List[TransactionRequest] = Field(..., description="List of transactions")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transactions": [
                    {
                        "transaction_id": "TXN_001",
                        "features": {"Time": 100, "Amount": 50.0, "V1": -1.2}
                    },
                    {
                        "transaction_id": "TXN_002",
                        "features": {"Time": 200, "Amount": 1500.0, "V1": 2.5}
                    }
                ]
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""

    predictions: List[PredictionResponse]
    total_processed: int
    fraud_count: int
    legitimate_count: int
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Model availability")
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error info")
    timestamp: str


class ModelInfo(BaseModel):
    """Model information response"""

    model_type: str
    num_features: int
    threshold: float
    training_date: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
