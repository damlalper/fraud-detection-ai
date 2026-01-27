"""
FastAPI Backend for Fraud Detection System
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    TransactionRequest, PredictionResponse, ExplanationResponse,
    BatchTransactionRequest, BatchPredictionResponse,
    HealthResponse, ErrorResponse, ModelInfo, RiskFactor
)
from explainability.fraud_explainer_pipeline import FraudExplanationPipeline
from config import settings
from utils.logger import setup_logger

logger = setup_logger("api")

# Global state
app_state = {
    "model_pipeline": None,
    "model_loaded": False,
    "start_time": datetime.utcnow()
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("="*70)
    logger.info("Starting Fraud Detection API")
    logger.info("="*70)

    try:
        # Load model pipeline
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "models" / "xgboost_fraud"

        logger.info(f"Loading model from: {model_path}")

        pipeline = FraudExplanationPipeline(
            model_path=str(model_path),
            model_type="xgboost",
            use_llm=settings.enable_rag,  # Use LLM if RAG enabled
            use_rag=settings.enable_rag,
            llm_provider=settings.llm_provider
        )

        # Fit SHAP explainer
        if settings.enable_shap_explanation:
            data_dir = project_root / "data" / "processed"
            if data_dir.exists():
                X_train = pd.read_parquet(data_dir / "X_train.parquet")
                pipeline.fit_explainer(X_train.sample(n=min(500, len(X_train))))
                logger.info("✓ SHAP explainer fitted")

        app_state["model_pipeline"] = pipeline
        app_state["model_loaded"] = True

        logger.info("✓ API ready to serve requests")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app_state["model_loaded"] = False

    yield

    # Shutdown
    logger.info("Shutting down Fraud Detection API")


# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Fraud Detection API",
    description="Real-time fraud detection with explainable AI and LLM-powered explanations",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline():
    """Dependency to get model pipeline"""
    if not app_state["model_loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable."
        )
    return app_state["model_pipeline"]


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI-Powered Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if app_state["model_loaded"] else "unhealthy",
        version="1.0.0",
        model_loaded=app_state["model_loaded"],
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info(pipeline: FraudExplanationPipeline = Depends(get_pipeline)):
    """Get model information"""
    return ModelInfo(
        model_type=pipeline.model_type,
        num_features=len(pipeline.feature_names),
        threshold=pipeline.threshold,
        metrics=None  # Can add saved metrics here
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(
    request: TransactionRequest,
    pipeline: FraudExplanationPipeline = Depends(get_pipeline)
):
    """
    Predict fraud for a single transaction

    Returns fraud probability and classification
    """
    try:
        start_time = time.time()

        # Convert features to DataFrame
        X = pd.DataFrame([request.features])

        # Ensure all required features are present
        missing_features = set(pipeline.feature_names) - set(X.columns)
        if missing_features:
            # Fill missing features with 0
            for feature in missing_features:
                X[feature] = 0.0

        # Reorder columns to match training
        X = X[pipeline.feature_names]

        # Get prediction
        if pipeline.model_type == "xgboost":
            fraud_prob = pipeline.model.predict_proba(X)[0, 1]
        else:
            fraud_prob = pipeline.model.predict_proba(X)[0]

        is_fraud = fraud_prob >= pipeline.threshold
        confidence = abs(fraud_prob - 0.5) * 2

        # Generate transaction ID if not provided
        transaction_id = request.transaction_id or f"TXN_{int(time.time()*1000)}"

        processing_time = (time.time() - start_time) * 1000  # ms

        logger.info(
            f"Prediction: {transaction_id} | "
            f"Prob: {fraud_prob:.3f} | "
            f"Fraud: {is_fraud} | "
            f"Time: {processing_time:.1f}ms"
        )

        return PredictionResponse(
            transaction_id=transaction_id,
            fraud_probability=float(fraud_prob),
            is_fraud=bool(is_fraud),
            confidence=float(confidence),
            threshold=float(pipeline.threshold),
            model_type=pipeline.model_type,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/explain", response_model=ExplanationResponse, tags=["Explanation"])
async def explain_fraud(
    request: TransactionRequest,
    include_llm: bool = True,
    include_rag: bool = True,
    pipeline: FraudExplanationPipeline = Depends(get_pipeline)
):
    """
    Predict fraud and generate detailed explanation

    Includes:
    - Fraud prediction
    - SHAP-based risk factors
    - LLM-powered explanation (optional)
    - RAG policy context (optional)
    """
    try:
        start_time = time.time()

        # Convert features to DataFrame
        X = pd.DataFrame([request.features])

        # Ensure all required features are present
        missing_features = set(pipeline.feature_names) - set(X.columns)
        if missing_features:
            for feature in missing_features:
                X[feature] = 0.0

        X = X[pipeline.feature_names]

        # Generate transaction ID
        transaction_id = request.transaction_id or f"TXN_{int(time.time()*1000)}"
        X.index = [transaction_id]

        # Check if SHAP explainer is available
        if pipeline.shap_explainer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SHAP explainer not initialized. Explanations unavailable."
            )

        # Generate full explanation
        explanation = pipeline.explain_transaction(
            X,
            include_llm=include_llm and settings.enable_rag,
            include_rag=include_rag and settings.enable_rag
        )

        # Extract prediction info
        pred = explanation['prediction']

        # Format risk factors
        risk_factors = [
            RiskFactor(
                feature=f['feature'],
                value=f['value'],
                shap_value=f['shap_value'],
                impact=f['impact'],
                magnitude=f['magnitude']
            )
            for f in explanation['shap_analysis']['top_features'][:10]
        ]

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Explanation: {transaction_id} | "
            f"Prob: {pred['fraud_probability']:.3f} | "
            f"Factors: {len(risk_factors)} | "
            f"Time: {processing_time:.1f}ms"
        )

        return ExplanationResponse(
            transaction_id=transaction_id,
            prediction=PredictionResponse(
                transaction_id=transaction_id,
                fraud_probability=pred['fraud_probability'],
                is_fraud=pred['is_fraud'],
                confidence=pred['confidence'],
                threshold=pred['threshold'],
                model_type=pipeline.model_type,
                timestamp=datetime.utcnow().isoformat()
            ),
            top_risk_factors=risk_factors,
            llm_explanation=explanation.get('llm_explanation'),
            policy_context=explanation.get('policy_context'),
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failed: {str(e)}"
        )


@app.post("/batch/predict", response_model=BatchPredictionResponse, tags=["Batch"])
async def batch_predict(
    request: BatchTransactionRequest,
    pipeline: FraudExplanationPipeline = Depends(get_pipeline)
):
    """
    Batch fraud prediction for multiple transactions

    Optimized for high throughput
    """
    try:
        start_time = time.time()

        predictions = []
        fraud_count = 0
        legitimate_count = 0

        for txn in request.transactions:
            X = pd.DataFrame([txn.features])

            # Handle missing features
            missing_features = set(pipeline.feature_names) - set(X.columns)
            if missing_features:
                for feature in missing_features:
                    X[feature] = 0.0

            X = X[pipeline.feature_names]

            # Predict
            if pipeline.model_type == "xgboost":
                fraud_prob = pipeline.model.predict_proba(X)[0, 1]
            else:
                fraud_prob = pipeline.model.predict_proba(X)[0]

            is_fraud = fraud_prob >= pipeline.threshold
            confidence = abs(fraud_prob - 0.5) * 2

            if is_fraud:
                fraud_count += 1
            else:
                legitimate_count += 1

            transaction_id = txn.transaction_id or f"TXN_{int(time.time()*1000)}_{len(predictions)}"

            predictions.append(
                PredictionResponse(
                    transaction_id=transaction_id,
                    fraud_probability=float(fraud_prob),
                    is_fraud=bool(is_fraud),
                    confidence=float(confidence),
                    threshold=float(pipeline.threshold),
                    model_type=pipeline.model_type,
                    timestamp=datetime.utcnow().isoformat()
                )
            )

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Batch prediction: {len(predictions)} transactions | "
            f"Fraud: {fraud_count} | "
            f"Legitimate: {legitimate_count} | "
            f"Time: {processing_time:.1f}ms"
        )

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            fraud_count=fraud_count,
            legitimate_count=legitimate_count,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
