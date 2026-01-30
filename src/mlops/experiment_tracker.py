"""
MLflow Experiment Tracking for Model Lifecycle Management
Handles: model development, training, testing, deployment, and monitoring
"""
import os
import json
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("mlops")


class ExperimentTracker:
    """MLflow-based experiment tracking for fraud detection models"""

    def __init__(
        self,
        experiment_name: str = "fraud-detection",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize experiment tracker

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI (default: local)
            artifact_location: Where to store artifacts
        """
        self.experiment_name = experiment_name

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Local tracking
            project_root = Path(__file__).parent.parent.parent
            mlflow_dir = project_root / "mlruns"
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlflow_dir}")

        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracker initialized: {experiment_name}")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start a new MLflow run"""
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        default_tags = {
            "project": "fraud-detection",
            "environment": os.getenv("ENVIRONMENT", "development")
        }
        if tags:
            default_tags.update(tags)

        self.active_run = mlflow.start_run(run_name=run_name, tags=default_tags)
        logger.info(f"Started run: {run_name} (ID: {self.active_run.info.run_id})")

        return self.active_run.info.run_id

    def end_run(self):
        """End the current run"""
        mlflow.end_run()
        logger.info("Run ended")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.info(f"Logged {len(metrics)} metrics")

    def log_model_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Calculate and log classification metrics"""
        metrics = {
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
            f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
            f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0)
        }

        if y_proba is not None:
            try:
                metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_proba)
            except:
                pass

        self.log_metrics(metrics)

        # Log confusion matrix as artifact
        cm = confusion_matrix(y_true, y_pred)
        cm_dict = {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        }
        self.log_artifact_dict(cm_dict, "confusion_matrix.json")

        return metrics

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as artifact"""
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def log_artifact_dict(self, data: Dict, filename: str):
        """Log a dictionary as JSON artifact"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name

        mlflow.log_artifact(temp_path)
        os.unlink(temp_path)

    def log_sklearn_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """Log sklearn/XGBoost model"""
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info(f"Logged sklearn model: {artifact_path}")

    def log_pytorch_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """Log PyTorch model"""
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info(f"Logged PyTorch model: {artifact_path}")

    def log_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        importance_type: str = "gain"
    ):
        """Log feature importance"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)

        # Log as artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            importance_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, f"feature_importance_{importance_type}")
            os.unlink(f.name)

        # Log top features as params
        for i, row in importance_df.head(10).iterrows():
            mlflow.log_param(f"top_feature_{i+1}", row['feature'])

        logger.info(f"Logged feature importance ({importance_type})")

    def log_training_data_info(self, X: pd.DataFrame, y: pd.Series):
        """Log training data statistics"""
        info = {
            "num_samples": len(X),
            "num_features": len(X.columns),
            "fraud_ratio": float(y.mean()),
            "class_0_count": int((y == 0).sum()),
            "class_1_count": int((y == 1).sum())
        }
        self.log_params(info)
        logger.info("Logged training data info")

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """Register model in MLflow Model Registry"""
        result = mlflow.register_model(model_uri, name)
        logger.info(f"Registered model: {name} (version {result.version})")
        return result

    def load_model(self, model_uri: str, model_type: str = "sklearn"):
        """Load a model from MLflow"""
        if model_type == "sklearn":
            return mlflow.sklearn.load_model(model_uri)
        elif model_type == "pytorch":
            return mlflow.pytorch.load_model(model_uri)
        else:
            return mlflow.pyfunc.load_model(model_uri)

    def get_best_run(self, metric: str = "roc_auc", ascending: bool = False) -> Dict:
        """Get the best run based on a metric"""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )

        if len(runs) > 0:
            return runs.iloc[0].to_dict()
        return {}


def track_experiment(func):
    """Decorator for tracking experiments"""
    def wrapper(*args, **kwargs):
        tracker = ExperimentTracker()
        run_id = tracker.start_run(run_name=func.__name__)

        try:
            result = func(*args, tracker=tracker, **kwargs)
            return result
        except Exception as e:
            mlflow.log_param("error", str(e))
            raise
        finally:
            tracker.end_run()

    return wrapper


if __name__ == "__main__":
    # Demo
    logger.info("="*70)
    logger.info("MLflow Experiment Tracking Demo")
    logger.info("="*70)

    # Initialize tracker
    tracker = ExperimentTracker(experiment_name="fraud-detection-demo")

    # Start run
    tracker.start_run(run_name="demo_run", tags={"model_type": "xgboost"})

    # Log parameters
    tracker.log_params({
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    })

    # Simulate metrics
    tracker.log_metrics({
        "train_accuracy": 0.95,
        "val_accuracy": 0.92,
        "roc_auc": 0.89
    })

    # End run
    tracker.end_run()

    logger.info("\nMLflow tracking demo completed!")
    logger.info(f"View results at: {mlflow.get_tracking_uri()}")
