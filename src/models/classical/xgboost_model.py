"""
XGBoost Fraud Detection Model
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve
)
import joblib
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logger import setup_logger

logger = setup_logger("xgboost_model")


class XGBoostFraudDetector:
    """XGBoost-based fraud detection model"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = None,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            scale_pos_weight: Balance of positive and negative weights
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

        self.model = None
        self.feature_names = None
        self.best_threshold = 0.5

        logger.info("Initialized XGBoostFraudDetector")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ):
        """
        Train the XGBoost model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info("="*70)
        logger.info("Training XGBoost Model")
        logger.info("="*70)

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            self.scale_pos_weight = neg_count / pos_count
            logger.info(f"Calculated scale_pos_weight: {self.scale_pos_weight:.2f}")

        # Initialize model
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric='auc',
            use_label_encoder=False,
            tree_method='hist'  # Faster training
        )

        # Prepare evaluation set if provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        logger.info(f"Training on {len(X_train)} samples...")

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        # Get training metrics
        train_pred = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)

        logger.info(f"Training AUC: {train_auc:.4f}")

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred)
            logger.info(f"Validation AUC: {val_auc:.4f}")

        logger.info("✓ Training completed")

        return self

    def calibrate_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_metric: str = 'f1'
    ):
        """
        Find optimal classification threshold

        Args:
            X_val: Validation features
            y_val: Validation labels
            target_metric: Metric to optimize ('f1', 'precision', 'recall')
        """
        logger.info(f"Calibrating threshold to optimize {target_metric}...")

        y_pred_proba = self.model.predict_proba(X_val)[:, 1]

        best_score = 0
        best_threshold = 0.5

        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            if target_metric == 'f1':
                score = f1_score(y_val, y_pred, zero_division=0)
            elif target_metric == 'precision':
                score = precision_score(y_val, y_pred, zero_division=0)
            elif target_metric == 'recall':
                score = recall_score(y_val, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {target_metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.best_threshold = best_threshold

        logger.info(f"Best threshold: {best_threshold:.3f}")
        logger.info(f"Best {target_metric}: {best_score:.4f}")

        return best_threshold

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Predict fraud labels

        Args:
            X: Features
            threshold: Classification threshold

        Returns:
            Binary predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if threshold is None:
            threshold = self.best_threshold

        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        return y_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probabilities

        Args:
            X: Features

        Returns:
            Fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        logger.info("="*70)
        logger.info("Model Evaluation")
        logger.info("="*70)

        # Predictions
        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)

        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'threshold': self.best_threshold
        }

        # Log metrics
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f} (Target: > 0.85)")
        logger.info(f"Precision: {metrics['precision']:.4f} (Target: > 0.80)")
        logger.info(f"Recall:    {metrics['recall']:.4f} (Target: > 0.75)")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f} (Target: > 0.78)")
        logger.info(f"Threshold: {metrics['threshold']:.3f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
        logger.info(f"FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")

        # Classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop {top_n} Important Features:")
        for idx, row in feature_importance.head(top_n).iterrows():
            logger.info(f"{row['feature']:30s}: {row['importance']:.4f}")

        return feature_importance

    def save(self, path: str):
        """
        Save model to disk

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path.parent / f"{path.stem}_model.pkl"
        joblib.dump(self.model, model_path)

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'best_threshold': self.best_threshold,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'scale_pos_weight': self.scale_pos_weight
        }

        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Saved model to {model_path}")
        logger.info(f"✓ Saved metadata to {metadata_path}")

    def load(self, path: str):
        """
        Load model from disk

        Args:
            path: Path to saved model
        """
        path = Path(path)

        # Load model
        model_path = path.parent / f"{path.stem}_model.pkl"
        self.model = joblib.load(model_path)

        # Load metadata
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.feature_names = metadata['feature_names']
        self.best_threshold = metadata['best_threshold']

        logger.info(f"✓ Loaded model from {model_path}")


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys

    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from data.pipeline import DataPipeline

    # Load processed data
    data_dir = project_root / "data" / "processed"

    if not data_dir.exists():
        logger.error("Processed data not found. Run data pipeline first.")
        sys.exit(1)

    logger.info("Loading processed data...")
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    X_test = pd.read_parquet(data_dir / "X_test.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet")['Class']
    y_test = pd.read_parquet(data_dir / "y_test.parquet")['Class']

    # Split train into train/val
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Train model
    model = XGBoostFraudDetector(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    model.train(X_train_split, y_train_split, X_val, y_val)
    model.calibrate_threshold(X_val, y_val, target_metric='f1')

    # Evaluate
    metrics = model.evaluate(X_test, y_test)

    # Feature importance
    importance = model.get_feature_importance(top_n=20)

    # Save model
    model_path = project_root / "models" / "xgboost_fraud"
    model.save(str(model_path))

    logger.info("\n✓ XGBoost training completed!")
