"""
Train all fraud detection models
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.classical.xgboost_model import XGBoostFraudDetector
from models.deep_learning.pytorch_model import PyTorchFraudDetector
from utils.logger import setup_logger

logger = setup_logger("train_all_models")


def train_all():
    """Train all models and compare performance"""

    logger.info("="*80)
    logger.info("TRAINING ALL FRAUD DETECTION MODELS")
    logger.info("="*80)

    # Load processed data
    data_dir = project_root / "data" / "processed"

    if not data_dir.exists():
        logger.error("Processed data not found!")
        logger.info("Run: python src/data/pipeline.py")
        return

    logger.info("\nLoading processed data...")
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    X_test = pd.read_parquet(data_dir / "X_test.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet")['Class']
    y_test = pd.read_parquet(data_dir / "y_test.parquet")['Class']

    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {len(X_train.columns)}")
    logger.info(f"Fraud rate (train): {y_train.mean()*100:.2f}%")
    logger.info(f"Fraud rate (test): {y_test.mean()*100:.2f}%")

    # Split train into train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    logger.info(f"\nTrain split: {len(X_train_split)}")
    logger.info(f"Validation: {len(X_val)}")

    results = {}

    # =========================================================================
    # MODEL 1: XGBoost
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODEL 1: XGBoost")
    logger.info("="*80)

    xgb_model = XGBoostFraudDetector(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    xgb_model.train(X_train_split, y_train_split, X_val, y_val)
    xgb_model.calibrate_threshold(X_val, y_val, target_metric='f1')
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_model.get_feature_importance(top_n=10)

    # Save model
    xgb_path = project_root / "models" / "xgboost_fraud"
    xgb_model.save(str(xgb_path))

    results['XGBoost'] = xgb_metrics

    # =========================================================================
    # MODEL 2: PyTorch Neural Network
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODEL 2: PyTorch Neural Network")
    logger.info("="*80)

    pytorch_model = PyTorchFraudDetector(
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=256,
        epochs=50
    )

    pytorch_model.train(X_train_split, y_train_split, X_val, y_val)
    pytorch_model.calibrate_threshold(X_val, y_val, target_metric='f1')
    pytorch_metrics = pytorch_model.evaluate(X_test, y_test)

    # Save model
    pytorch_path = project_root / "models" / "pytorch_fraud"
    pytorch_model.save(str(pytorch_path))

    results['PyTorch_NN'] = pytorch_metrics

    # =========================================================================
    # COMPARISON
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)

    comparison = pd.DataFrame(results).T
    logger.info("\n" + str(comparison))

    # Find best model
    best_model = comparison['roc_auc'].idxmax()
    logger.info(f"\nBest Model (by AUC): {best_model}")
    logger.info(f"AUC: {comparison.loc[best_model, 'roc_auc']:.4f}")

    # Save comparison
    comparison_path = project_root / "models" / "model_comparison.csv"
    comparison.to_csv(comparison_path)
    logger.info(f"\n✓ Saved comparison to {comparison_path}")

    # Check if targets are met
    logger.info("\n" + "="*80)
    logger.info("TARGET METRICS CHECK")
    logger.info("="*80)

    targets = {
        'roc_auc': 0.85,
        'precision': 0.80,
        'recall': 0.75,
        'f1_score': 0.78
    }

    for metric, target in targets.items():
        best_value = comparison[metric].max()
        status = "✓ PASS" if best_value >= target else "✗ FAIL"
        logger.info(f"{metric:12s}: {best_value:.4f} (target: {target:.2f}) {status}")

    logger.info("\n" + "="*80)
    logger.info("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    logger.info("="*80)

    return results


if __name__ == "__main__":
    train_all()
