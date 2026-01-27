"""
SHAP-based Explainability for Fraud Detection Models
"""
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("shap_explainer")


class FraudExplainer:
    """SHAP-based explanation generator for fraud predictions"""

    def __init__(self, model, feature_names: List[str], model_type: str = "xgboost"):
        """
        Initialize SHAP explainer

        Args:
            model: Trained model (XGBoost or PyTorch)
            feature_names: List of feature names
            model_type: Type of model ('xgboost', 'pytorch', 'tree')
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.base_value = None

        logger.info(f"Initializing SHAP explainer for {model_type}")

    def fit_explainer(self, X_background: pd.DataFrame, max_samples: int = 100):
        """
        Fit SHAP explainer on background data

        Args:
            X_background: Background dataset for SHAP
            max_samples: Maximum samples to use for background
        """
        logger.info(f"Fitting SHAP explainer with {len(X_background)} background samples...")

        # Sample background data if too large
        if len(X_background) > max_samples:
            X_background = X_background.sample(n=max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} background samples")

        # Create appropriate explainer based on model type
        if self.model_type == "xgboost":
            # TreeExplainer is faster for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer for XGBoost")

        elif self.model_type == "pytorch":
            # DeepExplainer for neural networks
            import torch
            X_background_tensor = torch.FloatTensor(X_background.values)
            self.explainer = shap.DeepExplainer(self.model.model, X_background_tensor)
            logger.info("Using DeepExplainer for PyTorch")

        else:
            # KernelExplainer as fallback (model-agnostic but slower)
            def model_predict(X):
                if self.model_type == "xgboost":
                    return self.model.predict_proba(pd.DataFrame(X, columns=self.feature_names))
                else:
                    return self.model.predict_proba(pd.DataFrame(X, columns=self.feature_names))

            self.explainer = shap.KernelExplainer(model_predict, X_background)
            logger.info("Using KernelExplainer (model-agnostic)")

        logger.info("✓ SHAP explainer fitted successfully")

    def explain_prediction(
        self,
        X: pd.DataFrame,
        top_n: int = 10
    ) -> Dict:
        """
        Generate SHAP explanation for a single prediction

        Args:
            X: Single transaction features (1 row)
            top_n: Number of top features to return

        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer() first.")

        # Get SHAP values
        if self.model_type == "pytorch":
            import torch
            X_tensor = torch.FloatTensor(X.values)
            shap_values = self.explainer.shap_values(X_tensor)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = shap_values[0]  # Get first sample
        else:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                # For binary classification, take positive class
                shap_values = shap_values[1]
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

        # Get base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.0

        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values,
            'feature_value': X.iloc[0].values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=False)

        # Get top contributing features
        top_features = feature_importance.head(top_n)

        # Calculate prediction
        prediction_score = base_value + shap_values.sum()

        explanation = {
            'prediction_score': float(prediction_score),
            'base_value': float(base_value),
            'top_features': [],
            'all_features': feature_importance.to_dict('records')
        }

        # Format top features
        for _, row in top_features.iterrows():
            explanation['top_features'].append({
                'feature': row['feature'],
                'value': float(row['feature_value']),
                'shap_value': float(row['shap_value']),
                'impact': 'increases fraud risk' if row['shap_value'] > 0 else 'decreases fraud risk',
                'magnitude': float(row['abs_shap'])
            })

        return explanation

    def generate_text_explanation(
        self,
        explanation: Dict,
        threshold: float = 0.5
    ) -> str:
        """
        Generate human-readable text explanation

        Args:
            explanation: Explanation dictionary from explain_prediction()
            threshold: Classification threshold

        Returns:
            Human-readable explanation text
        """
        score = explanation['prediction_score']
        is_fraud = score >= threshold

        # Start explanation
        text = []

        # Fraud determination
        if is_fraud:
            text.append(f"🚨 FRAUD DETECTED (Score: {score:.3f})")
            text.append("\nThis transaction is flagged as FRAUDULENT.")
        else:
            text.append(f"✓ LEGITIMATE (Score: {score:.3f})")
            text.append("\nThis transaction appears legitimate.")

        text.append(f"\nBaseline fraud probability: {explanation['base_value']:.3f}")
        text.append(f"\n{'='*60}")

        # Top risk factors
        text.append("\n🔍 KEY RISK FACTORS:\n")

        for i, feature in enumerate(explanation['top_features'][:5], 1):
            direction = "⬆️ INCREASES" if feature['shap_value'] > 0 else "⬇️ DECREASES"
            text.append(
                f"{i}. {feature['feature']} = {feature['value']:.2f}\n"
                f"   {direction} fraud risk by {feature['magnitude']:.4f}\n"
            )

        return '\n'.join(text)

    def get_feature_importance_summary(
        self,
        X: pd.DataFrame,
        sample_size: int = 100
    ) -> pd.DataFrame:
        """
        Get overall feature importance across multiple samples

        Args:
            X: Dataset to analyze
            sample_size: Number of samples to use

        Returns:
            DataFrame with feature importance
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted.")

        logger.info(f"Computing feature importance for {min(sample_size, len(X))} samples...")

        # Sample if needed
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X

        # Get SHAP values
        if self.model_type == "pytorch":
            import torch
            X_tensor = torch.FloatTensor(X_sample.values)
            shap_values = self.explainer.shap_values(X_tensor)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        else:
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        logger.info("Feature importance computed")

        return importance

    def save_explanation(self, explanation: Dict, output_path: str):
        """
        Save explanation to JSON file

        Args:
            explanation: Explanation dictionary
            output_path: Path to save JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(explanation, f, indent=2)

        logger.info(f"✓ Saved explanation to {output_path}")


if __name__ == "__main__":
    # Example usage
    import joblib
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent

    # Load model
    logger.info("Loading XGBoost model...")
    model_path = project_root / "models" / "xgboost_fraud_model.pkl"
    model = joblib.load(model_path)

    # Load metadata
    metadata_path = project_root / "models" / "xgboost_fraud_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']

    # Load test data
    logger.info("Loading test data...")
    data_dir = project_root / "data" / "processed"
    X_test = pd.read_parquet(data_dir / "X_test.parquet")
    y_test = pd.read_parquet(data_dir / "y_test.parquet")['Class']

    # Initialize explainer
    explainer = FraudExplainer(model, feature_names, model_type="xgboost")

    # Fit explainer
    explainer.fit_explainer(X_test, max_samples=100)

    # Explain a fraud transaction
    fraud_idx = y_test[y_test == 1].index[0]
    X_fraud = X_test.loc[[fraud_idx]]

    logger.info(f"\nExplaining fraud transaction at index {fraud_idx}")
    explanation = explainer.explain_prediction(X_fraud, top_n=10)

    # Generate text explanation
    text_explanation = explainer.generate_text_explanation(explanation)
    print("\n" + text_explanation)

    # Save explanation
    output_path = project_root / "logs" / "sample_explanation.json"
    explainer.save_explanation(explanation, str(output_path))

    # Feature importance summary
    logger.info("\nComputing overall feature importance...")
    importance = explainer.get_feature_importance_summary(X_test, sample_size=100)

    logger.info("\nTop 10 Most Important Features:")
    for idx, row in importance.head(10).iterrows():
        logger.info(f"{row['feature']:30s}: {row['mean_abs_shap']:.4f}")

    logger.info("\n✓ SHAP explainability demo completed!")
