"""
End-to-End Fraud Explanation Pipeline
Combines: Model Prediction + SHAP + RAG + LLM
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from explainability.shap_explainer import FraudExplainer
from llm.llm_service import FraudLLMExplainer
from rag.rag_system import FraudPolicyRAG
from utils.logger import setup_logger

logger = setup_logger("fraud_explainer_pipeline")


class FraudExplanationPipeline:
    """Complete fraud explanation pipeline"""

    def __init__(
        self,
        model_path: str,
        model_type: str = "xgboost",
        use_llm: bool = True,
        use_rag: bool = True,
        llm_provider: str = "huggingface"
    ):
        """
        Initialize explanation pipeline

        Args:
            model_path: Path to trained model
            model_type: Type of model ('xgboost', 'pytorch')
            use_llm: Whether to use LLM for explanations
            use_rag: Whether to use RAG for policy context
            llm_provider: LLM provider ('huggingface', 'openai', 'anthropic')
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.use_llm = use_llm
        self.use_rag = use_rag

        logger.info("="*70)
        logger.info("Initializing Fraud Explanation Pipeline")
        logger.info("="*70)

        # Load model
        self._load_model()

        # Initialize SHAP explainer
        self.shap_explainer = None

        # Initialize LLM (optional)
        self.llm_explainer = None
        if use_llm:
            try:
                self.llm_explainer = FraudLLMExplainer(provider=llm_provider)
                logger.info(f"✓ LLM explainer initialized ({llm_provider})")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}")
                logger.info("Will use fallback explanations")
                self.use_llm = False

        # Initialize RAG (optional)
        self.rag_system = None
        if use_rag:
            try:
                self.rag_system = FraudPolicyRAG()
                logger.info("✓ RAG system initialized")
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}")
                logger.info("Will proceed without policy context")
                self.use_rag = False

        logger.info("="*70)

    def _load_model(self):
        """Load trained model and metadata"""
        logger.info(f"Loading {self.model_type} model...")

        # Load model
        if self.model_type == "xgboost":
            model_file = self.model_path.parent / f"{self.model_path.stem}_model.pkl"
            self.model = joblib.load(model_file)

        elif self.model_type == "pytorch":
            from models.deep_learning.pytorch_model import PyTorchFraudDetector
            self.model = PyTorchFraudDetector()
            self.model.load(str(self.model_path))

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load metadata
        metadata_file = self.model_path.parent / f"{self.model_path.stem}_metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata['feature_names']
        self.threshold = self.metadata.get('best_threshold', 0.5)

        logger.info(f"✓ Model loaded: {model_file.name}")
        logger.info(f"✓ Features: {len(self.feature_names)}")
        logger.info(f"✓ Threshold: {self.threshold}")

    def fit_explainer(self, X_background: pd.DataFrame):
        """
        Fit SHAP explainer on background data

        Args:
            X_background: Background dataset
        """
        logger.info("Fitting SHAP explainer...")

        self.shap_explainer = FraudExplainer(
            self.model,
            self.feature_names,
            self.model_type
        )

        self.shap_explainer.fit_explainer(X_background, max_samples=100)

        logger.info("✓ SHAP explainer fitted")

    def explain_transaction(
        self,
        X: pd.DataFrame,
        include_llm: bool = True,
        include_rag: bool = True
    ) -> Dict:
        """
        Generate complete explanation for a transaction

        Args:
            X: Single transaction features (1 row DataFrame)
            include_llm: Whether to include LLM explanation
            include_rag: Whether to include RAG context

        Returns:
            Complete explanation dictionary
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not fitted. Call fit_explainer() first.")

        logger.info("="*70)
        logger.info("Generating Fraud Explanation")
        logger.info("="*70)

        # Step 1: Get prediction
        # Ensure feature order matches training
        X = X[self.feature_names]

        if self.model_type == "xgboost":
            fraud_prob = self.model.predict_proba(X)[0, 1]
        else:
            fraud_prob = self.model.predict_proba(X)[0]

        is_fraud = fraud_prob >= self.threshold

        logger.info(f"Fraud Probability: {fraud_prob:.4f}")
        logger.info(f"Classification: {'FRAUD' if is_fraud else 'LEGITIMATE'}")

        # Step 2: Get SHAP explanation
        logger.info("Generating SHAP explanation...")
        shap_explanation = self.shap_explainer.explain_prediction(X, top_n=10)

        # Step 3: Get RAG context (if enabled)
        rag_context = None
        if include_rag and self.use_rag and self.rag_system:
            logger.info("Retrieving policy context...")

            # Build query from top features
            top_features = shap_explanation['top_features'][:3]
            query = " ".join([f"{f['feature']}" for f in top_features])

            context_chunks = self.rag_system.retrieve_context(query, top_k=2)
            rag_context = self.rag_system.format_context_for_llm(context_chunks)

            logger.info(f"✓ Retrieved {len(context_chunks)} policy chunks")

        # Step 4: Generate LLM explanation (if enabled)
        llm_explanation = None
        if include_llm and self.use_llm and self.llm_explainer:
            logger.info("Generating LLM explanation...")

            transaction_data = {
                'Amount': X.iloc[0].get('Amount', 0),
                'Time': X.iloc[0].get('Time', 0)
            }

            try:
                llm_explanation = self.llm_explainer.generate_explanation(
                    shap_explanation=shap_explanation,
                    rag_context=rag_context,
                    transaction_data=transaction_data,
                    max_tokens=300,
                    temperature=0.3
                )
                logger.info("✓ LLM explanation generated")
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
                llm_explanation = self.llm_explainer._fallback_explanation(shap_explanation)
                logger.info("✓ Using fallback explanation")

        # Step 5: Compile complete explanation
        explanation = {
            'transaction_id': X.index[0] if hasattr(X.index[0], '__int__') else str(X.index[0]),
            'prediction': {
                'fraud_probability': float(fraud_prob),
                'is_fraud': bool(is_fraud),
                'threshold': float(self.threshold),
                'confidence': abs(fraud_prob - 0.5) * 2  # 0-1 confidence
            },
            'shap_analysis': shap_explanation,
            'llm_explanation': llm_explanation,
            'policy_context': rag_context,
            'model_info': {
                'model_type': self.model_type,
                'num_features': len(self.feature_names)
            }
        }

        logger.info("✓ Complete explanation generated")
        logger.info("="*70)

        return explanation

    def format_explanation_report(self, explanation: Dict) -> str:
        """
        Format explanation as human-readable report

        Args:
            explanation: Explanation dictionary

        Returns:
            Formatted report string
        """
        pred = explanation['prediction']
        shap = explanation['shap_analysis']

        report = []
        report.append("="*70)
        report.append("FRAUD DETECTION EXPLANATION REPORT")
        report.append("="*70)

        # Prediction
        status = "🚨 FRAUDULENT" if pred['is_fraud'] else "✓ LEGITIMATE"
        report.append(f"\nSTATUS: {status}")
        report.append(f"Fraud Probability: {pred['fraud_probability']:.4f}")
        report.append(f"Confidence: {pred['confidence']:.2%}")
        report.append(f"Threshold: {pred['threshold']:.2f}")

        # Top Risk Factors
        report.append(f"\n{'='*70}")
        report.append("TOP RISK FACTORS (SHAP Analysis):")
        report.append("="*70)

        for i, feature in enumerate(shap['top_features'][:5], 1):
            impact = "⬆️" if feature['shap_value'] > 0 else "⬇️"
            report.append(
                f"\n{i}. {feature['feature']} = {feature['value']:.3f}\n"
                f"   {impact} Impact: {feature['shap_value']:+.4f}\n"
                f"   {feature['impact'].title()}"
            )

        # LLM Explanation
        if explanation['llm_explanation']:
            report.append(f"\n{'='*70}")
            report.append("AI ANALYST EXPLANATION:")
            report.append("="*70)
            report.append(f"\n{explanation['llm_explanation']}")

        # Policy Context
        if explanation['policy_context']:
            report.append(f"\n{'='*70}")
            report.append("RELEVANT POLICIES:")
            report.append("="*70)
            report.append(f"\n{explanation['policy_context']}")

        report.append(f"\n{'='*70}")
        report.append(f"Model: {explanation['model_info']['model_type'].upper()}")
        report.append("="*70)

        return '\n'.join(report)

    def save_explanation(self, explanation: Dict, output_path: str):
        """Save explanation to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(explanation, f, indent=2)

        logger.info(f"✓ Saved explanation to {output_path}")


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent.parent

    # Initialize pipeline
    pipeline = FraudExplanationPipeline(
        model_path=str(project_root / "models" / "xgboost_fraud"),
        model_type="xgboost",
        use_llm=True,
        use_rag=True,
        llm_provider="huggingface"
    )

    # Load test data
    data_dir = project_root / "data" / "processed"
    X_test = pd.read_parquet(data_dir / "X_test.parquet")
    y_test = pd.read_parquet(data_dir / "y_test.parquet")['Class']

    # Fit explainer
    pipeline.fit_explainer(X_test)

    # Explain a fraud transaction
    fraud_indices = y_test[y_test == 1].index
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        X_fraud = X_test.loc[[fraud_idx]]

        logger.info(f"\nExplaining fraud transaction at index {fraud_idx}")

        # Generate explanation
        explanation = pipeline.explain_transaction(
            X_fraud,
            include_llm=True,
            include_rag=True
        )

        # Print report
        report = pipeline.format_explanation_report(explanation)
        print("\n" + report + "\n")

        # Save explanation
        output_path = project_root / "logs" / f"fraud_explanation_{fraud_idx}.json"
        pipeline.save_explanation(explanation, str(output_path))

    logger.info("\n✓ Fraud explanation pipeline demo completed!")
