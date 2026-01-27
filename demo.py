#!/usr/bin/env python
"""
Fraud Detection System - Interactive Demo
==========================================
Demonstrates the complete fraud detection pipeline with:
- ML Model prediction (XGBoost)
- SHAP-based explainability
- RAG-enhanced policy context
- LLM-powered explanations (optional)
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from explainability.fraud_explainer_pipeline import FraudExplanationPipeline
from utils.logger import setup_logger

logger = setup_logger("demo")


def print_header():
    """Print demo header"""
    print("\n" + "="*70)
    print("     AI-POWERED FRAUD DETECTION SYSTEM")
    print("     with Explainable AI & LLM Explanations")
    print("="*70 + "\n")


def load_sample_transactions():
    """Load sample transactions for demo"""
    project_root = Path(__file__).parent

    # Load test data
    X_test = pd.read_parquet(project_root / "data" / "processed" / "X_test.parquet")
    y_test = pd.read_parquet(project_root / "data" / "processed" / "y_test.parquet")["Class"]

    # Get some fraud and legitimate samples
    fraud_indices = y_test[y_test == 1].index[:5].tolist()
    legit_indices = y_test[y_test == 0].index[:5].tolist()

    samples = []
    for idx in fraud_indices + legit_indices:
        samples.append({
            'index': idx,
            'features': X_test.loc[idx],
            'actual_label': int(y_test.loc[idx])
        })

    return samples, X_test


def run_demo(use_llm: bool = False):
    """Run the interactive demo"""
    print_header()

    project_root = Path(__file__).parent

    # Initialize pipeline
    print("[1/4] Initializing Fraud Detection Pipeline...")
    print("-" * 50)

    pipeline = FraudExplanationPipeline(
        model_path=str(project_root / "models" / "xgboost_fraud"),
        model_type="xgboost",
        use_llm=use_llm,
        use_rag=True,
        llm_provider="huggingface"
    )

    # Load samples
    print("\n[2/4] Loading sample transactions...")
    print("-" * 50)

    samples, X_test = load_sample_transactions()
    print(f"Loaded {len(samples)} sample transactions (5 fraud, 5 legitimate)")

    # Fit SHAP explainer
    print("\n[3/4] Fitting SHAP explainer...")
    print("-" * 50)

    pipeline.fit_explainer(X_test.sample(n=200, random_state=42))
    print("SHAP explainer ready")

    # Analyze transactions
    print("\n[4/4] Analyzing transactions...")
    print("=" * 70)

    results = []

    for i, sample in enumerate(samples[:3], 1):  # Demo with 3 samples
        print(f"\n{'='*70}")
        print(f"TRANSACTION #{i} (Index: {sample['index']})")
        print(f"Actual Label: {'FRAUD' if sample['actual_label'] == 1 else 'LEGITIMATE'}")
        print("="*70)

        X = pd.DataFrame([sample['features']])
        X.index = [sample['index']]

        # Generate explanation
        explanation = pipeline.explain_transaction(
            X,
            include_llm=use_llm,
            include_rag=True
        )

        pred = explanation['prediction']

        # Print prediction
        print(f"\n{'PREDICTION':^40}")
        print("-" * 40)
        print(f"  Fraud Probability: {pred['fraud_probability']:.4f}")
        print(f"  Classification:    {'FRAUD' if pred['is_fraud'] else 'LEGITIMATE'}")
        print(f"  Confidence:        {pred['confidence']:.2%}")
        print(f"  Threshold:         {pred['threshold']:.4f}")

        # Correct prediction?
        correct = (pred['is_fraud'] == (sample['actual_label'] == 1))
        print(f"\n  Prediction: {'CORRECT' if correct else 'INCORRECT'}")

        # Top risk factors
        print(f"\n{'TOP RISK FACTORS':^40}")
        print("-" * 40)

        for j, factor in enumerate(explanation['shap_analysis']['top_features'][:5], 1):
            sign = "+" if factor['shap_value'] > 0 else ""
            impact = "increases" if factor['shap_value'] > 0 else "decreases"
            print(f"  {j}. {factor['feature']:20s} = {factor['value']:8.3f}")
            print(f"     SHAP: {sign}{factor['shap_value']:.4f} ({impact} fraud risk)")

        # Policy context
        if explanation.get('policy_context'):
            print(f"\n{'RELEVANT POLICIES':^40}")
            print("-" * 40)
            context = explanation['policy_context'][:300]
            print(f"  {context}...")

        # LLM explanation
        if explanation.get('llm_explanation'):
            print(f"\n{'AI ANALYST EXPLANATION':^40}")
            print("-" * 40)
            print(f"  {explanation['llm_explanation'][:400]}...")

        results.append({
            'transaction_id': sample['index'],
            'actual': sample['actual_label'],
            'predicted': 1 if pred['is_fraud'] else 0,
            'probability': pred['fraud_probability'],
            'correct': correct
        })

    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)

    correct_count = sum(1 for r in results if r['correct'])
    print(f"\nTransactions analyzed: {len(results)}")
    print(f"Correct predictions:   {correct_count}/{len(results)}")
    print(f"Accuracy:              {correct_count/len(results):.1%}")

    print("\nResults:")
    for r in results:
        status = "CORRECT" if r['correct'] else "MISS"
        actual = "FRAUD" if r['actual'] == 1 else "LEGIT"
        pred = "FRAUD" if r['predicted'] == 1 else "LEGIT"
        print(f"  Transaction {r['transaction_id']}: {actual} -> {pred} (prob: {r['probability']:.3f}) [{status}]")

    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection Demo")
    parser.add_argument("--llm", action="store_true", help="Enable LLM explanations")
    parser.add_argument("--api", action="store_true", help="Start API server")

    args = parser.parse_args()

    if args.api:
        # Start FastAPI server
        import uvicorn
        print("Starting FastAPI server...")
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )
    else:
        # Run interactive demo
        run_demo(use_llm=args.llm)


if __name__ == "__main__":
    main()
