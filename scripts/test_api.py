"""
Test script for Fraud Detection API
"""
import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("TEST: Model Info")
    print("="*60)

    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_single_prediction():
    """Test single transaction prediction"""
    print("\n" + "="*60)
    print("TEST: Single Prediction")
    print("="*60)

    # Sample legitimate transaction
    transaction = {
        "transaction_id": "TEST_LEGIT_001",
        "features": {
            "Time": 13547.0,
            "Amount": 45.50,
            "V1": -1.359807,
            "V2": -0.072781,
            "V3": 2.536347,
            "V4": 1.378155,
            "V5": -0.338321,
            "V6": 0.462388,
            "V7": 0.239599,
            "V8": 0.098698,
            "V9": 0.363787,
            "V10": 0.090794,
            "V11": -0.551600,
            "V12": -0.617801,
            "V13": -0.991390,
            "V14": -0.311169,
            "V15": 1.468177,
            "V16": -0.470401,
            "V17": 0.207971,
            "V18": 0.025791,
            "V19": 0.403993,
            "V20": 0.251412,
            "V21": -0.018307,
            "V22": 0.277838,
            "V23": -0.110474,
            "V24": 0.066928,
            "V25": 0.128539,
            "V26": -0.189115,
            "V27": 0.133558,
            "V28": -0.021053
        }
    }

    start_time = time.time()
    response = requests.post(f"{BASE_URL}/predict", json=transaction)
    latency = (time.time() - start_time) * 1000

    print(f"Status: {response.status_code}")
    print(f"Latency: {latency:.2f}ms")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_fraud_prediction():
    """Test fraud transaction prediction"""
    print("\n" + "="*60)
    print("TEST: Fraud Prediction")
    print("="*60)

    # Sample fraud transaction (abnormal values)
    transaction = {
        "transaction_id": "TEST_FRAUD_001",
        "features": {
            "Time": 3547.0,
            "Amount": 1234.56,
            "V1": -5.359807,
            "V2": 3.072781,
            "V3": -2.536347,
            "V4": 4.378155,
            "V5": -2.338321,
            "V6": -1.462388,
            "V7": -3.239599,
            "V8": 0.098698,
            "V9": -2.363787,
            "V10": -4.090794,
            "V11": 3.551600,
            "V12": -5.617801,
            "V13": 2.991390,
            "V14": -19.311169,  # Abnormal V14
            "V15": -1.468177,
            "V16": 5.470401,
            "V17": -8.207971,
            "V18": -2.025791,
            "V19": 1.403993,
            "V20": -0.251412,
            "V21": 0.718307,
            "V22": -0.277838,
            "V23": 0.110474,
            "V24": -0.866928,
            "V25": 0.328539,
            "V26": 0.189115,
            "V27": -0.533558,
            "V28": 0.421053
        }
    }

    start_time = time.time()
    response = requests.post(f"{BASE_URL}/predict", json=transaction)
    latency = (time.time() - start_time) * 1000

    print(f"Status: {response.status_code}")
    print(f"Latency: {latency:.2f}ms")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_explanation():
    """Test fraud explanation endpoint"""
    print("\n" + "="*60)
    print("TEST: Fraud Explanation")
    print("="*60)

    transaction = {
        "transaction_id": "TEST_EXPLAIN_001",
        "features": {
            "Time": 3547.0,
            "Amount": 1234.56,
            "V1": -5.359807,
            "V2": 3.072781,
            "V3": -2.536347,
            "V4": 4.378155,
            "V14": -19.311169,
            "V17": -8.207971
        }
    }

    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/explain",
        json=transaction,
        params={"include_llm": False, "include_rag": False}
    )
    latency = (time.time() - start_time) * 1000

    print(f"Status: {response.status_code}")
    print(f"Latency: {latency:.2f}ms")

    if response.status_code == 200:
        data = response.json()
        print(f"\nPrediction: {data['prediction']['is_fraud']}")
        print(f"Fraud Probability: {data['prediction']['fraud_probability']:.4f}")
        print("\nTop Risk Factors:")
        for i, factor in enumerate(data['top_risk_factors'][:5], 1):
            print(f"  {i}. {factor['feature']} = {factor['value']:.3f} ({factor['impact']})")
    else:
        print(f"Response: {response.text}")

    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("TEST: Batch Prediction")
    print("="*60)

    transactions = {
        "transactions": [
            {
                "transaction_id": f"BATCH_{i}",
                "features": {
                    "Time": 1000 + i * 100,
                    "Amount": 50 + i * 10,
                    "V1": -1.0 + i * 0.1,
                    "V14": -5 if i < 3 else -15
                }
            }
            for i in range(5)
        ]
    }

    start_time = time.time()
    response = requests.post(f"{BASE_URL}/batch/predict", json=transactions)
    latency = (time.time() - start_time) * 1000

    print(f"Status: {response.status_code}")
    print(f"Total Latency: {latency:.2f}ms")

    if response.status_code == 200:
        data = response.json()
        print(f"Processed: {data['total_processed']}")
        print(f"Fraud: {data['fraud_count']}")
        print(f"Legitimate: {data['legitimate_count']}")
        print(f"Per-transaction latency: {data['processing_time_ms'] / data['total_processed']:.2f}ms")
    else:
        print(f"Response: {response.text}")

    return response.status_code == 200


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*70)
    print("FRAUD DETECTION API TEST SUITE")
    print("="*70)
    print(f"Base URL: {BASE_URL}")

    results = {
        "Health Check": test_health(),
        "Model Info": test_model_info(),
        "Single Prediction": test_single_prediction(),
        "Fraud Prediction": test_fraud_prediction(),
        "Explanation": test_explanation(),
        "Batch Prediction": test_batch_prediction()
    }

    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:25s}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("="*70)
    print(f"Total: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    import sys

    print("Make sure the API is running: python scripts/run_api.py")
    print()

    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running: python scripts/run_api.py")
        sys.exit(1)
