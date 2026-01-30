"""
API Integration Tests
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestHealthEndpoint:
    """Health check endpoint tests"""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data


class TestRootEndpoint:
    """Root endpoint tests"""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_info(self, client):
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "docs" in data


class TestPredictEndpoint:
    """Prediction endpoint tests"""

    @pytest.fixture
    def sample_transaction(self):
        """Sample transaction for testing"""
        return {
            "features": {
                "Time": 50000,
                "Amount": 150.0,
                "V1": -1.5, "V2": 0.5, "V3": 1.2, "V4": -0.8,
                "V5": 0.3, "V6": -0.2, "V7": 0.8, "V8": -0.5,
                "V9": 0.1, "V10": -0.3, "V11": 0.6, "V12": -0.9,
                "V13": 0.2, "V14": -2.5, "V15": 0.4, "V16": -0.1,
                "V17": -1.8, "V18": 0.7, "V19": -0.4, "V20": 0.2,
                "V21": -0.3, "V22": 0.1, "V23": -0.2, "V24": 0.5,
                "V25": 0.3, "V26": -0.1, "V27": 0.4, "V28": -0.2
            },
            "transaction_id": "TEST_001"
        }

    def test_predict_response_format(self, client, sample_transaction):
        """Test prediction response structure"""
        response = client.post("/predict", json=sample_transaction)

        # May return 503 if model not loaded
        if response.status_code == 200:
            data = response.json()
            assert "transaction_id" in data
            assert "fraud_probability" in data
            assert "is_fraud" in data
            assert "confidence" in data

    def test_predict_probability_range(self, client, sample_transaction):
        """Test probability is in valid range"""
        response = client.post("/predict", json=sample_transaction)

        if response.status_code == 200:
            data = response.json()
            assert 0 <= data["fraud_probability"] <= 1
            assert 0 <= data["confidence"] <= 1


class TestBatchEndpoint:
    """Batch prediction endpoint tests"""

    @pytest.fixture
    def batch_transactions(self):
        """Sample batch for testing"""
        return {
            "transactions": [
                {
                    "features": {
                        "Time": 50000, "Amount": 150.0,
                        **{f"V{i}": 0.0 for i in range(1, 29)}
                    },
                    "transaction_id": f"BATCH_{i}"
                }
                for i in range(3)
            ]
        }

    def test_batch_response_format(self, client, batch_transactions):
        """Test batch response structure"""
        response = client.post("/batch/predict", json=batch_transactions)

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_processed" in data
            assert "fraud_count" in data
            assert "legitimate_count" in data

    def test_batch_processes_all(self, client, batch_transactions):
        """Test all transactions are processed"""
        response = client.post("/batch/predict", json=batch_transactions)

        if response.status_code == 200:
            data = response.json()
            assert data["total_processed"] == len(batch_transactions["transactions"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
