"""
Unit tests for ML models
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestXGBoostModel:
    """Tests for XGBoost fraud detection model"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample test data"""
        np.random.seed(42)
        n_samples = 100

        data = {
            'Time': np.random.uniform(0, 172800, n_samples),
            'Amount': np.random.exponential(100, n_samples),
        }

        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)

        X = pd.DataFrame(data)
        y = pd.Series(np.random.binomial(1, 0.1, n_samples))

        return X, y

    def test_model_loading(self):
        """Test model can be loaded"""
        import joblib
        model_path = Path(__file__).parent.parent / "models" / "xgboost_fraud_model.pkl"

        if model_path.exists():
            model = joblib.load(model_path)
            assert model is not None
            assert hasattr(model, 'predict_proba')

    def test_prediction_shape(self, sample_data):
        """Test prediction output shape"""
        import joblib
        model_path = Path(__file__).parent.parent / "models" / "xgboost_fraud_model.pkl"

        if not model_path.exists():
            pytest.skip("Model file not found")

        model = joblib.load(model_path)
        X, _ = sample_data

        # Add engineered features
        X['Time_hours'] = X['Time'] / 3600
        X['Time_hour_of_day'] = (X['Time'] / 3600) % 24
        X['Time_period_numeric'] = pd.cut(X['Time_hour_of_day'], bins=[0,6,12,18,24], labels=[0,1,2,3]).astype(int)
        X['Amount_log'] = np.log1p(X['Amount'])
        X['Amount_category_numeric'] = pd.cut(X['Amount'], bins=[0,50,200,1000,float('inf')], labels=[0,1,2,3]).astype(int)
        X['Is_high_value'] = (X['Amount'] > 1000).astype(int)
        X['V1_Amount_interaction'] = X['V1'] * X['Amount_log']
        X['V2_Amount_interaction'] = X['V2'] * X['Amount_log']

        v_cols = [f'V{i}' for i in range(1, 29)]
        X['V_mean'] = X[v_cols].mean(axis=1)
        X['V_std'] = X[v_cols].std(axis=1)
        X['V_max'] = X[v_cols].max(axis=1)
        X['V_min'] = X[v_cols].min(axis=1)
        X['V_range'] = X['V_max'] - X['V_min']

        predictions = model.predict_proba(X)

        assert predictions.shape[0] == len(X)
        assert predictions.shape[1] == 2

    def test_prediction_range(self, sample_data):
        """Test predictions are valid probabilities"""
        import joblib
        model_path = Path(__file__).parent.parent / "models" / "xgboost_fraud_model.pkl"

        if not model_path.exists():
            pytest.skip("Model file not found")

        model = joblib.load(model_path)
        X, _ = sample_data

        # Add engineered features (same as above)
        X['Time_hours'] = X['Time'] / 3600
        X['Time_hour_of_day'] = (X['Time'] / 3600) % 24
        X['Time_period_numeric'] = pd.cut(X['Time_hour_of_day'], bins=[0,6,12,18,24], labels=[0,1,2,3]).astype(int)
        X['Amount_log'] = np.log1p(X['Amount'])
        X['Amount_category_numeric'] = pd.cut(X['Amount'], bins=[0,50,200,1000,float('inf')], labels=[0,1,2,3]).astype(int)
        X['Is_high_value'] = (X['Amount'] > 1000).astype(int)
        X['V1_Amount_interaction'] = X['V1'] * X['Amount_log']
        X['V2_Amount_interaction'] = X['V2'] * X['Amount_log']

        v_cols = [f'V{i}' for i in range(1, 29)]
        X['V_mean'] = X[v_cols].mean(axis=1)
        X['V_std'] = X[v_cols].std(axis=1)
        X['V_max'] = X[v_cols].max(axis=1)
        X['V_min'] = X[v_cols].min(axis=1)
        X['V_range'] = X['V_max'] - X['V_min']

        predictions = model.predict_proba(X)[:, 1]

        assert all(0 <= p <= 1 for p in predictions)


class TestRAGSystem:
    """Tests for RAG system"""

    def test_rag_initialization(self):
        """Test RAG system can be initialized"""
        from rag.rag_system import FraudPolicyRAG

        rag = FraudPolicyRAG(collection_name="test_collection")
        assert rag is not None

    def test_policy_loading(self):
        """Test policies can be loaded"""
        from rag.rag_system import FraudPolicyRAG

        rag = FraudPolicyRAG(collection_name="test_policies")
        project_root = Path(__file__).parent.parent
        rag.load_policy_documents(str(project_root / "data" / "policies"))

        assert rag.count() > 0

    def test_context_retrieval(self):
        """Test context retrieval"""
        from rag.rag_system import FraudPolicyRAG

        rag = FraudPolicyRAG(collection_name="test_retrieval")
        project_root = Path(__file__).parent.parent
        rag.load_policy_documents(str(project_root / "data" / "policies"))

        context = rag.retrieve_context("V14 high risk transaction", top_k=2)

        assert len(context) > 0
        assert 'text' in context[0]
        assert 'source' in context[0]


class TestAPI:
    """Tests for FastAPI endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestMonitoring:
    """Tests for monitoring module"""

    def test_metrics_collector(self):
        """Test metrics collection"""
        from monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_prediction(
            transaction_id="test_123",
            fraud_probability=0.75,
            is_fraud=True,
            response_time_ms=50.0
        )

        metrics = collector.get_metrics()

        assert metrics['total_predictions'] == 1
        assert metrics['fraud_detected'] == 1

    def test_drift_detector(self):
        """Test drift detection"""
        from monitoring.drift_detector import DriftDetector

        np.random.seed(42)
        reference_data = pd.DataFrame({
            'V1': np.random.normal(0, 1, 500),
            'V2': np.random.normal(0, 1, 500)
        })

        detector = DriftDetector(reference_data=reference_data)

        # Add observations
        for _ in range(200):
            detector.add_observation(
                {'V1': np.random.normal(0, 1), 'V2': np.random.normal(0, 1)},
                prediction=np.random.random()
            )

        result = detector.detect_data_drift()
        assert 'drift_detected' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
