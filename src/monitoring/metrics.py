"""
Monitoring & Metrics for Fraud Detection System
"""
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import statistics
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("monitoring")


@dataclass
class PredictionMetric:
    """Single prediction metric"""
    timestamp: str
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    response_time_ms: float
    model_type: str


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    total_predictions: int = 0
    fraud_detected: int = 0
    legitimate_count: int = 0
    avg_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    uptime_seconds: float = 0.0
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Rolling window metrics
    predictions_last_minute: int = 0
    predictions_last_hour: int = 0
    fraud_rate_last_hour: float = 0.0


class MetricsCollector:
    """Collects and aggregates system metrics"""

    def __init__(self, history_size: int = 10000):
        """
        Initialize metrics collector

        Args:
            history_size: Maximum number of predictions to keep in history
        """
        self.history_size = history_size
        self.predictions: deque = deque(maxlen=history_size)
        self.response_times: deque = deque(maxlen=history_size)

        self.system_metrics = SystemMetrics()
        self.start_time = datetime.utcnow()

        logger.info(f"Metrics collector initialized (history: {history_size})")

    def record_prediction(
        self,
        transaction_id: str,
        fraud_probability: float,
        is_fraud: bool,
        response_time_ms: float,
        model_type: str = "xgboost"
    ):
        """Record a prediction"""
        metric = PredictionMetric(
            timestamp=datetime.utcnow().isoformat(),
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            response_time_ms=response_time_ms,
            model_type=model_type
        )

        self.predictions.append(metric)
        self.response_times.append(response_time_ms)

        # Update system metrics
        self.system_metrics.total_predictions += 1

        if is_fraud:
            self.system_metrics.fraud_detected += 1
        else:
            self.system_metrics.legitimate_count += 1

        # Update response time stats
        self._update_response_time_stats(response_time_ms)

    def _update_response_time_stats(self, response_time_ms: float):
        """Update response time statistics"""
        times = list(self.response_times)
        if times:
            self.system_metrics.avg_response_time_ms = statistics.mean(times)
            self.system_metrics.max_response_time_ms = max(times)
            self.system_metrics.min_response_time_ms = min(times)

    def get_metrics(self) -> Dict:
        """Get current system metrics"""
        # Calculate uptime
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        self.system_metrics.uptime_seconds = uptime

        # Calculate rolling window metrics
        now = datetime.utcnow()
        minute_ago = now.timestamp() - 60
        hour_ago = now.timestamp() - 3600

        predictions_minute = 0
        predictions_hour = 0
        fraud_hour = 0

        for pred in self.predictions:
            pred_time = datetime.fromisoformat(pred.timestamp).timestamp()
            if pred_time > minute_ago:
                predictions_minute += 1
            if pred_time > hour_ago:
                predictions_hour += 1
                if pred.is_fraud:
                    fraud_hour += 1

        self.system_metrics.predictions_last_minute = predictions_minute
        self.system_metrics.predictions_last_hour = predictions_hour
        self.system_metrics.fraud_rate_last_hour = (
            fraud_hour / predictions_hour if predictions_hour > 0 else 0.0
        )

        return {
            'total_predictions': self.system_metrics.total_predictions,
            'fraud_detected': self.system_metrics.fraud_detected,
            'legitimate_count': self.system_metrics.legitimate_count,
            'fraud_rate': (
                self.system_metrics.fraud_detected / self.system_metrics.total_predictions
                if self.system_metrics.total_predictions > 0 else 0.0
            ),
            'response_time': {
                'avg_ms': round(self.system_metrics.avg_response_time_ms, 2),
                'max_ms': round(self.system_metrics.max_response_time_ms, 2),
                'min_ms': round(self.system_metrics.min_response_time_ms, 2)
            },
            'rolling_window': {
                'predictions_last_minute': predictions_minute,
                'predictions_last_hour': predictions_hour,
                'fraud_rate_last_hour': round(self.system_metrics.fraud_rate_last_hour, 4)
            },
            'uptime_seconds': round(uptime, 2),
            'start_time': self.system_metrics.start_time
        }

    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """Get recent predictions"""
        recent = list(self.predictions)[-limit:]
        return [
            {
                'timestamp': p.timestamp,
                'transaction_id': p.transaction_id,
                'fraud_probability': round(p.fraud_probability, 4),
                'is_fraud': p.is_fraud,
                'response_time_ms': round(p.response_time_ms, 2)
            }
            for p in reversed(recent)
        ]

    def export_metrics(self, output_path: str):
        """Export metrics to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'exported_at': datetime.utcnow().isoformat(),
            'system_metrics': self.get_metrics(),
            'recent_predictions': self.get_recent_predictions(100)
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics exported to {output_path}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics() -> Dict:
    """Get current metrics"""
    return metrics_collector.get_metrics()


def record_prediction(**kwargs):
    """Record a prediction"""
    metrics_collector.record_prediction(**kwargs)


if __name__ == "__main__":
    # Demo
    import random

    logger.info("="*70)
    logger.info("Metrics Collector Demo")
    logger.info("="*70)

    # Simulate predictions
    for i in range(100):
        fraud_prob = random.random()
        is_fraud = fraud_prob > 0.5
        response_time = random.uniform(10, 100)

        record_prediction(
            transaction_id=f"TXN_{i}",
            fraud_probability=fraud_prob,
            is_fraud=is_fraud,
            response_time_ms=response_time
        )

    # Get metrics
    metrics = get_metrics()
    print("\nSystem Metrics:")
    print(json.dumps(metrics, indent=2))

    # Get recent predictions
    recent = metrics_collector.get_recent_predictions(5)
    print("\nRecent Predictions:")
    for p in recent:
        print(f"  {p['transaction_id']}: {p['fraud_probability']:.3f} ({'FRAUD' if p['is_fraud'] else 'OK'})")

    logger.info("\n✓ Metrics collector demo completed!")
