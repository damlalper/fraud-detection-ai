"""
Model Drift Detection & Performance Monitoring
Detects data drift and model performance degradation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from scipy import stats
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("drift_detector")


class DriftDetector:
    """Detects data drift and model performance degradation"""

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        window_size: int = 1000,
        drift_threshold: float = 0.05
    ):
        """
        Initialize drift detector

        Args:
            reference_data: Baseline data for comparison
            window_size: Size of sliding window for drift detection
            drift_threshold: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.window_size = window_size
        self.drift_threshold = drift_threshold

        # Sliding windows for predictions
        self.prediction_window: deque = deque(maxlen=window_size)
        self.feature_windows: Dict[str, deque] = {}

        # Reference statistics
        self.reference_stats: Dict[str, Dict] = {}

        if reference_data is not None:
            self._compute_reference_stats(reference_data)

        logger.info(f"Drift detector initialized (window: {window_size})")

    def _compute_reference_stats(self, data: pd.DataFrame):
        """Compute reference statistics for each feature"""
        for column in data.columns:
            if data[column].dtype in [np.float64, np.float32, np.int64, np.int32]:
                self.reference_stats[column] = {
                    "mean": float(data[column].mean()),
                    "std": float(data[column].std()),
                    "min": float(data[column].min()),
                    "max": float(data[column].max()),
                    "median": float(data[column].median()),
                    "q25": float(data[column].quantile(0.25)),
                    "q75": float(data[column].quantile(0.75))
                }
                self.feature_windows[column] = deque(maxlen=self.window_size)

        logger.info(f"Computed reference stats for {len(self.reference_stats)} features")

    def set_reference_data(self, data: pd.DataFrame):
        """Set new reference data"""
        self.reference_data = data
        self._compute_reference_stats(data)

    def add_observation(self, features: Dict[str, float], prediction: float):
        """Add a new observation to the sliding window"""
        self.prediction_window.append(prediction)

        for feature, value in features.items():
            if feature in self.feature_windows:
                self.feature_windows[feature].append(value)

    def detect_data_drift(self) -> Dict[str, Dict]:
        """
        Detect data drift using Kolmogorov-Smirnov test

        Returns:
            Dictionary with drift status for each feature
        """
        if self.reference_data is None:
            return {"error": "No reference data set"}

        drift_results = {}

        for feature, window in self.feature_windows.items():
            if len(window) < 100:  # Need minimum samples
                continue

            current_data = np.array(window)
            reference_data = self.reference_data[feature].values

            # KS test
            ks_stat, p_value = stats.ks_2samp(reference_data, current_data)

            # Check for drift
            is_drift = p_value < self.drift_threshold

            drift_results[feature] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "is_drift": is_drift,
                "current_mean": float(np.mean(current_data)),
                "reference_mean": self.reference_stats[feature]["mean"],
                "mean_shift": float(np.mean(current_data) - self.reference_stats[feature]["mean"])
            }

        # Summary
        drifted_features = [f for f, r in drift_results.items() if r.get("is_drift", False)]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_features": len(drift_results),
            "drifted_features": drifted_features,
            "drift_detected": len(drifted_features) > 0,
            "details": drift_results
        }

    def detect_prediction_drift(self) -> Dict:
        """Detect drift in model predictions"""
        if len(self.prediction_window) < 100:
            return {"error": "Not enough predictions for drift detection"}

        predictions = np.array(self.prediction_window)

        # Calculate statistics
        current_stats = {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "fraud_rate": float(np.mean(predictions > 0.5))
        }

        # Check for significant changes (using simple threshold)
        # In production, compare against baseline
        alerts = []

        if current_stats["fraud_rate"] > 0.1:  # Alert if >10% fraud rate
            alerts.append({
                "type": "high_fraud_rate",
                "message": f"Fraud rate ({current_stats['fraud_rate']:.2%}) exceeds threshold",
                "severity": "warning"
            })

        if current_stats["fraud_rate"] < 0.001:  # Alert if <0.1% fraud rate
            alerts.append({
                "type": "low_fraud_rate",
                "message": f"Unusually low fraud rate ({current_stats['fraud_rate']:.2%})",
                "severity": "info"
            })

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_stats": current_stats,
            "sample_size": len(predictions),
            "alerts": alerts
        }

    def get_feature_statistics(self) -> Dict[str, Dict]:
        """Get current vs reference statistics for all features"""
        stats_comparison = {}

        for feature, window in self.feature_windows.items():
            if len(window) < 10:
                continue

            current_data = np.array(window)

            stats_comparison[feature] = {
                "current": {
                    "mean": float(np.mean(current_data)),
                    "std": float(np.std(current_data)),
                    "min": float(np.min(current_data)),
                    "max": float(np.max(current_data))
                },
                "reference": self.reference_stats.get(feature, {}),
                "sample_size": len(current_data)
            }

        return stats_comparison


class PerformanceMonitor:
    """Monitor model performance over time"""

    def __init__(self, alert_threshold: float = 0.1):
        """
        Initialize performance monitor

        Args:
            alert_threshold: Performance drop threshold for alerts
        """
        self.alert_threshold = alert_threshold
        self.performance_history: List[Dict] = []
        self.baseline_metrics: Optional[Dict] = None

        logger.info("Performance monitor initialized")

    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline performance metrics"""
        self.baseline_metrics = metrics
        logger.info(f"Baseline set: {metrics}")

    def record_performance(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """Record performance metrics"""
        record = {
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "metrics": metrics
        }
        self.performance_history.append(record)

        # Check for degradation
        alerts = self._check_degradation(metrics)
        if alerts:
            record["alerts"] = alerts
            logger.warning(f"Performance alerts: {alerts}")

        return record

    def _check_degradation(self, current_metrics: Dict[str, float]) -> List[Dict]:
        """Check for performance degradation"""
        if self.baseline_metrics is None:
            return []

        alerts = []

        for metric, baseline_value in self.baseline_metrics.items():
            if metric not in current_metrics:
                continue

            current_value = current_metrics[metric]
            drop = (baseline_value - current_value) / baseline_value

            if drop > self.alert_threshold:
                alerts.append({
                    "metric": metric,
                    "baseline": baseline_value,
                    "current": current_value,
                    "drop_percentage": f"{drop:.1%}",
                    "severity": "critical" if drop > 0.2 else "warning"
                })

        return alerts

    def get_performance_trend(
        self,
        metric: str,
        window_hours: int = 24
    ) -> Dict:
        """Get performance trend for a specific metric"""
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)

        recent_records = [
            r for r in self.performance_history
            if datetime.fromisoformat(r["timestamp"]) > cutoff
        ]

        if not recent_records:
            return {"error": "No data in time window"}

        values = [r["metrics"].get(metric, 0) for r in recent_records]

        return {
            "metric": metric,
            "window_hours": window_hours,
            "data_points": len(values),
            "current": values[-1] if values else None,
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "trend": "stable" if np.std(values) < 0.05 else ("improving" if values[-1] > values[0] else "degrading")
        }

    def generate_report(self) -> Dict:
        """Generate performance monitoring report"""
        if not self.performance_history:
            return {"error": "No performance data recorded"}

        latest = self.performance_history[-1]

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "baseline_metrics": self.baseline_metrics,
            "latest_metrics": latest["metrics"],
            "total_records": len(self.performance_history),
            "alerts": latest.get("alerts", [])
        }


if __name__ == "__main__":
    # Demo
    logger.info("="*70)
    logger.info("Drift Detection & Performance Monitoring Demo")
    logger.info("="*70)

    # Create sample reference data
    np.random.seed(42)
    reference_df = pd.DataFrame({
        'V1': np.random.normal(0, 1, 1000),
        'V2': np.random.normal(0, 1, 1000),
        'Amount': np.random.exponential(100, 1000)
    })

    # Initialize drift detector
    detector = DriftDetector(reference_data=reference_df)

    # Simulate incoming data (with drift in V1)
    for i in range(500):
        features = {
            'V1': np.random.normal(0.5, 1),  # Drift!
            'V2': np.random.normal(0, 1),
            'Amount': np.random.exponential(100)
        }
        prediction = np.random.random()
        detector.add_observation(features, prediction)

    # Check for drift
    drift_result = detector.detect_data_drift()
    print("\nDrift Detection Results:")
    print(f"  Drift detected: {drift_result['drift_detected']}")
    print(f"  Drifted features: {drift_result['drifted_features']}")

    # Performance monitoring
    monitor = PerformanceMonitor()
    monitor.set_baseline({"roc_auc": 0.90, "precision": 0.85})

    # Record some performance (simulating degradation)
    monitor.record_performance({"roc_auc": 0.88, "precision": 0.83})
    monitor.record_performance({"roc_auc": 0.85, "precision": 0.80})
    monitor.record_performance({"roc_auc": 0.78, "precision": 0.72})  # Significant drop

    # Generate report
    report = monitor.generate_report()
    print("\nPerformance Report:")
    print(f"  Baseline: {report['baseline_metrics']}")
    print(f"  Latest: {report['latest_metrics']}")
    print(f"  Alerts: {report['alerts']}")

    logger.info("\nDrift detection demo completed!")
