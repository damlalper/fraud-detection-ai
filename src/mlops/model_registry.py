"""
Model Registry & Deployment Manager
Handles model versioning, staging, and production deployment
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from enum import Enum
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("model_registry")


class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistry:
    """Local model registry for versioning and deployment"""

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize model registry

        Args:
            registry_path: Path to registry directory
        """
        if registry_path is None:
            project_root = Path(__file__).parent.parent.parent
            registry_path = str(project_root / "model_registry")

        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_path / "registry.json"
        self._load_registry()

        logger.info(f"Model registry initialized: {self.registry_path}")

    def _load_registry(self):
        """Load registry metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": {}, "production": {}}

    def _save_registry(self):
        """Save registry metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict[str, float],
        params: Optional[Dict] = None,
        tags: Optional[Dict[str, str]] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT
    ) -> str:
        """
        Register a new model version

        Args:
            model_name: Name of the model
            model_path: Path to model file
            metrics: Model performance metrics
            params: Model hyperparameters
            tags: Additional tags

        Returns:
            Version string
        """
        # Initialize model entry if needed
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {
                "versions": [],
                "latest_version": 0
            }

        # Create new version
        version = self.registry["models"][model_name]["latest_version"] + 1
        version_str = f"v{version}"

        # Create version directory
        version_dir = self.registry_path / model_name / version_str
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        model_file = Path(model_path)
        dest_path = version_dir / model_file.name
        shutil.copy2(model_path, dest_path)

        # Create version metadata
        version_metadata = {
            "version": version_str,
            "model_path": str(dest_path),
            "metrics": metrics,
            "params": params or {},
            "tags": tags or {},
            "stage": stage.value,
            "created_at": datetime.utcnow().isoformat(),
            "created_by": os.getenv("USER", "system")
        }

        # Save version metadata
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(version_metadata, f, indent=2)

        # Update registry
        self.registry["models"][model_name]["versions"].append(version_metadata)
        self.registry["models"][model_name]["latest_version"] = version
        self._save_registry()

        logger.info(f"Registered model: {model_name} {version_str}")
        return version_str

    def promote_model(
        self,
        model_name: str,
        version: str,
        target_stage: ModelStage
    ):
        """Promote model to a new stage"""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model not found: {model_name}")

        # Find version
        for v in self.registry["models"][model_name]["versions"]:
            if v["version"] == version:
                old_stage = v["stage"]
                v["stage"] = target_stage.value
                v["promoted_at"] = datetime.utcnow().isoformat()

                # Update production pointer
                if target_stage == ModelStage.PRODUCTION:
                    self.registry["production"][model_name] = version

                self._save_registry()
                logger.info(f"Promoted {model_name} {version}: {old_stage} -> {target_stage.value}")
                return

        raise ValueError(f"Version not found: {version}")

    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """Get the current production model"""
        if model_name not in self.registry["production"]:
            return None

        version = self.registry["production"][model_name]
        return self.get_model_version(model_name, version)

    def get_model_version(self, model_name: str, version: str) -> Optional[Dict]:
        """Get specific model version metadata"""
        if model_name not in self.registry["models"]:
            return None

        for v in self.registry["models"][model_name]["versions"]:
            if v["version"] == version:
                return v
        return None

    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.registry["models"].keys())

    def list_versions(self, model_name: str) -> List[Dict]:
        """List all versions of a model"""
        if model_name not in self.registry["models"]:
            return []
        return self.registry["models"][model_name]["versions"]

    def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str
    ) -> Dict:
        """Compare two model versions"""
        v_a = self.get_model_version(model_name, version_a)
        v_b = self.get_model_version(model_name, version_b)

        if not v_a or not v_b:
            raise ValueError("Version not found")

        comparison = {
            "versions": [version_a, version_b],
            "metrics_comparison": {},
            "params_comparison": {}
        }

        # Compare metrics
        all_metrics = set(v_a["metrics"].keys()) | set(v_b["metrics"].keys())
        for metric in all_metrics:
            val_a = v_a["metrics"].get(metric, None)
            val_b = v_b["metrics"].get(metric, None)
            comparison["metrics_comparison"][metric] = {
                version_a: val_a,
                version_b: val_b,
                "diff": (val_b - val_a) if val_a and val_b else None
            }

        return comparison

    def delete_version(self, model_name: str, version: str):
        """Delete a model version (only if not in production)"""
        if self.registry["production"].get(model_name) == version:
            raise ValueError("Cannot delete production model")

        if model_name not in self.registry["models"]:
            return

        # Remove from versions list
        self.registry["models"][model_name]["versions"] = [
            v for v in self.registry["models"][model_name]["versions"]
            if v["version"] != version
        ]

        # Delete files
        version_dir = self.registry_path / model_name / version
        if version_dir.exists():
            shutil.rmtree(version_dir)

        self._save_registry()
        logger.info(f"Deleted {model_name} {version}")


if __name__ == "__main__":
    # Demo
    logger.info("="*70)
    logger.info("Model Registry Demo")
    logger.info("="*70)

    registry = ModelRegistry()

    # Register a model
    version = registry.register_model(
        model_name="xgboost_fraud",
        model_path="models/xgboost_fraud_model.pkl",
        metrics={"roc_auc": 0.89, "precision": 0.85, "recall": 0.78},
        params={"n_estimators": 100, "max_depth": 6},
        tags={"trained_by": "demo"}
    )

    print(f"\nRegistered: {version}")

    # List models
    print(f"\nRegistered models: {registry.list_models()}")

    # Promote to staging
    registry.promote_model("xgboost_fraud", version, ModelStage.STAGING)

    # Promote to production
    registry.promote_model("xgboost_fraud", version, ModelStage.PRODUCTION)

    # Get production model
    prod = registry.get_production_model("xgboost_fraud")
    print(f"\nProduction model: {prod['version']}")
    print(f"Metrics: {prod['metrics']}")

    logger.info("\nModel registry demo completed!")
