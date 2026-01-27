"""
Complete data pipeline for fraud detection
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import FraudDataLoader
from data.preprocessor import FraudDataPreprocessor
from data.feature_engineering import FeatureEngineer
from utils.logger import setup_logger

logger = setup_logger("data_pipeline")


class DataPipeline:
    """End-to-end data pipeline for fraud detection"""

    def __init__(
        self,
        data_path: str,
        scaler_type: str = "robust",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize data pipeline

        Args:
            data_path: Path to raw dataset
            scaler_type: Type of scaler ('standard' or 'robust')
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state

        # Initialize components
        self.loader = FraudDataLoader(data_path)
        self.preprocessor = FraudDataPreprocessor(scaler_type)
        self.feature_engineer = FeatureEngineer()

        logger.info("Initialized DataPipeline")

    def run_pipeline(
        self,
        engineer_features: bool = True,
        save_processed: bool = True,
        output_dir: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Run the complete data pipeline

        Args:
            engineer_features: Whether to create engineered features
            save_processed: Whether to save processed data
            output_dir: Directory to save processed data

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("="*60)
        logger.info("Starting Data Pipeline")
        logger.info("="*60)

        # Step 1: Load data
        logger.info("\n[1/6] Loading data...")
        df = self.loader.load_data()

        # Step 2: Get basic statistics
        logger.info("\n[2/6] Analyzing data...")
        stats = self.loader.get_basic_stats()

        # Step 3: Clean data
        logger.info("\n[3/6] Cleaning data...")
        df = self.preprocessor.handle_missing_values(df)
        df = self.preprocessor.remove_duplicates(df)

        # Step 4: Feature engineering
        if engineer_features:
            logger.info("\n[4/6] Engineering features...")
            df = self.feature_engineer.create_all_features(df)
        else:
            logger.info("\n[4/6] Skipping feature engineering...")

        # Step 5: Split features and target
        logger.info("\n[5/6] Splitting features and target...")
        if 'Class' not in df.columns:
            raise ValueError("Target column 'Class' not found")

        X = df.drop('Class', axis=1)
        y = df['Class']

        # Step 6: Scale and split
        logger.info("\n[6/6] Scaling and splitting data...")
        X_scaled = self.preprocessor.scale_features(X, fit=True)

        X_train, X_test, y_train, y_test = self.preprocessor.create_train_test_split(
            X_scaled, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=True
        )

        # Save processed data
        if save_processed and output_dir:
            self._save_processed_data(
                X_train, X_test, y_train, y_test,
                output_dir
            )

        # Save scaler
        if output_dir:
            scaler_path = Path(output_dir) / "scaler.pkl"
            self.preprocessor.save_scaler(str(scaler_path))

        logger.info("\n" + "="*60)
        logger.info("Data Pipeline Completed Successfully")
        logger.info("="*60)

        return X_train, X_test, y_train, y_test

    def _save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str
    ) -> None:
        """
        Save processed data to disk

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            output_dir: Directory to save data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving processed data to {output_path}")

        # Save as parquet for efficiency
        X_train.to_parquet(output_path / "X_train.parquet")
        X_test.to_parquet(output_path / "X_test.parquet")
        y_train.to_frame('Class').to_parquet(output_path / "y_train.parquet")
        y_test.to_frame('Class').to_parquet(output_path / "y_test.parquet")

        # Save feature names
        feature_names = X_train.columns.tolist()
        with open(output_path / "feature_names.txt", 'w') as f:
            f.write('\n'.join(feature_names))

        logger.info("Saved processed data successfully")

    def load_processed_data(
        self,
        input_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load previously processed data

        Args:
            input_dir: Directory containing processed data

        Returns:
            X_train, X_test, y_train, y_test
        """
        input_path = Path(input_dir)

        logger.info(f"Loading processed data from {input_path}")

        X_train = pd.read_parquet(input_path / "X_train.parquet")
        X_test = pd.read_parquet(input_path / "X_test.parquet")
        y_train = pd.read_parquet(input_path / "y_train.parquet")['Class']
        y_test = pd.read_parquet(input_path / "y_test.parquet")['Class']

        # Load scaler
        scaler_path = input_path / "scaler.pkl"
        if scaler_path.exists():
            self.preprocessor.load_scaler(str(scaler_path))

        logger.info("Loaded processed data successfully")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "raw" / "sample" / "creditcard.csv"
    output_dir = project_root / "data" / "processed"

    pipeline = DataPipeline(
        data_path=str(data_path),
        scaler_type="robust",
        test_size=0.2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = pipeline.run_pipeline(
        engineer_features=True,
        save_processed=True,
        output_dir=str(output_dir)
    )

    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
