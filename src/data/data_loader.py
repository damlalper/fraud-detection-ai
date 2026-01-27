"""
Data loading utilities for fraud detection datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("data_loader")


class FraudDataLoader:
    """Load and prepare fraud detection datasets"""

    def __init__(self, data_path: str):
        """
        Initialize data loader

        Args:
            data_path: Path to the dataset file
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from file

        Returns:
            DataFrame with transaction data
        """
        logger.info(f"Loading data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        # Load based on file extension
        if self.data_path.suffix == '.csv':
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        logger.info(f"Loaded {len(self.df)} transactions")
        logger.info(f"Columns: {list(self.df.columns)}")

        return self.df

    def get_basic_stats(self) -> dict:
        """
        Get basic statistics about the dataset

        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        fraud_count = self.df['Class'].sum()
        total_count = len(self.df)
        fraud_rate = fraud_count / total_count * 100

        stats = {
            'total_transactions': total_count,
            'fraud_transactions': int(fraud_count),
            'legitimate_transactions': int(total_count - fraud_count),
            'fraud_rate_percent': round(fraud_rate, 2),
            'features': list(self.df.columns),
            'num_features': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
        }

        logger.info(f"Dataset Stats: {stats}")
        return stats

    def split_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataset into features and target

        Returns:
            Tuple of (features, target)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        X = self.df.drop('Class', axis=1)
        y = self.df['Class']

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    def get_fraud_examples(self, n: int = 5) -> pd.DataFrame:
        """
        Get sample fraud transactions

        Args:
            n: Number of examples to return

        Returns:
            DataFrame with fraud examples
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        fraud_samples = self.df[self.df['Class'] == 1].head(n)
        return fraud_samples

    def get_legitimate_examples(self, n: int = 5) -> pd.DataFrame:
        """
        Get sample legitimate transactions

        Args:
            n: Number of examples to return

        Returns:
            DataFrame with legitimate examples
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        legit_samples = self.df[self.df['Class'] == 0].head(n)
        return legit_samples


if __name__ == "__main__":
    # Example usage
    loader = FraudDataLoader("../../data/raw/sample/creditcard.csv")
    df = loader.load_data()
    stats = loader.get_basic_stats()
    print(f"\nDataset Statistics:\n{stats}")
