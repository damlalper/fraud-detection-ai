"""
Data preprocessing pipeline for fraud detection
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("preprocessor")


class FraudDataPreprocessor:
    """Preprocess fraud detection data"""

    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize preprocessor

        Args:
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None

        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        logger.info(f"Initialized preprocessor with {scaler_type} scaler")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")

        missing_before = df.isnull().sum().sum()

        if missing_before > 0:
            logger.warning(f"Found {missing_before} missing values")

            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

            missing_after = df.isnull().sum().sum()
            logger.info(f"Missing values after handling: {missing_after}")

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate transactions

        Args:
            df: Input DataFrame

        Returns:
            DataFrame without duplicates
        """
        logger.info("Removing duplicates")

        before = len(df)
        df = df.drop_duplicates()
        after = len(df)

        removed = before - after
        if removed > 0:
            logger.info(f"Removed {removed} duplicate transactions")

        return df

    def scale_features(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features

        Args:
            X: Features DataFrame
            fit: Whether to fit the scaler (True for training, False for test/inference)

        Returns:
            Scaled features
        """
        logger.info(f"Scaling features (fit={fit})")

        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()

        # Columns to scale (usually Amount and Time if present)
        cols_to_scale = []
        if 'Amount' in X.columns:
            cols_to_scale.append('Amount')
        if 'Time' in X.columns:
            cols_to_scale.append('Time')

        if len(cols_to_scale) > 0:
            X_scaled = X.copy()

            if fit:
                X_scaled[cols_to_scale] = self.scaler.fit_transform(X[cols_to_scale])
                logger.info(f"Fitted scaler on columns: {cols_to_scale}")
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Call with fit=True first.")
                X_scaled[cols_to_scale] = self.scaler.transform(X[cols_to_scale])

            return X_scaled
        else:
            logger.info("No columns to scale")
            return X

    def create_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets

        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed
            stratify: Whether to stratify by target

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Creating train/test split (test_size={test_size})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train fraud rate: {y_train.mean():.4f}")
        logger.info(f"Test fraud rate: {y_test.mean():.4f}")

        return X_train, X_test, y_train, y_test

    def save_scaler(self, path: str) -> None:
        """
        Save the fitted scaler to disk

        Args:
            path: Path to save the scaler
        """
        if self.scaler is None:
            raise ValueError("No scaler to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, path)
        logger.info(f"Saved scaler to {path}")

    def load_scaler(self, path: str) -> None:
        """
        Load a fitted scaler from disk

        Args:
            path: Path to the saved scaler
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler not found: {path}")

        self.scaler = joblib.load(path)
        logger.info(f"Loaded scaler from {path}")

    def get_feature_names(self) -> list:
        """
        Get feature names after preprocessing

        Returns:
            List of feature names
        """
        return self.feature_names


if __name__ == "__main__":
    # Example usage
    from data_loader import FraudDataLoader

    loader = FraudDataLoader("../../data/raw/sample/creditcard.csv")
    df = loader.load_data()
    X, y = loader.split_features_target()

    preprocessor = FraudDataPreprocessor(scaler_type="robust")
    X_scaled = preprocessor.scale_features(X, fit=True)

    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(
        X_scaled, y, test_size=0.2
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
