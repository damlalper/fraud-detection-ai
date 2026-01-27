"""
Feature engineering for fraud detection
"""
import pandas as pd
import numpy as np
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("feature_engineering")


class FeatureEngineer:
    """Create additional features for fraud detection"""

    def __init__(self):
        logger.info("Initialized FeatureEngineer")

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features

        Args:
            df: Input DataFrame with 'Time' column

        Returns:
            DataFrame with additional time features
        """
        if 'Time' not in df.columns:
            logger.warning("'Time' column not found, skipping time features")
            return df

        logger.info("Creating time-based features")

        df = df.copy()

        # Convert seconds to hours
        df['Time_hours'] = df['Time'] / 3600

        # Time of day (assuming Time is seconds from start)
        df['Time_hour_of_day'] = (df['Time'] / 3600) % 24

        # Time period categorization
        df['Time_period'] = pd.cut(
            df['Time_hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )

        # Convert categorical to numeric
        period_mapping = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
        df['Time_period_numeric'] = df['Time_period'].map(period_mapping)

        # Drop the categorical column
        df = df.drop('Time_period', axis=1)

        logger.info("Created time features: Time_hours, Time_hour_of_day, Time_period_numeric")

        return df

    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amount-based features

        Args:
            df: Input DataFrame with 'Amount' column

        Returns:
            DataFrame with additional amount features
        """
        if 'Amount' not in df.columns:
            logger.warning("'Amount' column not found, skipping amount features")
            return df

        logger.info("Creating amount-based features")

        df = df.copy()

        # Log transform of amount (add 1 to avoid log(0))
        df['Amount_log'] = np.log1p(df['Amount'])

        # Amount categories
        df['Amount_category'] = pd.cut(
            df['Amount'],
            bins=[0, 50, 200, 500, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very_High'],
            include_lowest=True
        )

        # Convert categorical to numeric
        amount_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Very_High': 3}
        df['Amount_category_numeric'] = df['Amount_category'].map(amount_mapping)

        # Drop the categorical column
        df = df.drop('Amount_category', axis=1)

        # Is high value transaction
        df['Is_high_value'] = (df['Amount'] > 500).astype(int)

        logger.info("Created amount features: Amount_log, Amount_category_numeric, Is_high_value")

        return df

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        v_features: List[str] = None
    ) -> pd.DataFrame:
        """
        Create interaction features between V features

        Args:
            df: Input DataFrame
            v_features: List of V feature columns to use

        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features")

        df = df.copy()

        # Get V columns if not specified
        if v_features is None:
            v_features = [col for col in df.columns if col.startswith('V')]

        if len(v_features) == 0:
            logger.warning("No V features found, skipping interaction features")
            return df

        # Create interactions with Amount
        if 'Amount' in df.columns:
            # V1 * Amount (example)
            if 'V1' in v_features:
                df['V1_Amount_interaction'] = df['V1'] * df['Amount']

            # V2 * Amount (example)
            if 'V2' in v_features:
                df['V2_Amount_interaction'] = df['V2'] * df['Amount']

        # Create aggregate features
        if len(v_features) > 0:
            # Mean of all V features
            df['V_mean'] = df[v_features].mean(axis=1)

            # Standard deviation of all V features
            df['V_std'] = df[v_features].std(axis=1)

            # Max and min
            df['V_max'] = df[v_features].max(axis=1)
            df['V_min'] = df[v_features].min(axis=1)

            # Range
            df['V_range'] = df['V_max'] - df['V_min']

        logger.info("Created interaction features")

        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Creating all engineered features")

        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_interaction_features(df)

        logger.info(f"Final feature count: {len(df.columns)}")

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of all feature names

        Args:
            df: DataFrame with features

        Returns:
            List of feature names
        """
        return df.columns.tolist()


if __name__ == "__main__":
    # Example usage
    from data_loader import FraudDataLoader

    loader = FraudDataLoader("../../data/raw/sample/creditcard.csv")
    df = loader.load_data()

    engineer = FeatureEngineer()
    df_engineered = engineer.create_all_features(df)

    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_engineered.columns)}")
    print(f"New features added: {len(df_engineered.columns) - len(df.columns)}")
