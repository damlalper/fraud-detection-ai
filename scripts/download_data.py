"""
Script to download fraud detection datasets
"""
import os
import sys
from pathlib import Path
import urllib.request
import zipfile
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger

logger = setup_logger("download_data")


def download_file(url: str, destination: Path) -> None:
    """
    Download a file from URL to destination

    Args:
        url: URL to download from
        destination: Path to save the file
    """
    logger.info(f"Downloading {url} to {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, destination)
        logger.info(f"Successfully downloaded {destination.name}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extract a zip file

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting {zip_path} to {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Successfully extracted {zip_path.name}")
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        raise


def download_kaggle_dataset(dataset_name: str, output_dir: Path) -> None:
    """
    Download dataset from Kaggle using kaggle CLI

    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'mlg-ulb/creditcardfraud')
        output_dir: Directory to save the dataset
    """
    logger.info(f"Downloading Kaggle dataset: {dataset_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=True
        )
        logger.info(f"Successfully downloaded {dataset_name}")
    except ImportError:
        logger.error("Kaggle package not installed. Install with: pip install kaggle")
        logger.info("Please download datasets manually from Kaggle")
        raise
    except Exception as e:
        logger.error(f"Failed to download {dataset_name}: {e}")
        logger.info("Make sure you have Kaggle API credentials configured")
        logger.info("See: https://github.com/Kaggle/kaggle-api#api-credentials")
        raise


def create_sample_data(output_path: Path) -> None:
    """
    Create a sample fraud dataset for testing

    Args:
        output_path: Path to save sample data
    """
    logger.info(f"Creating sample data at {output_path}")

    import pandas as pd
    import numpy as np

    np.random.seed(42)

    # Create sample transactions
    n_samples = 10000
    n_fraud = 200  # 2% fraud rate

    data = {
        'Time': np.random.randint(0, 172800, n_samples),
        'Amount': np.random.exponential(50, n_samples),
    }

    # Add V1-V28 features (PCA components in real dataset)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)

    # Add Class (0 = legitimate, 1 = fraud)
    data['Class'] = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    data['Class'][fraud_indices] = 1

    df = pd.DataFrame(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Created sample dataset with {n_samples} transactions ({n_fraud} fraudulent)")


def main():
    """Main function to download all datasets"""
    # Get project root directory
    project_root = Path(__file__).parent.parent
    data_raw_dir = project_root / "data" / "raw"

    logger.info("Starting dataset download process")

    # Option 1: Download from Kaggle (requires API credentials)
    try:
        logger.info("Attempting to download from Kaggle...")

        # Credit Card Fraud Dataset
        download_kaggle_dataset(
            "mlg-ulb/creditcardfraud",
            data_raw_dir / "creditcard"
        )

        # IEEE-CIS Fraud Detection Dataset
        download_kaggle_dataset(
            "c/ieee-fraud-detection",
            data_raw_dir / "ieee-fraud"
        )

        logger.info("All datasets downloaded successfully!")

    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        logger.info("Creating sample data instead...")

        # Option 2: Create sample data if Kaggle fails
        create_sample_data(data_raw_dir / "sample" / "creditcard.csv")

        logger.info("Sample data created successfully!")
        logger.info("\n" + "="*60)
        logger.info("IMPORTANT: Using sample data for demonstration")
        logger.info("For production, download real datasets from:")
        logger.info("  - https://www.kaggle.com/mlg-ulb/creditcardfraud")
        logger.info("  - https://www.kaggle.com/c/ieee-fraud-detection")
        logger.info("="*60 + "\n")

    # Create policy documents directory
    policies_dir = project_root / "data" / "policies"
    policies_dir.mkdir(parents=True, exist_ok=True)

    # Create sample fraud policy document
    sample_policy = policies_dir / "fraud_policy.txt"
    if not sample_policy.exists():
        with open(sample_policy, 'w') as f:
            f.write("""
FRAUD DETECTION POLICY

1. TRANSACTION MONITORING
- Monitor all transactions for unusual patterns
- Flag transactions with abnormal amounts
- Detect location anomalies and velocity changes

2. RISK FACTORS
- High-value transactions (>$500)
- Multiple transactions in short time periods
- Unusual location or country changes
- Abnormal merchant categories
- New device or IP address usage

3. THRESHOLDS
- Standard risk: 0.3-0.5
- High risk: 0.5-0.7
- Critical risk: >0.7

4. COMPLIANCE
- All flagged transactions must be reviewed within 24 hours
- Maintain audit trail for all decisions
- Report suspicious activities to relevant authorities
            """)
        logger.info("Created sample fraud policy document")


if __name__ == "__main__":
    main()
