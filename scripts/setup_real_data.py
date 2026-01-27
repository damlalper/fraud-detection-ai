"""
Download real Kaggle Credit Card Fraud dataset
"""
import os
import sys
from pathlib import Path
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.logger import setup_logger

logger = setup_logger("setup_real_data")


def download_from_url(url: str, output_path: Path) -> None:
    """
    Download file from direct URL

    Args:
        url: Download URL
        output_path: Where to save the file
    """
    logger.info(f"Downloading from {url}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if count % 100 == 0:  # Print every 100 blocks
                logger.info(f"Download progress: {percent}%")

        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        logger.info(f"Downloaded successfully to {output_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def setup_kaggle_credentials():
    """Guide user to setup Kaggle credentials"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        logger.info("Kaggle credentials found!")
        return True

    logger.warning("Kaggle credentials not found!")
    logger.info("\n" + "="*70)
    logger.info("KAGGLE SETUP INSTRUCTIONS:")
    logger.info("="*70)
    logger.info("1. Go to: https://www.kaggle.com/settings")
    logger.info("2. Scroll to 'API' section")
    logger.info("3. Click 'Create New Token'")
    logger.info("4. Download kaggle.json file")
    logger.info(f"5. Move it to: {kaggle_dir}")
    logger.info("6. Run this script again")
    logger.info("="*70 + "\n")

    return False


def download_kaggle_dataset():
    """Download dataset using Kaggle API"""
    try:
        import kaggle

        project_root = Path(__file__).parent.parent
        output_dir = project_root / "data" / "raw" / "creditcard"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading Credit Card Fraud Dataset from Kaggle...")
        logger.info("Dataset: mlg-ulb/creditcardfraud (143 MB)")

        # Download and unzip
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=str(output_dir),
            unzip=True
        )

        # Check if file exists
        csv_file = output_dir / "creditcard.csv"
        if csv_file.exists():
            # Get file size
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Downloaded successfully: {csv_file}")
            logger.info(f"✓ File size: {size_mb:.2f} MB")

            # Quick stats
            import pandas as pd
            df = pd.read_csv(csv_file)
            logger.info(f"✓ Total transactions: {len(df):,}")
            logger.info(f"✓ Fraud transactions: {df['Class'].sum():,}")
            logger.info(f"✓ Fraud rate: {df['Class'].mean()*100:.3f}%")

            return True
        else:
            logger.error("Download completed but file not found!")
            return False

    except ImportError:
        logger.error("Kaggle package not installed!")
        logger.info("Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def main():
    """Main function"""
    logger.info("="*70)
    logger.info("REAL KAGGLE FRAUD DATASET SETUP")
    logger.info("="*70)

    # Check if kaggle is installed
    try:
        import kaggle
        logger.info("✓ Kaggle package installed")
    except ImportError:
        logger.error("✗ Kaggle package not installed")
        logger.info("\nInstalling kaggle package...")
        os.system("pip install kaggle")

    # Check credentials
    if not setup_kaggle_credentials():
        logger.error("Cannot proceed without Kaggle credentials")
        logger.info("\nAlternatively, download manually:")
        logger.info("1. Visit: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        logger.info("2. Click 'Download'")
        logger.info("3. Extract creditcard.csv to: data/raw/creditcard/")
        return False

    # Download dataset
    success = download_kaggle_dataset()

    if success:
        logger.info("\n" + "="*70)
        logger.info("✓ REAL DATASET READY FOR TRAINING!")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("1. Run data pipeline: python src/data/pipeline.py")
        logger.info("2. Train models: python src/models/train_classical.py")
    else:
        logger.error("\n✗ Setup failed. Please check errors above.")

    return success


if __name__ == "__main__":
    main()
