"""
Script to run the Fraud Detection API
"""
import uvicorn
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import settings
from utils.logger import setup_logger

logger = setup_logger("run_api")


def main():
    """Run the FastAPI server"""
    logger.info("="*70)
    logger.info("Starting AI-Powered Fraud Detection API")
    logger.info("="*70)
    logger.info(f"Host: {settings.api_host}")
    logger.info(f"Port: {settings.api_port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"SHAP Explanation: {settings.enable_shap_explanation}")
    logger.info(f"RAG + LLM: {settings.enable_rag}")
    logger.info("="*70)

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
        workers=1 if settings.api_reload else settings.api_workers
    )


if __name__ == "__main__":
    main()
