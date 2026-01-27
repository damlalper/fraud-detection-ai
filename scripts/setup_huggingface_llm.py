"""
Setup Hugging Face for FREE LLM usage
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.logger import setup_logger

logger = setup_logger("setup_huggingface")


def test_huggingface_api():
    """Test Hugging Face API access"""
    try:
        from huggingface_hub import InferenceClient

        logger.info("Testing Hugging Face Inference API (FREE)...")

        # Use free inference API
        client = InferenceClient()

        # Test with a small model
        test_prompt = "Explain why a transaction might be fraudulent."

        logger.info(f"Sending test prompt: {test_prompt}")

        response = client.text_generation(
            test_prompt,
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=100
        )

        logger.info(f"✓ Response received: {response[:100]}...")
        logger.info("✓ Hugging Face API working!")

        return True

    except ImportError:
        logger.error("huggingface_hub not installed")
        logger.info("Installing: pip install huggingface_hub")
        os.system("pip install huggingface_hub")
        return False
    except Exception as e:
        logger.warning(f"API test failed: {e}")
        logger.info("You may need to set HF_TOKEN for better access")
        return False


def setup_hf_token():
    """Guide user to setup HF token (optional but recommended)"""
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        logger.info("✓ HF_TOKEN found in environment")
        return True

    logger.info("\n" + "="*70)
    logger.info("HUGGING FACE TOKEN SETUP (Optional - for better rate limits)")
    logger.info("="*70)
    logger.info("1. Go to: https://huggingface.co/settings/tokens")
    logger.info("2. Create a new token (Read access is enough)")
    logger.info("3. Add to .env file: HF_TOKEN=your_token_here")
    logger.info("="*70 + "\n")

    return False


def recommend_models():
    """Recommend best free models"""
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDED FREE LLM MODELS")
    logger.info("="*70)

    models = [
        {
            "name": "Mistral-7B-Instruct",
            "id": "mistralai/Mistral-7B-Instruct-v0.2",
            "speed": "Fast",
            "quality": "High",
            "cost": "FREE"
        },
        {
            "name": "Llama-2-7B-Chat",
            "id": "meta-llama/Llama-2-7b-chat-hf",
            "speed": "Fast",
            "quality": "High",
            "cost": "FREE"
        },
        {
            "name": "Phi-2",
            "id": "microsoft/phi-2",
            "speed": "Very Fast",
            "quality": "Good",
            "cost": "FREE"
        },
        {
            "name": "Zephyr-7B",
            "id": "HuggingFaceH4/zephyr-7b-beta",
            "speed": "Fast",
            "quality": "High",
            "cost": "FREE"
        }
    ]

    for model in models:
        logger.info(f"\n{model['name']}")
        logger.info(f"  ID: {model['id']}")
        logger.info(f"  Speed: {model['speed']}")
        logger.info(f"  Quality: {model['quality']}")
        logger.info(f"  Cost: {model['cost']}")

    logger.info("\n" + "="*70)
    logger.info("We'll use Mistral-7B-Instruct (best balance)")
    logger.info("="*70 + "\n")


def main():
    """Main setup function"""
    logger.info("="*70)
    logger.info("HUGGING FACE FREE LLM SETUP")
    logger.info("="*70)

    # Install required packages
    logger.info("\nInstalling required packages...")
    os.system("pip install -q huggingface_hub transformers")

    # Setup token (optional)
    setup_hf_token()

    # Show recommended models
    recommend_models()

    # Test API
    logger.info("Testing Hugging Face API...")
    success = test_huggingface_api()

    if success:
        logger.info("\n✓ HUGGING FACE READY FOR USE!")
        logger.info("✓ 100% FREE - No API costs!")
    else:
        logger.info("\n⚠ API test failed, but you can still use it")
        logger.info("Note: First request may be slow (model loading)")

    logger.info("\n" + "="*70)
    logger.info("NEXT: Update .env file")
    logger.info("="*70)
    logger.info("LLM_PROVIDER=huggingface")
    logger.info("LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
