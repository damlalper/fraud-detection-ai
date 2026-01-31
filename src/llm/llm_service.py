"""
LLM-based Fraud Explanation Service (Hugging Face - FREE)
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("llm_service")


class FraudLLMExplainer:
    """LLM-powered fraud explanation generator using Hugging Face (FREE)"""

    def __init__(
        self,
        provider: str = "huggingface",
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM service

        Args:
            provider: LLM provider ('huggingface', 'openai', 'anthropic')
            model_name: Model identifier
            api_key: API key (optional for HuggingFace)
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_TOKEN")

        logger.info(f"Initializing {provider} LLM: {model_name}")

        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == "huggingface":
            try:
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(token=self.api_key)
                logger.info("✓ Hugging Face InferenceClient initialized (FREE)")
            except ImportError:
                logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
                raise

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("✓ OpenAI client initialized")
            except ImportError:
                logger.error("openai not installed. Run: pip install openai")
                raise

        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                logger.info("✓ Anthropic client initialized")
            except ImportError:
                logger.error("anthropic not installed. Run: pip install anthropic")
                raise

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate_explanation(
        self,
        shap_explanation: Dict,
        rag_context: Optional[str] = None,
        transaction_data: Optional[Dict] = None,
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> str:
        """
        Generate LLM-powered fraud explanation

        Args:
            shap_explanation: SHAP explanation dictionary
            rag_context: RAG-retrieved policy context (optional)
            transaction_data: Original transaction data (optional)
            max_tokens: Maximum response tokens
            temperature: LLM temperature (0-1)

        Returns:
            Human-readable fraud explanation
        """
        # Build prompt
        prompt = self._build_prompt(shap_explanation, rag_context, transaction_data)

        logger.info(f"Generating explanation with {self.provider}...")

        try:
            if self.provider == "huggingface":
                response = self._call_huggingface(prompt, max_tokens, temperature)

            elif self.provider == "openai":
                response = self._call_openai(prompt, max_tokens, temperature)

            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt, max_tokens, temperature)

            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            logger.info("✓ Explanation generated successfully")
            return response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to template-based explanation
            return self._fallback_explanation(shap_explanation)

    def _build_prompt(
        self,
        shap_explanation: Dict,
        rag_context: Optional[str],
        transaction_data: Optional[Dict]
    ) -> str:
        """Build LLM prompt from SHAP explanation and context"""

        score = shap_explanation['prediction_score']
        is_fraud = score >= 0.5
        top_features = shap_explanation['top_features'][:5]

        # Format top risk factors
        risk_factors = []
        for i, feature in enumerate(top_features, 1):
            direction = "increases" if feature['shap_value'] > 0 else "decreases"
            risk_factors.append(
                f"{i}. {feature['feature']} = {feature['value']:.2f} "
                f"({direction} fraud risk by {feature['magnitude']:.4f})"
            )

        risk_factors_text = '\n'.join(risk_factors)

        # Build prompt - Turkish friendly and user-focused
        prompt = f"""Sen bir yapay zeka fraud analisti olarak müşterilere işlemlerini açıklıyorsun.
Aşağıdaki kredi kartı işlemini analiz et ve neden {"dolandırıcılık olarak işaretlendiğini" if is_fraud else "güvenli bulunduğunu"} SADE VE ANLAŞILIR bir dille açıkla.

İŞLEM ANALİZİ:
Fraud Skoru: {score:.3f} (0=güvenli, 1=riskli)
Sonuç: {"⚠️ RİSKLİ İŞLEM" if is_fraud else "✓ GÜVENLİ İŞLEM"}

ÖNEMLİ FAKTÖRLER:
{risk_factors_text}

GÖREV:
Bu işlemi müşteriye açıkla. 2-3 cümle kullan. Teknik terimler yerine günlük dil kullan.
Örnek: "V14 değeri" demek yerine "işlem karakteristikleri", "anomali skorları" gibi ifadeler kullan.
Müşterinin endişelenmesine veya rahatlamasına yardımcı ol.
Eğer riskli ise ne yapması gerektiğini söyle. Güvenli ise neden güvenli olduğunu net açıkla.
"""

        # Add RAG context if available
        if rag_context:
            prompt += f"""

ŞİRKET POLİTİKALARI (sadece referans için, müşteriye direkt bahsetme):
{rag_context}


"""

        # Add transaction data if available
        if transaction_data:
            amount = transaction_data.get('Amount', 0)
            prompt += f"""

İŞLEM DETAYLARI:
Tutar: ${amount:.2f}
"""

        prompt += "\n\nMÜŞTERİYE AÇIKLAMA (Türkçe, sade dil):"

        return prompt

    def _call_huggingface(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Hugging Face Inference API (FREE)"""

        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False
            )

            return response.strip()

        except Exception as e:
            logger.warning(f"Hugging Face API error: {e}")
            logger.info("Note: First request may be slow (model loading)")
            raise

    def _call_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call OpenAI API"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a fraud detection AI analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content.strip()

    def _call_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Anthropic Claude API"""

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text.strip()

    def _fallback_explanation(self, shap_explanation: Dict) -> str:
        """Fallback template-based explanation if LLM fails - User friendly Turkish"""

        score = shap_explanation['prediction_score']
        is_fraud = score >= 0.5
        top_features = shap_explanation['top_features'][:3]

        if is_fraud:
            explanation = f"⚠️ Bu işlem riskli olarak değerlendirildi (risk skoru: {score:.0%}). "
            explanation += "Sistemimiz bu işlemde şüpheli aktivite tespit etti. "

            # Add actionable advice
            if score > 0.8:
                explanation += "İşleminizi onaylamadan önce lütfen bizimle iletişime geçin. "
            elif score > 0.6:
                explanation += "Eğer bu işlemi siz yapmadıysanız, kartınızı bloke etmenizi öneririz. "
            else:
                explanation += "İşleminiz inceleme altında, sonuç SMS ile bildirilecektir. "
        else:
            explanation = f"✓ Bu işlem güvenli bulundu (güvenlik skoru: {(1-score):.0%}). "
            explanation += "Sistemimiz işleminizde herhangi bir anormallik tespit etmedi. "

            # Explain why it's safe
            top_safe_factor = top_features[0] if top_features and top_features[0]['shap_value'] < 0 else None
            if top_safe_factor:
                explanation += f"İşlem karakteristikleri normal kullanım paternlerinize uygun. "

        return explanation


if __name__ == "__main__":
    # Example usage
    import json
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent

    # Sample SHAP explanation
    sample_explanation = {
        'prediction_score': 0.87,
        'base_value': 0.02,
        'top_features': [
            {
                'feature': 'V14',
                'value': -19.45,
                'shap_value': 0.35,
                'impact': 'increases fraud risk',
                'magnitude': 0.35
            },
            {
                'feature': 'Amount',
                'value': 1234.56,
                'shap_value': 0.28,
                'impact': 'increases fraud risk',
                'magnitude': 0.28
            },
            {
                'feature': 'V17',
                'value': -25.12,
                'shap_value': 0.22,
                'impact': 'increases fraud risk',
                'magnitude': 0.22
            }
        ]
    }

    # Sample transaction data
    transaction_data = {
        'Amount': 1234.56,
        'Time': 13547
    }

    # Sample RAG context
    rag_context = """
    FRAUD DETECTION POLICY:
    - Transactions over $1000 require enhanced monitoring
    - Abnormal V14 values (< -15 or > 15) indicate potential fraud
    - High-value transactions outside normal hours (9 AM - 9 PM) are suspicious
    """

    # Initialize LLM explainer
    logger.info("="*70)
    logger.info("Testing Hugging Face LLM Explainer (FREE)")
    logger.info("="*70)

    explainer = FraudLLMExplainer(
        provider="huggingface",
        model_name="mistralai/Mistral-7B-Instruct-v0.2"
    )

    # Generate explanation
    logger.info("\nGenerating fraud explanation...")

    try:
        explanation = explainer.generate_explanation(
            shap_explanation=sample_explanation,
            rag_context=rag_context,
            transaction_data=transaction_data,
            max_tokens=300,
            temperature=0.3
        )

        logger.info("\n" + "="*70)
        logger.info("LLM-GENERATED FRAUD EXPLANATION:")
        logger.info("="*70)
        print(f"\n{explanation}\n")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Failed to generate LLM explanation: {e}")
        logger.info("\nUsing fallback explanation...")

        fallback = explainer._fallback_explanation(sample_explanation)
        logger.info("\n" + "="*70)
        logger.info("FALLBACK EXPLANATION:")
        logger.info("="*70)
        print(f"\n{fallback}\n")
        logger.info("="*70)

    logger.info("\n✓ LLM service demo completed!")
