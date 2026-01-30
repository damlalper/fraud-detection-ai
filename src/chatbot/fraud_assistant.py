"""
Intelligent Fraud Analysis Assistant
LLM-powered chatbot for fraud investigation and knowledge management
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm.llm_service import FraudLLMExplainer
from rag.rag_system import FraudPolicyRAG
from utils.logger import setup_logger

logger = setup_logger("fraud_assistant")


class ConversationMemory:
    """Manages conversation history for context"""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict] = []

    def add_turn(self, role: str, content: str):
        """Add a conversation turn"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Keep only recent turns
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get_context(self) -> str:
        """Get conversation context for LLM"""
        if not self.history:
            return ""

        context = "Previous conversation:\n"
        for turn in self.history[-6:]:  # Last 3 exchanges
            role = "User" if turn["role"] == "user" else "Assistant"
            context += f"{role}: {turn['content']}\n"

        return context

    def clear(self):
        """Clear conversation history"""
        self.history = []


class FraudAssistant:
    """
    Intelligent assistant for fraud analysis and investigation
    Combines LLM capabilities with RAG for policy-aware responses
    """

    def __init__(
        self,
        llm_provider: str = "huggingface",
        enable_rag: bool = True
    ):
        """
        Initialize fraud assistant

        Args:
            llm_provider: LLM provider to use
            enable_rag: Whether to enable RAG for policy retrieval
        """
        logger.info("Initializing Fraud Analysis Assistant...")

        # Initialize LLM
        self.llm = FraudLLMExplainer(provider=llm_provider)

        # Initialize RAG
        self.rag = None
        if enable_rag:
            try:
                self.rag = FraudPolicyRAG()
                project_root = Path(__file__).parent.parent.parent
                self.rag.load_policy_documents(str(project_root / "data" / "policies"))
                logger.info("RAG system loaded")
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}")

        # Conversation memory
        self.memory = ConversationMemory()

        # Intent patterns
        self.intents = {
            "explain_transaction": ["explain", "why", "fraud", "suspicious", "analyze"],
            "policy_query": ["policy", "rule", "regulation", "compliance", "threshold"],
            "risk_factors": ["risk", "factor", "feature", "important", "contribute"],
            "general_help": ["help", "how", "what", "guide", "assist"],
            "statistics": ["stats", "statistics", "report", "summary", "metrics"]
        }

        logger.info("Fraud Assistant ready")

    def _detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()

        for intent, keywords in self.intents.items():
            if any(kw in message_lower for kw in keywords):
                return intent

        return "general"

    def _build_prompt(
        self,
        user_message: str,
        intent: str,
        rag_context: Optional[str] = None
    ) -> str:
        """Build prompt for LLM"""
        system_context = """You are an expert fraud analyst assistant. Your role is to:
1. Help investigate suspicious transactions
2. Explain fraud detection decisions
3. Answer questions about fraud policies and procedures
4. Provide guidance on risk factors and patterns

Be concise, professional, and helpful. Use specific examples when relevant."""

        prompt = f"{system_context}\n\n"

        # Add conversation context
        conv_context = self.memory.get_context()
        if conv_context:
            prompt += f"{conv_context}\n"

        # Add RAG context
        if rag_context:
            prompt += f"Relevant policy information:\n{rag_context}\n\n"

        # Add intent-specific instructions
        if intent == "explain_transaction":
            prompt += "Focus on explaining why the transaction was flagged and the key risk factors.\n"
        elif intent == "policy_query":
            prompt += "Reference specific policies and thresholds in your response.\n"
        elif intent == "risk_factors":
            prompt += "Explain the risk factors and their significance in fraud detection.\n"

        prompt += f"\nUser question: {user_message}\n\nAssistant response:"

        return prompt

    def chat(
        self,
        message: str,
        transaction_context: Optional[Dict] = None
    ) -> str:
        """
        Process a chat message and generate response

        Args:
            message: User message
            transaction_context: Optional transaction data for context

        Returns:
            Assistant response
        """
        # Add user message to memory
        self.memory.add_turn("user", message)

        # Detect intent
        intent = self._detect_intent(message)
        logger.info(f"Detected intent: {intent}")

        # Get RAG context if relevant
        rag_context = None
        if self.rag and intent in ["policy_query", "explain_transaction", "risk_factors"]:
            try:
                context_chunks = self.rag.retrieve_context(message, top_k=2)
                rag_context = self.rag.format_context_for_llm(context_chunks)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Build prompt
        prompt = self._build_prompt(message, intent, rag_context)

        # Add transaction context if provided
        if transaction_context:
            txn_info = f"\nTransaction details:\n"
            txn_info += f"- Amount: ${transaction_context.get('amount', 'N/A')}\n"
            txn_info += f"- Fraud Score: {transaction_context.get('fraud_score', 'N/A')}\n"
            if 'risk_factors' in transaction_context:
                txn_info += f"- Top Risk Factors: {', '.join(transaction_context['risk_factors'][:3])}\n"
            prompt = prompt.replace("User question:", f"{txn_info}\nUser question:")

        # Generate response
        try:
            response = self.llm._call_llm(prompt, max_tokens=500, temperature=0.7)

            # Clean up response
            response = response.strip()
            if response.startswith("Assistant response:"):
                response = response[19:].strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response = self._fallback_response(intent, message)

        # Add assistant response to memory
        self.memory.add_turn("assistant", response)

        return response

    def _fallback_response(self, intent: str, message: str) -> str:
        """Generate fallback response when LLM fails"""
        fallbacks = {
            "explain_transaction": "This transaction was flagged based on unusual patterns in the data. Key factors typically include transaction amount, timing, and historical behavior patterns.",
            "policy_query": "Our fraud detection policies include thresholds for transaction amounts, velocity checks, and geographic analysis. Please refer to the policy documentation for specific details.",
            "risk_factors": "Common fraud risk factors include: unusual transaction amounts, abnormal timing patterns, geographic anomalies, and deviations from historical behavior.",
            "statistics": "For detailed statistics and reports, please check the monitoring dashboard or contact the analytics team.",
            "general_help": "I can help you with: explaining fraud decisions, policy questions, risk factor analysis, and general fraud investigation guidance.",
            "general": "I'm here to help with fraud analysis and investigation. Please ask about specific transactions, policies, or risk factors."
        }
        return fallbacks.get(intent, fallbacks["general"])

    def analyze_transaction(
        self,
        transaction_id: str,
        features: Dict[str, float],
        fraud_score: float,
        shap_values: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Provide detailed analysis of a transaction

        Args:
            transaction_id: Transaction identifier
            features: Transaction features
            fraud_score: Model's fraud probability
            shap_values: SHAP feature attributions

        Returns:
            Analysis text
        """
        # Build context
        context = {
            "transaction_id": transaction_id,
            "fraud_score": fraud_score,
            "amount": features.get("Amount", 0),
            "risk_factors": []
        }

        if shap_values:
            # Get top contributing features
            sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            context["risk_factors"] = [f[0] for f in sorted_features[:5]]

        # Generate analysis
        message = f"Analyze transaction {transaction_id} with fraud score {fraud_score:.2%}"
        return self.chat(message, transaction_context=context)

    def get_policy_guidance(self, topic: str) -> str:
        """Get policy guidance on a specific topic"""
        return self.chat(f"What are the policies and procedures for {topic}?")

    def clear_conversation(self):
        """Clear conversation history"""
        self.memory.clear()
        logger.info("Conversation cleared")


# Interactive CLI interface
def run_interactive_chat():
    """Run interactive chat interface"""
    print("\n" + "="*60)
    print("    FRAUD ANALYSIS ASSISTANT")
    print("    Type 'quit' to exit, 'clear' to reset conversation")
    print("="*60 + "\n")

    assistant = FraudAssistant()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("\nGoodbye!")
                break

            if user_input.lower() == 'clear':
                assistant.clear_conversation()
                print("\nConversation cleared.")
                continue

            response = assistant.chat(user_input)
            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    run_interactive_chat()
