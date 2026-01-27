"""
RAG (Retrieval Augmented Generation) System for Fraud Policy Documents
Using ChromaDB with TF-IDF based similarity (No onnxruntime dependency)
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("rag_system")


class SimpleEmbeddingFunction:
    """TF-IDF based embedding function (no external dependencies)"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=512,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False
        self.document_vectors = None
        self.documents = []

    def fit(self, documents: List[str]):
        """Fit the vectorizer on documents"""
        if documents:
            self.documents = documents
            self.document_vectors = self.vectorizer.fit_transform(documents)
            self.is_fitted = True
            logger.info(f"✓ TF-IDF vectorizer fitted on {len(documents)} documents")

    def query(self, query_text: str, top_k: int = 3) -> List[int]:
        """Find most similar documents to query"""
        if not self.is_fitted:
            return []

        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices.tolist()

    def get_similarity_scores(self, query_text: str) -> np.ndarray:
        """Get similarity scores for all documents"""
        if not self.is_fitted:
            return np.array([])

        query_vector = self.vectorizer.transform([query_text])
        return cosine_similarity(query_vector, self.document_vectors)[0]


class FraudPolicyRAG:
    """RAG system for retrieving fraud policy context (TF-IDF based)"""

    def __init__(
        self,
        collection_name: str = "fraud_policies",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize RAG system

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist vector database
        """
        self.collection_name = collection_name

        if persist_directory is None:
            project_root = Path(__file__).parent.parent.parent
            persist_directory = str(project_root / "data" / "vector_db")

        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing RAG system: {collection_name}")
        logger.info(f"Vector DB directory: {persist_directory}")

        # Document storage
        self.documents = []
        self.metadatas = []
        self.ids = []

        # Embedding function
        self.embedding_fn = SimpleEmbeddingFunction()

        # Try to load existing data
        self._load_from_disk()

        logger.info(f"✓ RAG system initialized (TF-IDF based, no onnxruntime)")

    def _get_storage_path(self) -> Path:
        """Get path for persistence"""
        return Path(self.persist_directory) / f"{self.collection_name}.pkl"

    def _save_to_disk(self):
        """Save data to disk"""
        storage_path = self._get_storage_path()
        data = {
            'documents': self.documents,
            'metadatas': self.metadatas,
            'ids': self.ids,
            'vectorizer': self.embedding_fn.vectorizer if self.embedding_fn.is_fitted else None,
            'document_vectors': self.embedding_fn.document_vectors if self.embedding_fn.is_fitted else None
        }
        with open(storage_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✓ Saved {len(self.documents)} documents to {storage_path}")

    def _load_from_disk(self):
        """Load data from disk if exists"""
        storage_path = self._get_storage_path()
        if storage_path.exists():
            try:
                with open(storage_path, 'rb') as f:
                    data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.metadatas = data.get('metadatas', [])
                self.ids = data.get('ids', [])

                if data.get('vectorizer') is not None:
                    self.embedding_fn.vectorizer = data['vectorizer']
                    self.embedding_fn.document_vectors = data['document_vectors']
                    self.embedding_fn.documents = self.documents
                    self.embedding_fn.is_fitted = True

                logger.info(f"✓ Loaded {len(self.documents)} documents from disk")
            except Exception as e:
                logger.warning(f"Could not load from disk: {e}")

    def count(self) -> int:
        """Return document count"""
        return len(self.documents)

    def load_policy_documents(self, policy_dir: str):
        """
        Load fraud policy documents from directory

        Args:
            policy_dir: Directory containing policy text files
        """
        policy_path = Path(policy_dir)

        if not policy_path.exists():
            logger.warning(f"Policy directory not found: {policy_dir}")
            logger.info("Creating sample policy document...")
            self._create_sample_policies(policy_path)

        logger.info(f"Loading policies from {policy_dir}...")

        # Read all .txt files
        new_documents = []
        new_metadatas = []
        new_ids = []

        for i, file_path in enumerate(policy_path.glob("*.txt")):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if content:
                # Split into chunks (simple paragraph splitting)
                chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]

                for j, chunk in enumerate(chunks):
                    if len(chunk) > 50:  # Skip very short chunks
                        doc_id = f"{file_path.stem}_{j}"

                        # Skip if already exists
                        if doc_id in self.ids:
                            continue

                        new_documents.append(chunk)
                        new_metadatas.append({
                            'source': file_path.name,
                            'chunk_id': j
                        })
                        new_ids.append(doc_id)

        if new_documents:
            # Add to storage
            self.documents.extend(new_documents)
            self.metadatas.extend(new_metadatas)
            self.ids.extend(new_ids)

            # Re-fit embedding function
            self.embedding_fn.fit(self.documents)

            # Save to disk
            self._save_to_disk()

            logger.info(f"✓ Loaded {len(new_documents)} new policy chunks")
        else:
            logger.info("No new policy documents to load")

        logger.info(f"✓ Total documents in collection: {len(self.documents)}")

    def _create_sample_policies(self, policy_path: Path):
        """Create sample fraud policy documents"""
        policy_path.mkdir(parents=True, exist_ok=True)

        # Sample fraud detection policy
        fraud_policy = """
# FRAUD DETECTION POLICY

## 1. HIGH-RISK TRANSACTION INDICATORS

### Amount-Based Risks
- Transactions over $500 require enhanced monitoring
- Transactions over $1000 require immediate review
- Multiple transactions in short time periods (velocity)
- Round-number amounts (e.g., $1000.00, $5000.00)

### Time-Based Risks
- Transactions outside normal business hours (9 AM - 9 PM)
- Multiple transactions within 5 minutes
- Transactions at unusual times (2-5 AM)

### Pattern-Based Risks
- Abnormal location changes (different cities within hours)
- First-time merchant category usage
- Deviation from normal spending patterns
- Multiple failed attempts followed by success

## 2. FEATURE-SPECIFIC RULES

### V14 Feature (PCA Component)
- Values below -15 or above +15 are highly suspicious
- Correlates with unusual transaction patterns
- Common in card-not-present fraud

### V17 Feature (PCA Component)
- Negative values below -20 indicate potential fraud
- Often associated with stolen card usage

### V12 Feature (PCA Component)
- Extreme values (|V12| > 10) require investigation

## 3. RESPONSE PROCEDURES

### Fraud Score: 0.3 - 0.5 (Medium Risk)
- Flag for manual review within 24 hours
- Send notification to cardholder
- Monitor account for similar patterns

### Fraud Score: 0.5 - 0.7 (High Risk)
- Immediate manual review required
- Temporary hold on transaction
- Contact cardholder for verification

### Fraud Score: 0.7 - 1.0 (Critical Risk)
- Automatic transaction block
- Immediate cardholder notification
- Escalate to fraud investigation team
- Review last 7 days of transactions

## 4. COMPLIANCE REQUIREMENTS

### Data Privacy
- All flagged transactions must maintain PII encryption
- Audit trail required for all fraud decisions
- GDPR/KVKK compliance mandatory

### Reporting
- Daily summary of flagged transactions
- Weekly fraud trend analysis
- Monthly false positive rate review

## 5. MODEL PERFORMANCE THRESHOLDS

### Acceptable Ranges
- Precision: minimum 80%
- Recall: minimum 75%
- False Positive Rate: maximum 5%
- Response Time: maximum 300ms per transaction

### Model Monitoring
- Daily performance metrics review
- Weekly model drift detection
- Monthly model retraining evaluation
"""

        policy_file = policy_path / "fraud_policy.txt"
        with open(policy_file, 'w', encoding='utf-8') as f:
            f.write(fraud_policy)

        logger.info(f"✓ Created sample policy: {policy_file}")

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieve relevant policy context

        Args:
            query: Query string (e.g., feature names, risk factors)
            top_k: Number of top results to return

        Returns:
            List of relevant policy chunks
        """
        if len(self.documents) == 0:
            logger.warning("No policies in database. Loading defaults...")
            project_root = Path(__file__).parent.parent.parent
            self.load_policy_documents(str(project_root / "data" / "policies"))

        logger.info(f"Retrieving context for query: '{query}'")

        # Get top-k similar documents
        top_indices = self.embedding_fn.query(query, top_k)
        scores = self.embedding_fn.get_similarity_scores(query)

        context_chunks = []

        for idx in top_indices:
            if idx < len(self.documents):
                context_chunks.append({
                    'text': self.documents[idx],
                    'source': self.metadatas[idx]['source'],
                    'relevance_score': float(scores[idx])
                })

        logger.info(f"✓ Retrieved {len(context_chunks)} relevant policy chunks")

        return context_chunks

    def format_context_for_llm(
        self,
        context_chunks: List[Dict]
    ) -> str:
        """
        Format retrieved context for LLM prompt

        Args:
            context_chunks: List of context dictionaries

        Returns:
            Formatted context string
        """
        if not context_chunks:
            return ""

        formatted = "FRAUD DETECTION POLICY REFERENCES:\n\n"

        for i, chunk in enumerate(context_chunks, 1):
            formatted += f"[{i}] Source: {chunk['source']} (Relevance: {chunk['relevance_score']:.2f})\n"
            formatted += f"{chunk['text']}\n\n"

        return formatted.strip()

    def get_policy_summary(self) -> Dict:
        """Get summary of loaded policies"""
        return {
            'collection_name': self.collection_name,
            'total_chunks': len(self.documents),
            'persist_directory': self.persist_directory,
            'embedding_method': 'TF-IDF'
        }


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent

    logger.info("="*70)
    logger.info("RAG System Demo (TF-IDF based - No onnxruntime)")
    logger.info("="*70)

    # Initialize RAG
    rag = FraudPolicyRAG()

    # Load policies
    policy_dir = project_root / "data" / "policies"
    rag.load_policy_documents(str(policy_dir))

    # Get summary
    summary = rag.get_policy_summary()
    logger.info(f"\nRAG Summary:")
    logger.info(f"  Collection: {summary['collection_name']}")
    logger.info(f"  Total Chunks: {summary['total_chunks']}")
    logger.info(f"  Embedding: {summary['embedding_method']}")
    logger.info(f"  Storage: {summary['persist_directory']}")

    # Example queries
    queries = [
        "V14 feature high value",
        "high transaction amount over $1000",
        "unusual time 3 AM transaction"
    ]

    for query in queries:
        logger.info(f"\n{'='*70}")
        logger.info(f"Query: '{query}'")
        logger.info("="*70)

        # Retrieve context
        context = rag.retrieve_context(query, top_k=2)

        # Format for LLM
        formatted_context = rag.format_context_for_llm(context)

        print(f"\n{formatted_context}\n")

    logger.info("\n✓ RAG system demo completed!")
