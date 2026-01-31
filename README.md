# AI-Powered Fraud Detection & Explanation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A scalable FinTech fraud detection platform that identifies suspicious financial transactions in real-time and explains decisions using Large Language Models (LLMs). Trained on real Kaggle Credit Card Fraud dataset with **97.13% AUC-ROC** accuracy.

## 🎯 Features

- **ML-based Fraud Detection**: XGBoost model with **97.13% AUC-ROC** on Kaggle dataset (284K+ transactions)
- **LLM-based Explanation Engine**: Turkish language explanations using Mistral-7B via Hugging Face
- **RAG Policy Reference**: TF-IDF based retrieval for fraud policy context (no onnxruntime dependency)
- **Real-time API**: Production-ready FastAPI integration with ~200ms latency
- **MLOps Infrastructure**: MLflow experiment tracking, model registry, and drift detection
- **Explainable AI**: SHAP TreeExplainer for regulatory compliance
- **Modern Frontend**: Next.js 14 + Tailwind CSS with real-time fraud detection demo
- **AWS Deployment**: CloudFormation templates for ECS Fargate deployment
- **Security Compliance**: GDPR, KVKK, PCI-DSS documentation

## 📊 Performance Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| ROC-AUC | > 0.85 | **0.9713** | ✅ |
| Dataset Size | - | 284,807 txns | ✅ |
| Fraud Cases | - | 492 cases | ✅ |
| Threshold | - | 0.9969 | ✅ |
| Inference Latency | < 300ms | ~200ms | ✅ |
| Model Type | - | XGBoost | ✅ |

## 🏗️ Architecture

```
┌─────────────────┐
│   Dashboard UI  │
└────────┬────────┘
         │
┌────────▼────────────────────────────────────┐
│           API Gateway (FastAPI)             │
└────┬─────────────────┬──────────────────────┘
     │                 │
┌────▼─────────┐  ┌───▼──────────────────┐
│ Fraud Model  │  │ LLM Explanation      │
│   Service    │  │ Service (+ RAG)      │
│ (XGBoost/NN) │  │                      │
└──────────────┘  └──────────────────────┘
```

## 📁 Project Structure

```
fintech-ai-freud/
├── data/                      # Data storage
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed datasets
│   └── policies/              # Fraud policy documents for RAG
├── notebooks/                 # Jupyter notebooks for EDA
├── src/                       # Source code
│   ├── data/                  # Data pipeline & ETL
│   ├── models/                # ML model implementations
│   │   ├── classical/         # XGBoost, LightGBM
│   │   └── deep_learning/     # PyTorch models
│   ├── explainability/        # XAI & SHAP integration
│   ├── llm/                   # LLM explanation service
│   ├── rag/                   # RAG implementation
│   ├── api/                   # FastAPI backend
│   └── utils/                 # Utilities
├── frontend/                  # TypeScript React dashboard
├── tests/                     # Unit and integration tests
├── docker/                    # Docker configurations
├── scripts/                   # Automation scripts
├── models/                    # Trained model artifacts
├── logs/                      # Application logs
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- AWS CLI (for deployment)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/damlalper/fraud-detection-ai.git
cd fraud-detection-ai
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Download datasets**
```bash
python scripts/download_data.py
```

4. **Train models**
```bash
python src/models/train_classical.py
python src/models/train_deep_learning.py
```

5. **Start API server**
```bash
uvicorn src.api.main:app --reload
```

6. **Start frontend dashboard**
```bash
cd frontend
npm install
npm run dev
```

## 🔧 Configuration

Create a [.env](.env) file in the root directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4

# Database
VECTOR_DB_URL=your_vector_db_url

# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

## 📚 API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

- `POST /api/v1/predict` - Single transaction fraud prediction
- `POST /api/v1/batch-predict` - Batch inference
- `GET /api/v1/explain/{transaction_id}` - Get fraud explanation
- `GET /api/v1/health` - Health check

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Load testing
locust -f tests/load/locustfile.py
```

## 🚢 Deployment

### Docker

```bash
docker-compose up -d
```

### AWS Deployment

```bash
# Build and push Docker images
./scripts/deploy.sh

# Deploy to EC2
terraform apply
```

## 📈 Monitoring

- **Metrics**: Prometheus + Grafana
- **Logging**: CloudWatch / ELK Stack
- **Model Monitoring**: MLflow

Access Grafana dashboard at http://localhost:3000

## 🛡️ Security & Compliance

- PII encryption at rest and in transit
- JWT-based authentication
- GDPR/KVKK awareness
- Audit logging for all predictions
- Bias monitoring and fairness analysis

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 AI Engineer Position - Technical Skills Showcase

This project demonstrates expertise in all required areas:

### ✅ Qualifications Match
- **Machine Learning & Deep Learning**: XGBoost (97.1% AUC), PyTorch models
- **Large Language Models**: Mistral-7B via Hugging Face Inference API
- **RAG Architecture**: TF-IDF based retrieval system for fraud policies
- **Python Proficiency**: Full-stack Python application with FastAPI
- **AI Frameworks**: TensorFlow, PyTorch, scikit-learn
- **MLOps Experience**: MLflow tracking, model registry, drift detection
- **APIs & Integration**: RESTful FastAPI with Swagger documentation
- **Problem Solving**: End-to-end fraud detection pipeline

### ✅ Job Responsibilities Covered
- **AI/ML Solution Design**: Complete fraud detection architecture
- **LLM Applications**: Turkish language chatbot/explanation system
- **Model Lifecycle**: Training, testing, deployment, monitoring
- **Scalable Architecture**: Docker, AWS CloudFormation deployment
- **Data Engineering Integration**: ETL pipelines, feature engineering
- **Cloud Deployment**: AWS ECS Fargate ready
- **Performance Optimization**: Model tuning, API optimization
- **Security & Compliance**: GDPR, KVKK, PCI-DSS documentation

### 📊 Technical Stack Alignment
| Required | Implemented |
|----------|------------|
| Machine Learning | XGBoost, scikit-learn ✅ |
| NLP | SHAP explanations, LLM integration ✅ |
| Deep Learning | PyTorch models ✅ |
| LLMs | Mistral-7B (Hugging Face) ✅ |
| RAG | TF-IDF retrieval system ✅ |
| Python | FastAPI, Pydantic, async/await ✅ |
| Frameworks | TensorFlow, PyTorch ✅ |
| MLOps | MLflow, model registry ✅ |
| APIs | RESTful FastAPI with docs ✅ |
| Cloud | AWS CloudFormation ✅ |

## 📞 Contact

For questions or feedback, please open an issue or contact [damlanuralper20@gmail.com](mailto:damlanuralper20@gmail.com).

## 🙏 Acknowledgments

- Kaggle Credit Card Fraud Dataset
- IEEE-CIS Fraud Detection Dataset
- Open-source ML community


