import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Transaction {
  transaction_id?: string;
  features: Record<string, number>;
}

export interface Prediction {
  transaction_id: string;
  fraud_probability: number;
  is_fraud: boolean;
  confidence: number;
  threshold: number;
  model_type: string;
  timestamp: string;
}

export interface RiskFactor {
  feature: string;
  value: number;
  shap_value: number;
  impact: string;
  magnitude: number;
}

export interface Explanation {
  transaction_id: string;
  prediction: Prediction;
  top_risk_factors: RiskFactor[];
  llm_explanation: string | null;
  policy_context: string | null;
  timestamp: string;
}

export interface BatchPrediction {
  predictions: Prediction[];
  total_processed: number;
  fraud_count: number;
  legitimate_count: number;
  processing_time_ms: number;
  timestamp: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  model_loaded: boolean;
  timestamp: string;
}

export interface ModelInfo {
  model_type: string;
  num_features: number;
  threshold: number;
}

export interface DemoResult {
  index: number;
  actual_label: string;
  predicted_label: string;
  fraud_probability: number;
  is_correct: boolean;
  amount: number;
  time: number;
  features: Record<string, number>;
}

export interface DemoData {
  dataset: string;
  total_transactions: number;
  total_fraud: number;
  total_legitimate: number;
  results: DemoResult[];
  model_info: {
    type: string;
    threshold: number;
    num_features: number;
  };
}

// API Functions
export const checkHealth = async (): Promise<HealthStatus> => {
  const response = await api.get('/health');
  return response.data;
};

export const getModelInfo = async (): Promise<ModelInfo> => {
  const response = await api.get('/model/info');
  return response.data;
};

export const predictFraud = async (transaction: Transaction): Promise<Prediction> => {
  const response = await api.post('/predict', transaction);
  return response.data;
};

export const explainFraud = async (
  transaction: Transaction,
  includeLlm: boolean = false,
  includeRag: boolean = false
): Promise<Explanation> => {
  const response = await api.post('/explain', transaction, {
    params: { include_llm: includeLlm, include_rag: includeRag }
  });
  return response.data;
};

export const batchPredict = async (transactions: Transaction[]): Promise<BatchPrediction> => {
  const response = await api.post('/batch/predict', { transactions });
  return response.data;
};

export const getRealDataDemo = async (limit: number = 10, fraudOnly: boolean = true): Promise<DemoData> => {
  const response = await api.get('/demo/realdata', {
    params: { limit, fraud_only: fraudOnly }
  });
  return response.data;
};

export default api;
