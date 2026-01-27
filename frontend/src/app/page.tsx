'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/Header';
import TransactionForm from '@/components/TransactionForm';
import PredictionResult from '@/components/PredictionResult';
import RiskFactors from '@/components/RiskFactors';
import LLMExplanation from '@/components/LLMExplanation';
import StatsCards from '@/components/StatsCards';
import {
  checkHealth,
  explainFraud,
  Transaction,
  Explanation,
  Prediction,
  RiskFactor
} from '@/lib/api';

export default function Home() {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [riskFactors, setRiskFactors] = useState<RiskFactor[] | null>(null);
  const [llmExplanation, setLlmExplanation] = useState<string | null>(null);
  const [policyContext, setPolicyContext] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [stats, setStats] = useState({
    totalAnalyzed: 0,
    fraudDetected: 0,
    legitTransactions: 0,
    avgLatency: 0
  });

  const [latencies, setLatencies] = useState<number[]>([]);

  // Check API health on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await checkHealth();
        setIsConnected(true);
      } catch {
        setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (transaction: Transaction) => {
    setIsLoading(true);
    setError(null);

    const startTime = Date.now();

    try {
      const result = await explainFraud(transaction, false, false);
      const latency = Date.now() - startTime;

      // Update state
      setPrediction(result.prediction);
      setRiskFactors(result.top_risk_factors);
      setLlmExplanation(result.llm_explanation);
      setPolicyContext(result.policy_context);

      // Update stats
      const newLatencies = [...latencies, latency].slice(-100);
      setLatencies(newLatencies);

      setStats(prev => ({
        totalAnalyzed: prev.totalAnalyzed + 1,
        fraudDetected: prev.fraudDetected + (result.prediction.is_fraud ? 1 : 0),
        legitTransactions: prev.legitTransactions + (result.prediction.is_fraud ? 0 : 1),
        avgLatency: newLatencies.reduce((a, b) => a + b, 0) / newLatencies.length
      }));

    } catch (err: any) {
      setError(err.message || 'Failed to analyze transaction');
      console.error('Analysis error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Header isConnected={isConnected} />

      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        {/* Error Banner */}
        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        {/* Stats */}
        <div className="mb-6">
          <StatsCards stats={stats} />
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Form */}
          <div className="lg:col-span-1">
            <TransactionForm onSubmit={handleSubmit} isLoading={isLoading} />
          </div>

          {/* Middle Column - Results */}
          <div className="lg:col-span-1 space-y-6">
            <PredictionResult prediction={prediction} />
            <RiskFactors factors={riskFactors} />
          </div>

          {/* Right Column - Explanation */}
          <div className="lg:col-span-1">
            <LLMExplanation
              explanation={llmExplanation}
              policyContext={policyContext}
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-sm text-gray-400">
          <p>AI-Powered Fraud Detection System | ParamTECH</p>
          <p>Built with FastAPI, XGBoost, PyTorch, SHAP, and Hugging Face</p>
        </footer>
      </main>
    </div>
  );
}
