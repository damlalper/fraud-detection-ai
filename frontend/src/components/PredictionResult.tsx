'use client';

import { AlertTriangle, CheckCircle, TrendingUp, Clock } from 'lucide-react';
import { Prediction } from '@/lib/api';

interface PredictionResultProps {
  prediction: Prediction | null;
}

export default function PredictionResult({ prediction }: PredictionResultProps) {
  if (!prediction) {
    return (
      <div className="card">
        <h2 className="text-lg font-semibold mb-4">Prediction Result</h2>
        <p className="text-gray-500 text-center py-8">
          Submit a transaction to see the prediction result
        </p>
      </div>
    );
  }

  const getFraudLevel = (prob: number) => {
    if (prob < 0.3) return { level: 'Low', color: 'green', badge: 'fraud-badge-low' };
    if (prob < 0.5) return { level: 'Medium', color: 'yellow', badge: 'fraud-badge-medium' };
    if (prob < 0.7) return { level: 'High', color: 'orange', badge: 'fraud-badge-high' };
    return { level: 'Critical', color: 'red', badge: 'fraud-badge-critical' };
  };

  const { level, badge } = getFraudLevel(prediction.fraud_probability);

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Prediction Result</h2>

      {/* Main Result */}
      <div className={`p-4 rounded-lg mb-4 ${
        prediction.is_fraud ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'
      }`}>
        <div className="flex items-center gap-3">
          {prediction.is_fraud ? (
            <AlertTriangle className="h-8 w-8 text-red-500" />
          ) : (
            <CheckCircle className="h-8 w-8 text-green-500" />
          )}
          <div>
            <h3 className={`text-xl font-bold ${
              prediction.is_fraud ? 'text-red-700' : 'text-green-700'
            }`}>
              {prediction.is_fraud ? 'FRAUD DETECTED' : 'LEGITIMATE'}
            </h3>
            <p className="text-sm text-gray-600">
              Transaction ID: {prediction.transaction_id}
            </p>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <TrendingUp className="h-4 w-4" />
            Fraud Probability
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold">
              {(prediction.fraud_probability * 100).toFixed(1)}%
            </span>
            <span className={`fraud-badge ${badge}`}>{level}</span>
          </div>
        </div>

        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <CheckCircle className="h-4 w-4" />
            Confidence
          </div>
          <span className="text-2xl font-bold">
            {(prediction.confidence * 100).toFixed(1)}%
          </span>
        </div>

        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-500 mb-1">Model</div>
          <span className="font-medium uppercase">{prediction.model_type}</span>
        </div>

        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-500 mb-1">Threshold</div>
          <span className="font-medium">{prediction.threshold}</span>
        </div>
      </div>

      {/* Probability Bar */}
      <div className="mt-4">
        <div className="flex justify-between text-sm text-gray-500 mb-1">
          <span>Legitimate</span>
          <span>Fraud</span>
        </div>
        <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${
              prediction.fraud_probability < 0.3
                ? 'bg-green-500'
                : prediction.fraud_probability < 0.5
                ? 'bg-yellow-500'
                : prediction.fraud_probability < 0.7
                ? 'bg-orange-500'
                : 'bg-red-500'
            }`}
            style={{ width: `${prediction.fraud_probability * 100}%` }}
          />
        </div>
      </div>

      <div className="mt-4 flex items-center gap-1 text-xs text-gray-400">
        <Clock className="h-3 w-3" />
        {new Date(prediction.timestamp).toLocaleString()}
      </div>
    </div>
  );
}
