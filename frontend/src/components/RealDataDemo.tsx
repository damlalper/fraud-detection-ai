'use client';

import { useState, useEffect } from 'react';
import { getRealDataDemo, DemoData, DemoResult } from '@/lib/api';

export default function RealDataDemo() {
  const [demoData, setDemoData] = useState<DemoData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDemo();
  }, []);

  const loadDemo = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Try API first
      const data = await getRealDataDemo(10, true);
      setDemoData(data);
    } catch (err: any) {
      console.log('API failed, trying static file...', err);

      // Fallback to static JSON file
      try {
        const response = await fetch('/demo_fraud_results.json');
        const fileData = await response.json();

        // Transform to match API format
        const transformedData: DemoData = {
          dataset: fileData.dataset,
          total_transactions: fileData.total_transactions,
          total_fraud: fileData.fraud_count,
          total_legitimate: fileData.total_transactions - fileData.fraud_count,
          results: fileData.results.map((r: any) => ({
            index: r.index,
            actual_label: r.actual,
            predicted_label: r.predicted,
            fraud_probability: r.probability,
            is_correct: r.correct,
            amount: r.amount,
            time: 0,
            features: {}
          })),
          model_info: {
            type: 'xgboost',
            threshold: 0.997,
            num_features: 43
          }
        };

        setDemoData(transformedData);
      } catch (fileErr: any) {
        setError(fileErr.message || 'Failed to load demo data');
        console.error('File load error:', fileErr);
      }
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-white p-8 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">
          Real Kaggle Fraud Data Demo
        </h2>
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading real fraud transactions...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white p-8 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">
          Real Kaggle Fraud Data Demo
        </h2>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          {error}
        </div>
        <button
          onClick={loadDemo}
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!demoData) return null;

  const accuracy = demoData.results.filter(r => r.is_correct).length;
  const total = demoData.results.length;

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">
          Real Kaggle Fraud Data Demo
        </h2>
        <button
          onClick={loadDemo}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition text-sm"
        >
          Refresh
        </button>
      </div>

      {/* Dataset Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">Total Transactions</div>
          <div className="text-2xl font-bold text-blue-600">
            {demoData.total_transactions.toLocaleString()}
          </div>
        </div>

        <div className="bg-gradient-to-br from-red-50 to-red-100 p-4 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">Fraud Cases</div>
          <div className="text-2xl font-bold text-red-600">
            {demoData.total_fraud.toLocaleString()}
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">Model AUC-ROC</div>
          <div className="text-2xl font-bold text-green-600">97.13%</div>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">Threshold</div>
          <div className="text-2xl font-bold text-purple-600">
            {demoData.model_info.threshold.toFixed(4)}
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-700">
            Fraud Detection Results
          </h3>
          <div className="text-sm text-gray-600">
            Accuracy: {accuracy}/{total} ({((accuracy / total) * 100).toFixed(1)}%)
          </div>
        </div>

        {demoData.results.map((result, index) => (
          <DemoResultCard key={index} result={result} />
        ))}
      </div>
    </div>
  );
}

function DemoResultCard({ result }: { result: DemoResult }) {
  const probPercent = (result.fraud_probability * 100).toFixed(2);
  const isActualFraud = result.actual_label === 'FRAUD';
  const isPredictedFraud = result.predicted_label === 'FRAUD';
  const isCorrect = result.is_correct;

  return (
    <div className={`border-2 ${isCorrect ? 'border-green-200' : 'border-red-200'} rounded-lg p-4 hover:shadow-md transition`}>
      {/* Header */}
      <div className="flex justify-between items-center mb-3">
        <div className="font-mono text-sm text-gray-600">
          Transaction #{result.index}
        </div>
        <div className="text-lg font-bold text-blue-600">
          ${result.amount.toFixed(2)}
        </div>
      </div>

      {/* Labels */}
      <div className="flex gap-3 mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Actual:</span>
          <span className={`px-3 py-1 rounded-full text-xs font-bold ${isActualFraud ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
            {result.actual_label}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Predicted:</span>
          <span className={`px-3 py-1 rounded-full text-xs font-bold ${isPredictedFraud ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
            {result.predicted_label}
          </span>
        </div>

        {isCorrect && (
          <span className="ml-auto px-3 py-1 rounded-full text-xs font-bold bg-green-100 text-green-700">
            Correct
          </span>
        )}
      </div>

      {/* Probability Bar */}
      <div className="relative">
        <div className="h-8 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 flex items-center justify-end pr-3"
            style={{ width: `${probPercent}%` }}
          >
            <span className="text-white text-xs font-bold">{probPercent}%</span>
          </div>
        </div>
        <div className="mt-1 text-xs text-gray-500 text-center">
          Fraud Probability
        </div>
      </div>
    </div>
  );
}
