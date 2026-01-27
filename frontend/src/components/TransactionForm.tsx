'use client';

import { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { Transaction } from '@/lib/api';

interface TransactionFormProps {
  onSubmit: (transaction: Transaction) => void;
  isLoading: boolean;
}

export default function TransactionForm({ onSubmit, isLoading }: TransactionFormProps) {
  const [amount, setAmount] = useState('');
  const [time, setTime] = useState('');
  const [transactionId, setTransactionId] = useState('');

  // Sample V features (normally from PCA)
  const generateVFeatures = () => {
    const features: Record<string, number> = {};
    for (let i = 1; i <= 28; i++) {
      features[`V${i}`] = (Math.random() - 0.5) * 4;
    }
    return features;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const transaction: Transaction = {
      transaction_id: transactionId || `TXN_${Date.now()}`,
      features: {
        Amount: parseFloat(amount) || 0,
        Time: parseFloat(time) || 0,
        ...generateVFeatures()
      }
    };

    onSubmit(transaction);
  };

  const loadSampleLegitimate = () => {
    setAmount('45.50');
    setTime('13547');
    setTransactionId('SAMPLE_LEGIT');
  };

  const loadSampleFraud = () => {
    setAmount('1234.56');
    setTime('3547');
    setTransactionId('SAMPLE_FRAUD');
  };

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Analyze Transaction</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Transaction ID (optional)
          </label>
          <input
            type="text"
            value={transactionId}
            onChange={(e) => setTransactionId(e.target.value)}
            placeholder="TXN_12345"
            className="input-field w-full"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Amount ($)
            </label>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="0.00"
              step="0.01"
              className="input-field w-full"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Time (seconds)
            </label>
            <input
              type="number"
              value={time}
              onChange={(e) => setTime(e.target.value)}
              placeholder="0"
              className="input-field w-full"
              required
            />
          </div>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={loadSampleLegitimate}
            className="text-sm text-green-600 hover:text-green-700"
          >
            Load Legitimate Sample
          </button>
          <span className="text-gray-300">|</span>
          <button
            type="button"
            onClick={loadSampleFraud}
            className="text-sm text-red-600 hover:text-red-700"
          >
            Load Fraud Sample
          </button>
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Search className="h-4 w-4" />
              Analyze Transaction
            </>
          )}
        </button>
      </form>
    </div>
  );
}
