'use client';

import { ArrowUp, ArrowDown } from 'lucide-react';
import { RiskFactor } from '@/lib/api';

interface RiskFactorsProps {
  factors: RiskFactor[] | null;
}

export default function RiskFactors({ factors }: RiskFactorsProps) {
  if (!factors || factors.length === 0) {
    return (
      <div className="card">
        <h2 className="text-lg font-semibold mb-4">Risk Factors (SHAP)</h2>
        <p className="text-gray-500 text-center py-8">
          Submit a transaction to see risk factors
        </p>
      </div>
    );
  }

  const maxMagnitude = Math.max(...factors.map(f => f.magnitude));

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Risk Factors (SHAP Analysis)</h2>

      <div className="space-y-3">
        {factors.slice(0, 10).map((factor, index) => (
          <div key={index} className="flex items-center gap-3">
            <div className="flex-shrink-0 w-6 text-center text-sm font-medium text-gray-400">
              {index + 1}
            </div>

            <div className="flex-grow">
              <div className="flex items-center justify-between mb-1">
                <span className="font-medium text-sm">{factor.feature}</span>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">
                    {factor.value.toFixed(3)}
                  </span>
                  <span className={`flex items-center text-xs font-medium ${
                    factor.shap_value > 0 ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {factor.shap_value > 0 ? (
                      <ArrowUp className="h-3 w-3" />
                    ) : (
                      <ArrowDown className="h-3 w-3" />
                    )}
                    {factor.shap_value.toFixed(4)}
                  </span>
                </div>
              </div>

              <div className="flex gap-1">
                {/* Negative bar (green - decreases risk) */}
                <div className="flex-1 flex justify-end">
                  {factor.shap_value < 0 && (
                    <div
                      className="h-2 bg-green-500 rounded-l"
                      style={{
                        width: `${(Math.abs(factor.shap_value) / maxMagnitude) * 100}%`
                      }}
                    />
                  )}
                </div>

                {/* Center line */}
                <div className="w-px bg-gray-300" />

                {/* Positive bar (red - increases risk) */}
                <div className="flex-1">
                  {factor.shap_value > 0 && (
                    <div
                      className="h-2 bg-red-500 rounded-r"
                      style={{
                        width: `${(factor.shap_value / maxMagnitude) * 100}%`
                      }}
                    />
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 pt-4 border-t flex justify-between text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <div className="w-3 h-3 bg-green-500 rounded" />
          Decreases fraud risk
        </span>
        <span className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-500 rounded" />
          Increases fraud risk
        </span>
      </div>
    </div>
  );
}
