'use client';

import { Shield, Activity, AlertTriangle } from 'lucide-react';

interface HeaderProps {
  isConnected: boolean;
}

export default function Header({ isConnected }: HeaderProps) {
  return (
    <header className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-xl font-bold text-gray-900">
                AI Fraud Detection
              </h1>
              <p className="text-sm text-gray-500">
                Real-time fraud monitoring dashboard
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
              isConnected
                ? 'bg-green-100 text-green-700'
                : 'bg-red-100 text-red-700'
            }`}>
              <Activity className="h-4 w-4" />
              {isConnected ? 'API Connected' : 'API Disconnected'}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
