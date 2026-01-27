'use client';

import { Activity, AlertTriangle, CheckCircle, Clock } from 'lucide-react';

interface Stats {
  totalAnalyzed: number;
  fraudDetected: number;
  legitTransactions: number;
  avgLatency: number;
}

interface StatsCardsProps {
  stats: Stats;
}

export default function StatsCards({ stats }: StatsCardsProps) {
  const cards = [
    {
      title: 'Total Analyzed',
      value: stats.totalAnalyzed,
      icon: Activity,
      color: 'blue',
      bg: 'bg-blue-50',
      text: 'text-blue-600'
    },
    {
      title: 'Fraud Detected',
      value: stats.fraudDetected,
      icon: AlertTriangle,
      color: 'red',
      bg: 'bg-red-50',
      text: 'text-red-600'
    },
    {
      title: 'Legitimate',
      value: stats.legitTransactions,
      icon: CheckCircle,
      color: 'green',
      bg: 'bg-green-50',
      text: 'text-green-600'
    },
    {
      title: 'Avg Latency',
      value: `${stats.avgLatency.toFixed(0)}ms`,
      icon: Clock,
      color: 'purple',
      bg: 'bg-purple-50',
      text: 'text-purple-600'
    }
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => (
        <div key={card.title} className="card">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${card.bg}`}>
              <card.icon className={`h-5 w-5 ${card.text}`} />
            </div>
            <div>
              <p className="text-sm text-gray-500">{card.title}</p>
              <p className={`text-xl font-bold ${card.text}`}>{card.value}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
