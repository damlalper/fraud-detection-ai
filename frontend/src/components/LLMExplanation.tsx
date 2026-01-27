'use client';

import { Bot, FileText } from 'lucide-react';

interface LLMExplanationProps {
  explanation: string | null;
  policyContext: string | null;
}

export default function LLMExplanation({ explanation, policyContext }: LLMExplanationProps) {
  if (!explanation && !policyContext) {
    return (
      <div className="card">
        <h2 className="text-lg font-semibold mb-4">AI Explanation</h2>
        <p className="text-gray-500 text-center py-8">
          Enable LLM explanation to see AI-generated analysis
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Bot className="h-5 w-5 text-blue-600" />
        AI Explanation
      </h2>

      {explanation && (
        <div className="mb-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-gray-700 whitespace-pre-line">{explanation}</p>
          </div>
        </div>
      )}

      {policyContext && (
        <div>
          <h3 className="text-sm font-medium text-gray-500 mb-2 flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Policy Reference (RAG)
          </h3>
          <div className="bg-gray-50 border rounded-lg p-4 text-sm text-gray-600 max-h-40 overflow-y-auto">
            <pre className="whitespace-pre-wrap font-sans">{policyContext}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
