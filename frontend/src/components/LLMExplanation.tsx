'use client';

import { Bot, FileText } from 'lucide-react';

interface LLMExplanationProps {
  explanation: string | null;
  policyContext: string | null;
  fraudProbability?: number;
  isFraud?: boolean;
}

export default function LLMExplanation({ explanation, policyContext, fraudProbability, isFraud }: LLMExplanationProps) {
  // Generate fallback Turkish explanation if LLM is unavailable
  const getFallbackExplanation = () => {
    if (fraudProbability === undefined) return null;

    const score = fraudProbability;
    if (isFraud) {
      let message = `⚠️ Bu işlem riskli olarak değerlendirildi (risk skoru: ${(score * 100).toFixed(0)}%). `;
      message += "Sistemimiz bu işlemde şüpheli aktivite tespit etti. ";

      if (score > 0.8) {
        message += "İşleminizi onaylamadan önce lütfen bizimle iletişime geçin.";
      } else if (score > 0.6) {
        message += "Eğer bu işlemi siz yapmadıysanız, kartınızı bloke etmenizi öneririz.";
      } else {
        message += "İşleminiz inceleme altında, sonuç SMS ile bildirilecektir.";
      }
      return message;
    } else {
      let message = `✓ Bu işlem güvenli bulundu (güvenlik skoru: ${((1 - score) * 100).toFixed(0)}%). `;
      message += "Sistemimiz işleminizde herhangi bir anormallik tespit etmedi. ";
      message += "İşlem karakteristikleri normal kullanım paternlerinize uygun.";
      return message;
    }
  };

  const displayExplanation = explanation || getFallbackExplanation();

  if (!displayExplanation && !policyContext) {
    return (
      <div className="card">
        <h2 className="text-lg font-semibold mb-4">AI Açıklama</h2>
        <p className="text-gray-500 text-center py-8">
          İşlem analizi için lütfen bir tahmin yapın
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

      {displayExplanation && (
        <div className="mb-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-gray-700 whitespace-pre-line">{displayExplanation}</p>
            {!explanation && (
              <p className="text-xs text-gray-500 mt-2 italic">
                * AI analizi şu anda kullanılamıyor, otomatik açıklama gösteriliyor
              </p>
            )}
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
