import React from "react";
import { PredictionResponse } from "@/types/prediction";
import { RiskGauge } from "./RiskGauge";

interface ResultCardProps {
  result: PredictionResponse;
  onReset: () => void;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, onReset }) => {
  const getBadgeStyle = () => {
    switch (result.risk_level) {
      case "high":
        return "bg-red-100 text-red-800 border-red-200";
      case "moderate":
        return "bg-amber-100 text-amber-800 border-amber-200";
      case "low":
      default:
        return "bg-emerald-100 text-emerald-800 border-emerald-200";
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 sm:p-8 space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <span className="text-xs uppercase tracking-wider font-semibold text-slate-400">
            Assessment Outcome
          </span>
          <h2 className="text-xl font-bold text-slate-900">
            10-Year Coronary Heart Disease Risk
          </h2>
        </div>
        <span
          className={`px-3 py-1 text-xs font-semibold rounded-full border ${getBadgeStyle()}`}
        >
          {result.risk_label}
        </span>
      </div>

      {/* Gauge Visualization */}
      <RiskGauge probability={result.probability} threshold={result.threshold} />

      {/* Clinical Interpretation */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-2">
        <h4 className="text-xs uppercase tracking-wider font-semibold text-slate-500">
          Clinical Interpretation
        </h4>
        <p className="text-sm text-slate-700 leading-relaxed">
          {result.explanation}
        </p>
        <p className="text-xs text-slate-500">
          Model screening threshold is set at{" "}
          <strong>{(result.threshold * 100).toFixed(1)}%</strong> to prioritize sensitivity
          in cardiovascular risk detection and minimize missed high-risk individuals.
        </p>
      </div>

      {/* Action CTA */}
      <div className="pt-2 flex justify-end">
        <button
          onClick={onReset}
          className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium text-sm rounded-lg transition-colors"
        >
          Calculate Another Assessment
        </button>
      </div>
    </div>
  );
};

