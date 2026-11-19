import React from "react";

interface RiskGaugeProps {
  probability: number; // 0.0 to 1.0
  threshold: number;   // e.g. 0.12
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({ probability, threshold }) => {
  const percent = Math.min(Math.max(probability * 100, 0), 100);
  const thresholdPercent = Math.min(Math.max(threshold * 100, 0), 100);

  // SVG Gauge calculations (semi-circle 180 degrees)
  const radius = 80;
  const circumference = Math.PI * radius;
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  let strokeColor = "#10b981"; // green
  if (probability >= 0.20) {
    strokeColor = "#ef4444"; // red
  } else if (probability >= 0.10) {
    strokeColor = "#f59e0b"; // amber
  }

  return (
    <div className="flex flex-col items-center justify-center py-2">
      <div className="relative w-48 h-28 flex items-end justify-center">
        <svg className="w-48 h-28 overflow-visible" viewBox="0 0 200 110">
          {/* Background Arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#e2e8f0"
            strokeWidth="16"
            strokeLinecap="round"
          />
          {/* Value Arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={strokeColor}
            strokeWidth="16"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-1000 ease-out"
          />
        </svg>

        {/* Center Display */}
        <div className="absolute bottom-0 flex flex-col items-center">
          <span className="text-3xl font-extrabold tracking-tight text-slate-900">
            {percent.toFixed(1)}%
          </span>
          <span className="text-xs uppercase font-semibold text-slate-500 tracking-wider">
            10-Yr CHD Risk
          </span>
        </div>
      </div>

      {/* Threshold Reference Indicator */}
      <div className="mt-3 flex items-center space-x-2 text-xs text-slate-500 bg-slate-100 px-2.5 py-1 rounded-full">
        <span className="inline-block w-2 h-2 rounded-full bg-indigo-500"></span>
        <span>
          Screening Threshold: <strong>{thresholdPercent.toFixed(1)}%</strong>
        </span>
      </div>
    </div>
  );
};

