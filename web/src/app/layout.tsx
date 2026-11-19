import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "10-Year Coronary Heart Disease Risk Predictor",
  description:
    "Calibrated clinical machine learning model for estimating 10-year coronary heart disease (CHD) risk based on the Framingham cohort.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-slate-50 text-slate-900 antialiased">
        {/* Navigation Header */}
        <header className="border-b border-slate-200 bg-white/80 backdrop-blur sticky top-0 z-40">
          <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
            <Link href="/" className="flex items-center space-x-2.5">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-red-600 text-white font-bold text-lg shadow-sm">
                ♥
              </span>
              <div>
                <span className="text-base font-semibold text-slate-900 block leading-tight">
                  CardioRisk AI
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  10-Year CHD Predictor
                </span>
              </div>
            </Link>

            <nav className="flex items-center space-x-1 sm:space-x-3 text-sm font-medium">
              <Link
                href="/"
                className="px-3 py-1.5 rounded-md text-slate-700 hover:text-slate-900 hover:bg-slate-100 transition-colors"
              >
                Risk Calculator
              </Link>
              <Link
                href="/about-model"
                className="px-3 py-1.5 rounded-md text-slate-700 hover:text-slate-900 hover:bg-slate-100 transition-colors"
              >
                About the Model
              </Link>
            </nav>
          </div>
        </header>

        {/* Persistent Medical Disclaimer Banner */}
        <div className="bg-amber-50 border-b border-amber-200 text-amber-900 px-4 py-2 text-xs sm:text-sm text-center font-medium">
          ⚠️ <strong>Educational demo. Not medical advice. Not clinically validated.</strong> This tool uses historical cohort statistics for scientific evaluation and must never replace clinical assessment.
        </div>

        {/* Main Content Area */}
        <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-500">
          <div className="max-w-6xl mx-auto px-4 space-y-2">
            <p>
              Framingham Heart Study ML Risk Prediction System • Production Architecture
            </p>
            <p className="text-slate-400">
              Calibrated Probabilities • Out-of-Fold Threshold Selection • Zero Data Leakage Pipeline
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
