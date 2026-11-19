import React from "react";
import Image from "next/image";
import Link from "next/link";
import fs from "fs";
import path from "path";
import { ModelMetadata, FeatureImportance } from "@/types/prediction";

function getModelData(): { metadata: ModelMetadata | null; featureImportance: FeatureImportance[] } {
  try {
    const metaPath = path.join(process.cwd(), "public", "reports", "metrics.json");
    const impPath = path.join(process.cwd(), "public", "reports", "feature_importance.json");

    const metadata = fs.existsSync(metaPath)
      ? JSON.parse(fs.readFileSync(metaPath, "utf-8"))
      : null;

    const featureImportance = fs.existsSync(impPath)
      ? JSON.parse(fs.readFileSync(impPath, "utf-8"))
      : [];

    return { metadata, featureImportance };
  } catch {
    return { metadata: null, featureImportance: [] };
  }
}

export default function AboutModelPage() {
  const { metadata, featureImportance } = getModelData();

  return (
    <div className="space-y-12 max-w-5xl mx-auto py-2">
      {/* Page Title */}
      <div className="border-b border-slate-200 pb-6 space-y-2">
        <Link
          href="/"
          className="text-xs font-semibold text-blue-600 hover:text-blue-800 flex items-center space-x-1 mb-2"
        >
          <span>← Back to Risk Calculator</span>
        </Link>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          Model Architecture, Validation & Clinical Evaluation
        </h1>
        <p className="text-slate-600 text-sm sm:text-base leading-relaxed">
          Full technical disclosure of the modeling pipeline, cross-validation methodology, test set evaluation metrics, calibration analysis, and ethical limitations.
        </p>
      </div>

      {/* 1. Executive Summary & Zero-Leakage Policy */}
      <section className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
        <h2 className="text-xl font-bold text-slate-900 flex items-center space-x-2">
          <span>1. Architecture & Zero-Leakage Protocol</span>
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-700">
          <div className="space-y-3">
            <h3 className="font-semibold text-slate-900">Dataset & Cohort</h3>
            <p>
              Trained on the Framingham Heart Study dataset (4,240 participants, 15 clinical features). The primary target is <code>TenYearCHD</code> (~15.2% positive prevalence).
            </p>
            <p>
              <strong>Education Feature Dropped:</strong> Socioeconomic attainment is an indirect social proxy that biases algorithms across differing healthcare environments and is omitted in modern ASCVD risk guidelines.
            </p>
          </div>
          <div className="space-y-3">
            <h3 className="font-semibold text-slate-900">Strict Leakage Guards</h3>
            <ul className="list-disc list-inside space-y-1 text-slate-600">
              <li><strong>80/20 Stratified Split:</strong> Test set was completely held out and evaluated exactly once.</li>
              <li><strong>Pipeline Isolation:</strong> Median imputation and feature scaling are encapsulated inside an sklearn <code>Pipeline</code>, fit only within training folds.</li>
              <li><strong>OOF Threshold Optimization:</strong> Decision threshold was chosen purely from out-of-fold CV predictions to maximize sensitivity (F2 score).</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 2. 5-Fold CV Model Comparison */}
      {metadata && (
        <section className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
          <h2 className="text-xl font-bold text-slate-900">
            2. 5-Fold Cross-Validation Model Comparison
          </h2>
          <p className="text-sm text-slate-600">
            All algorithms were evaluated over identical 5-fold stratified cross-validation splits. Realistic clinical ROC-AUC is typically 0.70 - 0.75.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-slate-600 uppercase text-[11px] tracking-wider">
                  <th className="py-2.5 px-3">Model Candidate</th>
                  <th className="py-2.5 px-3">ROC-AUC (Mean ± Std)</th>
                  <th className="py-2.5 px-3">PR-AUC (Mean ± Std)</th>
                  <th className="py-2.5 px-3">Brier Score</th>
                  <th className="py-2.5 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-xs">
                {Object.entries(metadata.cross_validation_results).map(([name, m]) => (
                  <tr key={name} className={name === "LogisticRegression" ? "bg-blue-50/60 font-semibold" : ""}>
                    <td className="py-2.5 px-3 font-sans text-slate-900">
                      {name} {name === "LogisticRegression" && <span className="text-[10px] bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded ml-1 font-semibold">SELECTED</span>}
                    </td>
                    <td className="py-2.5 px-3">{m.roc_auc.mean.toFixed(4)} ± {m.roc_auc.std.toFixed(4)}</td>
                    <td className="py-2.5 px-3">{m.pr_auc.mean.toFixed(4)} ± {m.pr_auc.std.toFixed(4)}</td>
                    <td className="py-2.5 px-3">{m.brier.mean.toFixed(4)}</td>
                    <td className="py-2.5 px-3 font-sans text-xs">
                      {name === "LogisticRegression" ? "Deployed" : "Evaluated"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-xs text-slate-700 space-y-1">
            <strong>Model Selection Rationale:</strong> {metadata.model_selection_reason}
          </div>
        </section>
      )}

      {/* 3. Held-Out Test Evaluation Metrics */}
      {metadata && (
        <section className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
          <h2 className="text-xl font-bold text-slate-900">
            3. Final Test Set Evaluation (Evaluated Once on 20% Held-Out Cohort)
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
              <span className="text-xs text-slate-500 uppercase font-medium">ROC-AUC</span>
              <div className="text-2xl font-bold text-slate-900">{metadata.test_metrics.roc_auc.toFixed(4)}</div>
              <span className="text-[11px] text-slate-400">Realistic benchmark ~0.70-0.75</span>
            </div>
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
              <span className="text-xs text-slate-500 uppercase font-medium">PR-AUC</span>
              <div className="text-2xl font-bold text-slate-900">{metadata.test_metrics.pr_auc.toFixed(4)}</div>
              <span className="text-[11px] text-slate-400">Baseline prevalence ~15.2%</span>
            </div>
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
              <span className="text-xs text-slate-500 uppercase font-medium">Brier Score</span>
              <div className="text-2xl font-bold text-slate-900">{metadata.test_metrics.brier_score.toFixed(4)}</div>
              <span className="text-[11px] text-slate-400">Probability calibration loss</span>
            </div>
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
              <span className="text-xs text-slate-500 uppercase font-medium">Sensitivity (Recall)</span>
              <div className="text-2xl font-bold text-emerald-600">{(metadata.test_metrics.recall * 100).toFixed(1)}%</div>
              <span className="text-[11px] text-slate-400">At threshold = {metadata.decision_threshold.toFixed(2)}</span>
            </div>
          </div>
        </section>
      )}

      {/* 4. Validation Visualizations */}
      <section className="bg-white rounded-xl border border-slate-200 p-6 space-y-6">
        <h2 className="text-xl font-bold text-slate-900">
          4. Validation Plots & Calibration Analysis
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="border border-slate-200 rounded-lg p-3 bg-slate-50 space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">Receiver Operating Characteristic (ROC)</h4>
            <div className="relative h-64 w-full">
              <Image src="/reports/roc_curve.png" alt="ROC Curve" fill className="object-contain" />
            </div>
            <p className="text-[11px] text-slate-500">Discrimination across true positive vs false positive trade-offs.</p>
          </div>

          <div className="border border-slate-200 rounded-lg p-3 bg-slate-50 space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">Precision-Recall (PR) Curve</h4>
            <div className="relative h-64 w-full">
              <Image src="/reports/pr_curve.png" alt="PR Curve" fill className="object-contain" />
            </div>
            <p className="text-[11px] text-slate-500">Precision-Recall trade-off given the 15% positive class imbalance.</p>
          </div>

          <div className="border border-slate-200 rounded-lg p-3 bg-slate-50 space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">Probability Calibration Curve (Reliability Diagram)</h4>
            <div className="relative h-64 w-full">
              <Image src="/reports/calibration_curve.png" alt="Calibration Curve" fill className="object-contain" />
            </div>
            <p className="text-[11px] text-slate-500">Evaluates empirical outcome frequency against model-predicted probabilities.</p>
          </div>

          <div className="border border-slate-200 rounded-lg p-3 bg-slate-50 space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">Confusion Matrix</h4>
            <div className="relative h-64 w-full">
              <Image src="/reports/confusion_matrix.png" alt="Confusion Matrix" fill className="object-contain" />
            </div>
            <p className="text-[11px] text-slate-500">Classification matrix computed at tuned screening threshold = 0.12.</p>
          </div>
        </div>
      </section>

      {/* 5. Feature Importance */}
      {featureImportance.length > 0 && (
        <section className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
          <h2 className="text-xl font-bold text-slate-900">
            5. Global Feature Importance (Permutation Importance on ROC-AUC)
          </h2>
          <p className="text-sm text-slate-600">
            Permutation importance quantifies the drop in ROC-AUC when a feature is shuffled randomly.
          </p>
          <div className="space-y-3">
            {featureImportance.map((f) => (
              <div key={f.feature} className="space-y-1">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-800">{f.feature}</span>
                  <span className="font-mono text-slate-500">{f.importance_mean.toFixed(4)} ± {f.importance_std.toFixed(4)}</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${Math.max(f.importance_mean * 1500, 4)}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 6. Honest Limitations & Ethical Constraints */}
      <section className="bg-amber-50/70 border border-amber-200 rounded-xl p-6 space-y-4">
        <h2 className="text-xl font-bold text-amber-950 flex items-center space-x-2">
          <span>⚠️ 6. Clinical Limitations & Algorithmic Biases</span>
        </h2>
        <div className="text-xs sm:text-sm text-amber-900 space-y-3 leading-relaxed">
          <p>
            <strong>Framingham Cohort Bias:</strong> The underlying Framingham Heart Study data originated from an exclusively Caucasian cohort in Framingham, Massachusetts beginning in the mid-20th century. Cardiovascular disease epidemiology, smoking prevalence, and medication therapies have evolved substantially over the past several decades.
          </p>
          <p>
            <strong>Lack of Demographic & Genetic Diversity:</strong> This model has not been validated on ethnically, geographically, or socioeconomically diverse patient populations. Application of this model to diverse populations may overestimate or underestimate absolute risk.
          </p>
          <p>
            <strong>Absence of Modern Biomarkers:</strong> Modern cardiology relies on high-sensitivity cardiac troponin, high-sensitivity C-reactive protein (hs-CRP), coronary artery calcium (CAC) scoring, and detailed lipid subfractions (LDL-C, HDL-C, ApoB) which are absent from this historical dataset.
          </p>
          <p>
            <strong>Non-Clinical Demo:</strong> This application is intended solely as an educational demonstration of robust ML architecture, probability calibration, and zero-leakage development. It has not received FDA clearance or clinical trial validation and must never be utilized for diagnosis or treatment decisions.
          </p>
        </div>
      </section>
    </div>
  );
}

