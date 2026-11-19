"use client";

import React, { useState } from "react";
import { PredictionInput, PredictionResponse } from "@/types/prediction";
import { ResultCard } from "@/components/ResultCard";

// Default to internal Next.js API route (/api/predict) for Vercel serverless deployment
// or fallback to external Python FastAPI backend if NEXT_PUBLIC_API_URL is explicitly set
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/predict`
  : "/api/predict";

const INITIAL_FORM: PredictionInput = {
  age: 50,
  male: 1,
  currentSmoker: 0,
  cigsPerDay: null,
  BPMeds: 0,
  prevalentStroke: 0,
  prevalentHyp: 0,
  diabetes: 0,
  totChol: 220,
  sysBP: 125,
  diaBP: 80,
  BMI: 25.5,
  heartRate: 72,
  glucose: 85,
};

export default function HomePage() {
  const [formData, setFormData] = useState<PredictionInput>(INITIAL_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setError(null);

    let parsedVal: number | null = null;
    if (value !== "") {
      parsedVal = type === "number" ? parseFloat(value) : parseInt(value, 10);
    }

    setFormData((prev) => {
      const updated = { ...prev, [name]: parsedVal };
      if (name === "currentSmoker" && Number(value) === 0) {
        updated.cigsPerDay = 0;
      }
      return updated;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Client-side validation matching API schemas
    if (formData.age < 30 || formData.age > 80) {
      setError("Age must be between 30 and 80 years.");
      setLoading(false);
      return;
    }
    if (formData.sysBP < 80 || formData.sysBP > 300) {
      setError("Systolic Blood Pressure must be between 80 and 300 mmHg.");
      setLoading(false);
      return;
    }
    if (formData.diaBP < 40 || formData.diaBP > 160) {
      setError("Diastolic Blood Pressure must be between 40 and 160 mmHg.");
      setLoading(false);
      return;
    }
    if (formData.totChol && (formData.totChol < 100 || formData.totChol > 700)) {
      setError("Total Cholesterol must be between 100 and 700 mg/dL.");
      setLoading(false);
      return;
    }
    if (formData.BMI && (formData.BMI < 12 || formData.BMI > 60)) {
      setError("BMI must be between 12.0 and 60.0 kg/m².");
      setLoading(false);
      return;
    }
    if (formData.glucose && (formData.glucose < 40 || formData.glucose > 500)) {
      setError("Fasting Glucose must be between 40 and 500 mg/dL.");
      setLoading(false);
      return;
    }
    if (formData.heartRate && (formData.heartRate < 40 || formData.heartRate > 200)) {
      setError("Heart Rate must be between 40 and 200 bpm.");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(API_BASE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.detail?.[0]?.msg ||
            errorData.detail ||
            `Prediction failed with status ${response.status}`
        );
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch (err: unknown) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "Failed to connect to the prediction API. Ensure backend is running.";
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setFormData(INITIAL_FORM);
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Intro Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          10-Year Coronary Heart Disease Risk Predictor
        </h1>
        <p className="text-slate-600 max-w-2xl mx-auto text-sm sm:text-base">
          Clinical risk assessment tool using a calibrated, zero-leakage logistic regression model trained on the historical Framingham Heart Study cohort.
        </p>
      </div>

      {/* Result Card or Form */}
      {result ? (
        <ResultCard result={result} onReset={handleReset} />
      ) : (
        <form
          onSubmit={handleSubmit}
          className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 sm:p-8 space-y-8"
        >
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm flex items-center justify-between">
              <span>{error}</span>
              <button
                type="button"
                onClick={() => setError(null)}
                className="text-red-500 hover:text-red-700 font-bold ml-2"
              >
                ✕
              </button>
            </div>
          )}

          {/* Section 1: Demographics */}
          <div className="space-y-4">
            <div className="border-b border-slate-100 pb-2">
              <h3 className="text-base font-semibold text-slate-900">1. Demographics</h3>
              <p className="text-xs text-slate-500">Biological attributes recorded at initial examination.</p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label htmlFor="age" className="block text-xs font-medium text-slate-700 mb-1">
                  Age (years) <span className="text-red-500">*</span>
                </label>
                <input
                  id="age"
                  name="age"
                  type="number"
                  min="30"
                  max="80"
                  required
                  value={formData.age}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
                <span className="text-[11px] text-slate-400">Valid range: 30 - 80</span>
              </div>

              <div>
                <label htmlFor="male" className="block text-xs font-medium text-slate-700 mb-1">
                  Biological Sex at Birth <span className="text-red-500">*</span>
                </label>
                <select
                  id="male"
                  name="male"
                  required
                  value={formData.male}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white"
                >
                  <option value={1}>Male</option>
                  <option value={0}>Female</option>
                </select>
                <span className="text-[11px] text-slate-400">Cohort baseline predictor</span>
              </div>
            </div>
          </div>

          {/* Section 2: Blood Pressure & Vitals */}
          <div className="space-y-4">
            <div className="border-b border-slate-100 pb-2">
              <h3 className="text-base font-semibold text-slate-900">2. Blood Pressure & Hemodynamics</h3>
              <p className="text-xs text-slate-500">Blood pressure metrics and resting cardiovascular vitals.</p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label htmlFor="sysBP" className="block text-xs font-medium text-slate-700 mb-1">
                  Systolic BP (mmHg) <span className="text-red-500">*</span>
                </label>
                <input
                  id="sysBP"
                  name="sysBP"
                  type="number"
                  step="0.5"
                  min="80"
                  max="300"
                  required
                  value={formData.sysBP}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
                <span className="text-[11px] text-slate-400">Range: 80 - 300 mmHg</span>
              </div>

              <div>
                <label htmlFor="diaBP" className="block text-xs font-medium text-slate-700 mb-1">
                  Diastolic BP (mmHg) <span className="text-red-500">*</span>
                </label>
                <input
                  id="diaBP"
                  name="diaBP"
                  type="number"
                  step="0.5"
                  min="40"
                  max="160"
                  required
                  value={formData.diaBP}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
                <span className="text-[11px] text-slate-400">Range: 40 - 160 mmHg</span>
              </div>

              <div>
                <label htmlFor="heartRate" className="block text-xs font-medium text-slate-700 mb-1">
                  Resting Heart Rate (bpm)
                </label>
                <input
                  id="heartRate"
                  name="heartRate"
                  type="number"
                  min="40"
                  max="200"
                  value={formData.heartRate ?? ""}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
                <span className="text-[11px] text-slate-400">Optional (imputed if missing)</span>
              </div>
            </div>
          </div>

          {/* Section 3: Metabolic Labs & Body Mass */}
          <div className="space-y-4">
            <div className="border-b border-slate-100 pb-2">
              <h3 className="text-base font-semibold text-slate-900">3. Metabolic & Laboratory Vitals</h3>
              <p className="text-xs text-slate-500">Serum biochemistry and anthropometric measures.</p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label htmlFor="totChol" className="block text-xs font-medium text-slate-700 mb-1">
                  Total Cholesterol (mg/dL)
                </label>
                <input
                  id="totChol"
                  name="totChol"
                  type="number"
                  min="100"
                  max="700"
                  value={formData.totChol ?? ""}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
                <span className="text-[11px] text-slate-400">Range: 100 - 700 mg/dL</span>
              </div>

              <div>
                <label htmlFor="glucose" className="block text-xs font-medium text-slate-700 mb-1">
                  Fasting Glucose (mg/dL)
                </label>
                <input
                  id="glucose"
                  name="glucose"
                  type="number"
                  min="40"
                  max="500"
                  value={formData.glucose ?? ""}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
                <span className="text-[11px] text-slate-400">Range: 40 - 500 mg/dL</span>
              </div>

              <div>
                <label htmlFor="BMI" className="block text-xs font-medium text-slate-700 mb-1">
                  Body Mass Index (kg/m²)
                </label>
                <input
                  id="BMI"
                  name="BMI"
                  type="number"
                  step="0.1"
                  min="12"
                  max="60"
                  value={formData.BMI ?? ""}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
                <span className="text-[11px] text-slate-400">Range: 12.0 - 60.0 kg/m²</span>
              </div>
            </div>
          </div>

          {/* Section 4: Lifestyle & Medical History */}
          <div className="space-y-4">
            <div className="border-b border-slate-100 pb-2">
              <h3 className="text-base font-semibold text-slate-900">4. Lifestyle & Medical History</h3>
              <p className="text-xs text-slate-500">Smoking behavior and diagnosed cardiovascular conditions.</p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label htmlFor="currentSmoker" className="block text-xs font-medium text-slate-700 mb-1">
                  Current Smoker <span className="text-red-500">*</span>
                </label>
                <select
                  id="currentSmoker"
                  name="currentSmoker"
                  value={formData.currentSmoker}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white"
                >
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div>
                <label htmlFor="cigsPerDay" className="block text-xs font-medium text-slate-700 mb-1">
                  Cigarettes per Day
                </label>
                <input
                  id="cigsPerDay"
                  name="cigsPerDay"
                  type="number"
                  min="0"
                  max="80"
                  disabled={formData.currentSmoker === 0}
                  value={formData.cigsPerDay ?? ""}
                  onChange={handleChange}
                  placeholder={formData.currentSmoker === 0 ? "0 (non-smoker)" : "e.g. 15"}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none disabled:bg-slate-100 disabled:text-slate-400"
                />
                <span className="text-[11px] text-slate-400">0 - 80 cigs/day</span>
              </div>

              <div>
                <label htmlFor="BPMeds" className="block text-xs font-medium text-slate-700 mb-1">
                  On Blood Pressure Medication
                </label>
                <select
                  id="BPMeds"
                  name="BPMeds"
                  value={formData.BPMeds ?? 0}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white"
                >
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div>
                <label htmlFor="diabetes" className="block text-xs font-medium text-slate-700 mb-1">
                  Diagnosed Diabetes Mellitus <span className="text-red-500">*</span>
                </label>
                <select
                  id="diabetes"
                  name="diabetes"
                  value={formData.diabetes}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white"
                >
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div>
                <label htmlFor="prevalentHyp" className="block text-xs font-medium text-slate-700 mb-1">
                  Diagnosed Hypertension <span className="text-red-500">*</span>
                </label>
                <select
                  id="prevalentHyp"
                  name="prevalentHyp"
                  value={formData.prevalentHyp}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white"
                >
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div>
                <label htmlFor="prevalentStroke" className="block text-xs font-medium text-slate-700 mb-1">
                  Prior History of Stroke <span className="text-red-500">*</span>
                </label>
                <select
                  id="prevalentStroke"
                  name="prevalentStroke"
                  value={formData.prevalentStroke}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white"
                >
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>
            </div>
          </div>

          {/* Form Actions */}
          <div className="pt-4 border-t border-slate-100 flex items-center justify-end space-x-3">
            <button
              type="button"
              onClick={handleReset}
              className="px-4 py-2 border border-slate-300 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
            >
              Reset Values
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-semibold shadow-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                  </svg>
                  <span>Calculating Risk...</span>
                </>
              ) : (
                <span>Calculate 10-Year CHD Risk</span>
              )}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
