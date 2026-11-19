import { NextResponse } from "next/server";
import { predictProbability, PatientFeatures } from "@/lib/inference";
import metricsData from "../../../../public/reports/metrics.json";

export async function POST(request: Request) {
  try {
    const body = await request.json();

    // 1. Validation & Range Checks matching production schemas
    const {
      age,
      male,
      currentSmoker,
      cigsPerDay,
      BPMeds,
      prevalentStroke,
      prevalentHyp,
      diabetes,
      totChol,
      sysBP,
      diaBP,
      BMI,
      heartRate,
      glucose,
    } = body;

    if (age === undefined || age < 30 || age > 80) {
      return NextResponse.json(
        { detail: "Age must be between 30 and 80 years." },
        { status: 422 }
      );
    }
    if (male === undefined || (male !== 0 && male !== 1)) {
      return NextResponse.json(
        { detail: "Biological sex must be 0 (Female) or 1 (Male)." },
        { status: 422 }
      );
    }
    if (currentSmoker === undefined || (currentSmoker !== 0 && currentSmoker !== 1)) {
      return NextResponse.json(
        { detail: "Current smoker must be 0 (No) or 1 (Yes)." },
        { status: 422 }
      );
    }
    if (currentSmoker === 0 && cigsPerDay && cigsPerDay > 0) {
      return NextResponse.json(
        { detail: "cigsPerDay cannot be greater than 0 if currentSmoker is 0." },
        { status: 422 }
      );
    }
    if (cigsPerDay !== undefined && cigsPerDay !== null && (cigsPerDay < 0 || cigsPerDay > 80)) {
      return NextResponse.json(
        { detail: "cigsPerDay must be between 0 and 80." },
        { status: 422 }
      );
    }
    if (sysBP === undefined || sysBP < 80 || sysBP > 300) {
      return NextResponse.json(
        { detail: "Systolic Blood Pressure must be between 80 and 300 mmHg." },
        { status: 422 }
      );
    }
    if (diaBP === undefined || diaBP < 40 || diaBP > 160) {
      return NextResponse.json(
        { detail: "Diastolic Blood Pressure must be between 40 and 160 mmHg." },
        { status: 422 }
      );
    }
    if (totChol !== undefined && totChol !== null && (totChol < 100 || totChol > 700)) {
      return NextResponse.json(
        { detail: "Total Cholesterol must be between 100 and 700 mg/dL." },
        { status: 422 }
      );
    }
    if (BMI !== undefined && BMI !== null && (BMI < 12 || BMI > 60)) {
      return NextResponse.json(
        { detail: "BMI must be between 12.0 and 60.0 kg/m²." },
        { status: 422 }
      );
    }
    if (heartRate !== undefined && heartRate !== null && (heartRate < 40 || heartRate > 200)) {
      return NextResponse.json(
        { detail: "Heart Rate must be between 40 and 200 bpm." },
        { status: 422 }
      );
    }
    if (glucose !== undefined && glucose !== null && (glucose < 40 || glucose > 500)) {
      return NextResponse.json(
        { detail: "Fasting Glucose must be between 40 and 500 mg/dL." },
        { status: 422 }
      );
    }

    const patientInput: PatientFeatures = {
      age: Number(age),
      male: Number(male),
      currentSmoker: Number(currentSmoker),
      cigsPerDay: currentSmoker === 0 ? 0 : cigsPerDay !== null && cigsPerDay !== undefined ? Number(cigsPerDay) : null,
      BPMeds: BPMeds !== null && BPMeds !== undefined ? Number(BPMeds) : 0,
      prevalentStroke: Number(prevalentStroke ?? 0),
      prevalentHyp: Number(prevalentHyp ?? 0),
      diabetes: Number(diabetes ?? 0),
      totChol: totChol !== null && totChol !== undefined ? Number(totChol) : null,
      sysBP: Number(sysBP),
      diaBP: Number(diaBP),
      BMI: BMI !== null && BMI !== undefined ? Number(BMI) : null,
      heartRate: heartRate !== null && heartRate !== undefined ? Number(heartRate) : null,
      glucose: glucose !== null && glucose !== undefined ? Number(glucose) : null,
    };

    // 2. Compute exact calibrated probability
    const rawProb = predictProbability(patientInput);
    const prob = Math.round(rawProb * 10000) / 10000;

    // 3. Clinical threshold and stratification
    interface MetricsMeta {
      decision_threshold?: number;
    }
    const meta = metricsData as MetricsMeta;
    const threshold = meta.decision_threshold ?? 0.12;
    const binaryPred = prob >= threshold ? 1 : 0;

    let riskLevel: "low" | "moderate" | "high" = "low";
    let riskLabel = "Low Risk (<10%)";
    let explanation = `The estimated 10-year coronary heart disease risk is ${(prob * 100).toFixed(1)}%. This falls within the low cardiovascular risk tier (<10% 10-year risk).`;

    if (prob >= 0.20) {
      riskLevel = "high";
      riskLabel = "High Risk (>=20%)";
      explanation = `The estimated 10-year coronary heart disease risk is ${(prob * 100).toFixed(1)}%. This represents high cardiovascular risk (>=20% 10-year risk), warranting close medical review of lipids, blood pressure, and lifestyle.`;
    } else if (prob >= 0.10) {
      riskLevel = "moderate";
      riskLabel = "Moderate Risk (10% - 20%)";
      explanation = `The estimated 10-year coronary heart disease risk is ${(prob * 100).toFixed(1)}%. This indicates moderate/borderline cardiovascular risk (10-20% 10-year risk) and suggests lifestyle vigilance.`;
    }

    return NextResponse.json({
      probability: prob,
      risk_level: riskLevel,
      risk_label: riskLabel,
      prediction: binaryPred,
      threshold: threshold,
      explanation: explanation,
      disclaimer: "Educational demo. Not medical advice. Not clinically validated.",
    });
  } catch (error: unknown) {
    const errorMsg = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      { detail: `Inference error: ${errorMsg}` },
      { status: 500 }
    );
  }
}
