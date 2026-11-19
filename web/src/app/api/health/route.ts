import { NextResponse } from "next/server";
import metricsData from "../../../../public/reports/metrics.json";

interface MetricsMeta {
  model_name?: string;
}

export async function GET() {
  const meta = metricsData as MetricsMeta;
  return NextResponse.json({
    status: "healthy",
    model_loaded: true,
    model_name: meta.model_name ?? "Calibrated Logistic Regression (Framingham CHD Risk)",
    runtime: "Next.js Edge / Serverless API on Vercel",
  });
}
