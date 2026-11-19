import { NextResponse } from "next/server";
import metricsData from "../../../../public/reports/metrics.json";
import featureImportanceData from "../../../../public/reports/feature_importance.json";

export async function GET() {
  return NextResponse.json({
    metadata: metricsData,
    feature_importance: featureImportanceData,
  });
}
