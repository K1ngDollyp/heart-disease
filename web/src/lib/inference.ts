import modelWeights from "@/data/modelWeights.json";

export interface PatientFeatures {
  male: number;
  age: number;
  currentSmoker: number;
  cigsPerDay?: number | null;
  BPMeds?: number | null;
  prevalentStroke: number;
  prevalentHyp: number;
  diabetes: number;
  totChol?: number | null;
  sysBP: number;
  diaBP: number;
  BMI?: number | null;
  heartRate?: number | null;
  glucose?: number | null;
}

export function predictProbability(input: PatientFeatures): number {
  const featureNames = modelWeights.features;
  const numRaw = featureNames.length; // 14
  const probs: number[] = [];

  for (const cal of modelWeights.calibrators) {
    // 1. Median Imputation and Missing Indicator Flags (fit on CV training folds)
    const imputedVals: number[] = [];
    const missingFlags: number[] = [];

    for (let i = 0; i < numRaw; i++) {
      const featName = featureNames[i] as keyof PatientFeatures;
      const rawVal = input[featName];
      const isMissing = rawVal === null || rawVal === undefined || isNaN(Number(rawVal));

      if (isMissing) {
        imputedVals.push(cal.imputer_statistics[i]);
      } else {
        imputedVals.push(Number(rawVal));
      }

      if (cal.indicator_features.includes(i)) {
        missingFlags.push(isMissing ? 1.0 : 0.0);
      }
    }

    const transformedFeatures = [...imputedVals, ...missingFlags];

    // 2. StandardScaler: (x - mean) / scale
    const scaledFeatures: number[] = [];
    for (let j = 0; j < transformedFeatures.length; j++) {
      const mean = cal.scaler_mean[j];
      const scale = cal.scaler_scale[j];
      scaledFeatures.push((transformedFeatures[j] - mean) / scale);
    }

    // 3. Logistic Regression Decision Function (linear score)
    let decisionVal = cal.intercept;
    for (let k = 0; k < scaledFeatures.length; k++) {
      decisionVal += scaledFeatures[k] * cal.coef[k];
    }

    // 4. Sigmoid Calibration (Platt Scaling):
    // Scikit-learn SigmoidCalibration evaluates: 1 / (1 + exp(a * T + b)) for class 1
    const calibratedProb1 = 1 / (1 + Math.exp(cal.cal_a * decisionVal + cal.cal_b));
    probs.push(calibratedProb1);
  }

  // 5. Ensemble average over the 5 cross-validation folds
  const avgProb = probs.reduce((sum, p) => sum + p, 0) / probs.length;
  return avgProb;
}
