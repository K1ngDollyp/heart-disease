export interface PredictionInput {
  age: number;
  male: number;
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

export interface PredictionResponse {
  probability: number;
  risk_level: "low" | "moderate" | "high";
  risk_label: string;
  prediction: number;
  threshold: number;
  explanation: string;
  disclaimer: string;
}

export interface FeatureImportance {
  feature: string;
  importance_mean: number;
  importance_std: number;
}

export interface ModelMetadata {
  model_name: string;
  model_type: string;
  target: string;
  target_description: string;
  features: string[];
  decision_threshold: number;
  threshold_tuning_metric: string;
  training_date: string;
  library_versions: Record<string, string>;
  cross_validation_results: Record<
    string,
    Record<string, { mean: number; std: number }>
  >;
  test_metrics: {
    roc_auc: number;
    pr_auc: number;
    brier_score: number;
    recall: number;
    precision: number;
    f1_score: number;
    f2_score: number;
    confusion_matrix: {
      true_negative: number;
      false_positive: number;
      false_negative: number;
      true_positive: number;
    };
  };
  model_selection_reason: string;
}

