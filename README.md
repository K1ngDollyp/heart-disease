# 10-Year Coronary Heart Disease (CHD) Risk Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-compose-blue.svg)](https://www.docker.com/)

A production-grade, clinical machine learning risk prediction system built on the historical **Framingham Heart Study dataset** (`aasheesh200`). The project predicts 10-year risk of developing Coronary Heart Disease (`TenYearCHD`, ~15% positive prevalence) with strict zero-leakage standards, probability calibration, sensitivity-oriented threshold selection, a robust FastAPI backend service, and a responsive Next.js 14 App Router user interface.

> ⚠️ **Persistent Disclaimer:** Educational demo. Not medical advice. Not clinically validated.

---

## 1. Architecture Overview

```
SIWES PROJECT 2/
├── data/
│   └── framingham.csv          # Framingham Heart Study dataset (4,240 rows, 16 columns)
├── ml/
│   ├── eda.ipynb               # Systematic exploratory data analysis and clinical decisions
│   ├── train.py                # 5-fold CV comparison, calibration, threshold tuning, pipeline export
│   ├── evaluate.py             # Single-pass held-out test set evaluation script
│   ├── requirements.txt        # Pinned dependencies for reproducible ML environment
│   ├── split/                  # Stratified 80/20 train/test split (saved for full auditability)
│   └── reports/                # ROC, PR, Calibration, Confusion Matrix, and metrics.json
├── api/
│   ├── main.py                 # FastAPI service with startup model cache, CORS, and logging
│   ├── schemas.py              # Pydantic v2 schemas with clinical range validations
│   ├── model/                  # Exported calibrated pipeline and model_metadata.json
│   ├── tests/                  # Pytest suite (health, valid, boundary, missing fields, validation)
│   └── Dockerfile              # Production Python 3.11-slim container
├── web/
│   ├── src/app/                # Next.js 14 App Router (page.tsx, layout.tsx, about-model/page.tsx)
│   ├── src/components/         # Responsive form, semi-circular SVG RiskGauge, ResultCard
│   ├── src/types/              # Strong TypeScript interfaces matching API Pydantic schemas
│   ├── public/reports/         # Static evaluation plots and feature importance JSON
│   └── Dockerfile              # Multi-stage production Node.js 20-alpine container
├── docker-compose.yml          # Single-command orchestration for API and Web stack
└── README.md                   # System documentation, clinical results, and limitations
```

---

## 2. Experimental Results & Benchmark Comparison

All candidate models were evaluated across identical **5-fold Stratified Cross-Validation folds** on the 80% training set ($N = 3,392$). **The 20% test set ($N = 848$) was never used for model selection, hyperparameter tuning, calibration, or threshold selection.**

### 5-Fold Cross-Validation Comparison

| Model Candidate | CV ROC-AUC | CV PR-AUC | CV Brier Score | Strategy / Weights |
| :--- | :---: | :---: | :---: | :--- |
| **Dummy Baseline** | 0.5000 ± 0.0000 | 0.1518 ± 0.0001 | 0.1288 ± 0.0001 | Prior frequency baseline |
| **Logistic Regression (Selected)** | **0.7369 ± 0.0296** | **0.3597 ± 0.0335** | **0.2067 ± 0.0110** | Balanced weights + Sigmoid calibration |
| **Random Forest** | 0.7183 ± 0.0315 | 0.3414 ± 0.0494 | 0.1977 ± 0.0063 | Balanced weights, `max_depth=5` |
| **XGBoost** | 0.7172 ± 0.0329 | 0.3389 ± 0.0412 | 0.2024 ± 0.0079 | `scale_pos_weight=5.59`, `max_depth=3` |
| **LightGBM** | 0.7143 ± 0.0326 | 0.3378 ± 0.0476 | 0.2019 ± 0.0085 | `scale_pos_weight=5.59`, `num_leaves=8` |

### Why Logistic Regression Was Deployed (Occam's Razor)
1. **Performance:** Calibrated Logistic Regression achieved a CV ROC-AUC of **0.7369**, slightly exceeding tree ensembles (RF 0.7183, XGB 0.7172, LGBM 0.7143).
2. **Clinical Interpretability & Monotonicity:** Standard clinical guidelines (Framingham Risk Score, ACC/AHA ASCVD Risk Estimator) require monotonic, additive odds-ratio interpretations. Non-linear tree partitions risk producing step-function anomalies for borderline blood pressure or cholesterol values.
3. **Realistic Metrics:** The observed ROC-AUC of **~0.70-0.74** conforms directly to established medical literature for the Framingham cohort.

---

## 3. Final Held-Out Test Evaluation (Evaluated Once)

After freezing the pipeline, probability calibrator, and decision threshold on the training set, final evaluation was performed **once** on the held-out test cohort ($N = 848$, 129 positives):

- **Test ROC-AUC:** `0.6925` (Benchmark: ~0.70 - 0.75)
- **Test PR-AUC:** `0.2823` (Baseline prevalence: 15.2%)
- **Brier Score:** `0.1224` (Reliable, calibrated probability estimates)
- **Screening Decision Threshold:** `0.120` (Tuned on training out-of-fold predictions to optimize $F_2$-score)
- **Sensitivity / Recall:** **70.54%** ($91 / 129$ CHD cases successfully flagged)
- **Specificity:** **56.33%** ($405 / 719$ true negatives)
- **Precision / PPV:** **22.47%**
- **$F_2$-Score:** **0.4940**
- **Confusion Matrix:**
  - True Negatives (TN): **405**
  - False Positives (FP): **314**
  - False Negatives (FN): **38**
  - True Positives (TP): **91**

---

## 4. Key Preprocessing & Data Leakage Guards

1. **Dropping `education`:** `education` (1-4) is a non-clinical socioeconomic factor that introduces demographic and geographical bias into pathophysiological cardiovascular scoring. It was dropped before training.
2. **Pipeline Encapsulation:** Median imputation (`SimpleImputer(strategy='median', add_indicator=True)`) and standardization (`StandardScaler`) are encapsulated directly within the scikit-learn `Pipeline`. All statistical parameters (medians, means, standard deviations) are fit **exclusively** on training folds.
3. **Probability Calibration:** The base estimator is wrapped in `CalibratedClassifierCV(method='sigmoid')` trained via cross-validation to output well-calibrated clinical probabilities.
4. **Out-of-Fold Threshold Selection:** Rather than picking an arbitrary $0.50$ threshold on an imbalanced 15% dataset, the decision cutoff ($0.120$) was derived from out-of-fold cross-validation predictions maximizing the $F_2$-score, heavily penalizing false negatives in clinical screening.

---

## 5. API Reference (`/api`)

The FastAPI service loads the serialized pipeline into memory once at startup and provides structured JSON logging, CORS middleware, and Pydantic v2 validation.

### Endpoints
- `GET /health`: Health status and loaded model verification.
- `GET /model-info`: Complete model metadata, valid feature ranges, test metrics, and permutation feature importances.
- `POST /predict`: Computes calibrated risk probability, clinical tier, and binary recommendation.

#### Example Request (`POST /predict`):
```json
{
  "age": 55,
  "male": 1,
  "currentSmoker": 1,
  "cigsPerDay": 20.0,
  "BPMeds": 1.0,
  "prevalentStroke": 0,
  "prevalentHyp": 1,
  "diabetes": 0,
  "totChol": 260.0,
  "sysBP": 155.0,
  "diaBP": 95.0,
  "BMI": 28.5,
  "heartRate": 78.0,
  "glucose": 95.0
}
```

#### Example Response:
```json
{
  "probability": 0.443,
  "risk_level": "high",
  "risk_label": "High Risk (>=20%)",
  "prediction": 1,
  "threshold": 0.12,
  "explanation": "The estimated 10-year coronary heart disease risk is 44.3%. This represents high cardiovascular risk (>=20% 10-year risk), warranting close medical review of lipids, blood pressure, and lifestyle.",
  "disclaimer": "Educational demo. Not medical advice. Not clinically validated."
}
```

### Risk Stratification Cutoffs:
- **Low Risk:** $< 10\%$ 10-year CHD probability
- **Moderate / Borderline Risk:** $10\% - 20\%$ 10-year CHD probability
- **High Risk:** $\ge 20\%$ 10-year CHD probability (clinical trigger for statin & intensive therapy consideration)

---

## 6. Running Locally & Retraining

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Docker and Docker Compose (optional for containerized deployment)

### 1. Retrain the Machine Learning Model
```bash
# Install Python dependencies
pip install -r ml/requirements.txt

# Run training, cross-validation, calibration, and artifact generation
python ml/train.py

# Run standalone evaluation on held-out test split
python ml/evaluate.py
```

### 2. Run API Unit & Integration Tests
```bash
pytest api/tests -v
```

### 3. Run FastAPI Backend Locally
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation: `http://127.0.0.1:8000/docs`

### 4. Run Next.js 14 Frontend Locally
```bash
cd web
npm install
npm run dev
```
Open your browser to: `http://localhost:3000`

---

## 7. Containerized Deployment (`docker-compose`)

Build and launch the entire production stack with health checks:
```bash
docker-compose up --build
```
- Web Application: `http://localhost:3000`
- API Service: `http://localhost:8000`
- API Health Check: `http://localhost:8000/health`

---

## 8. Clinical Limitations & Ethical Boundaries

1. **Cohort Demographics:** The Framingham Heart Study cohort consists almost exclusively of white, middle-class residents of Framingham, Massachusetts enrolled in the 1950s–1970s. It lacks racial, ethnic, and socio-economic diversity.
2. **Temporal Changes:** Cardiovascular disease treatment (widespread statin use, modern antihypertensives, primary percutaneous coronary interventions) has dramatically improved over the last 50 years, changing baseline population hazard rates.
3. **Missing Contemporary Biomarkers:** Modern risk stratification relies on biomarkers absent from this dataset, including high-sensitivity C-reactive protein (hs-CRP), coronary artery calcium (CAC) scoring, and apolipoprotein B (ApoB).
4. **Not Medical Advice:** This application is strictly an educational demonstration of robust software and machine learning engineering. It is **not** cleared by regulatory bodies for clinical diagnosis or treatment.

