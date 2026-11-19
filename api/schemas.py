"""
api/schemas.py - Pydantic v2 schemas for Coronary Heart Disease risk prediction.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PredictionInput(BaseModel):
    # Demographics
    age: int = Field(
        ...,
        ge=30,
        le=80,
        description="Patient age in years (valid range: 30-80)",
        examples=[52]
    )
    male: int = Field(
        ...,
        ge=0,
        le=1,
        description="Biological sex at birth (0 = Female, 1 = Male)",
        examples=[1]
    )

    # Lifestyle
    currentSmoker: int = Field(
        ...,
        ge=0,
        le=1,
        description="Current smoker status (0 = No, 1 = Yes)",
        examples=[1]
    )
    cigsPerDay: Optional[float] = Field(
        None,
        ge=0,
        le=80,
        description="Number of cigarettes smoked per day (0-80, optional if non-smoker)",
        examples=[15.0]
    )

    # Medical History
    BPMeds: Optional[float] = Field(
        0.0,
        ge=0,
        le=1,
        description="Currently prescribed blood pressure medication (0 = No, 1 = Yes)",
        examples=[0.0]
    )
    prevalentStroke: int = Field(
        ...,
        ge=0,
        le=1,
        description="Prior history of stroke (0 = No, 1 = Yes)",
        examples=[0]
    )
    prevalentHyp: int = Field(
        ...,
        ge=0,
        le=1,
        description="Prevalent diagnosed hypertension (0 = No, 1 = Yes)",
        examples=[1]
    )
    diabetes: int = Field(
        ...,
        ge=0,
        le=1,
        description="Diagnosed diabetes mellitus (0 = No, 1 = Yes)",
        examples=[0]
    )

    # Vitals & Labs (Missing/optional values imputed via median imputer inside pipeline)
    totChol: Optional[float] = Field(
        None,
        ge=100.0,
        le=700.0,
        description="Total serum cholesterol in mg/dL (100-700)",
        examples=[220.0]
    )
    sysBP: float = Field(
        ...,
        ge=80.0,
        le=300.0,
        description="Systolic blood pressure in mmHg (80-300)",
        examples=[138.0]
    )
    diaBP: float = Field(
        ...,
        ge=40.0,
        le=160.0,
        description="Diastolic blood pressure in mmHg (40-160)",
        examples=[88.0]
    )
    BMI: Optional[float] = Field(
        None,
        ge=12.0,
        le=60.0,
        description="Body Mass Index in kg/m^2 (12-60)",
        examples=[26.5]
    )
    heartRate: Optional[float] = Field(
        None,
        ge=40.0,
        le=200.0,
        description="Resting heart rate in beats per minute (40-200)",
        examples=[72.0]
    )
    glucose: Optional[float] = Field(
        None,
        ge=40.0,
        le=500.0,
        description="Fasting blood glucose in mg/dL (40-500)",
        examples=[85.0]
    )

    @model_validator(mode="after")
    def reconcile_smoker_status(self):
        # Clinical consistency: if non-smoker and cigsPerDay is None, set to 0.0
        if self.currentSmoker == 0 and self.cigsPerDay is None:
            self.cigsPerDay = 0.0
        # If currentSmoker == 0 but cigsPerDay > 0, raise validation error
        if self.currentSmoker == 0 and self.cigsPerDay is not None and self.cigsPerDay > 0:
            raise ValueError("cigsPerDay cannot be greater than 0 if currentSmoker is 0.")
        return self


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Calibrated predicted 10-year coronary heart disease probability (0.0 to 1.0)")
    risk_level: str = Field(..., description="Risk tier: 'low', 'moderate', or 'high'")
    risk_label: str = Field(..., description="Human-readable risk category descriptor")
    prediction: int = Field(..., description="Binary classification (1 = elevated risk above threshold, 0 = below threshold)")
    threshold: float = Field(..., description="Screening decision threshold tuned for sensitivity on out-of-fold cross-validation")
    explanation: str = Field(..., description="Plain-language summary of the 10-year CHD risk estimate")
    disclaimer: str = Field(
        default="Educational demo. Not medical advice. Not clinically validated.",
        description="Mandatory medical liability disclaimer"
    )

