"""
api/tests/test_api.py - Comprehensive Unit & Integration Tests for CHD API
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health endpoint returns status 200 and indicates model is loaded."""
    with TestClient(app) as c:
        response = c.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "Logistic Regression" in data["model_name"]


def test_model_info_endpoint():
    """Verify model-info endpoint returns metadata, metrics, and feature importance."""
    with TestClient(app) as c:
        response = c.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "metadata" in data
        assert "feature_importance" in data
        assert data["metadata"]["target"] == "TenYearCHD"
        assert "valid_ranges" in data["metadata"]
        assert len(data["feature_importance"]) > 0


def test_valid_prediction():
    """Test valid prediction payload with complete feature set."""
    with TestClient(app) as c:
        payload = {
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
        response = c.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ["low", "moderate", "high"]
        assert data["prediction"] in [0, 1]
        assert "10-year coronary heart disease risk" in data["explanation"].lower()


def test_boundary_values():
    """Test boundary validation values (minimum and maximum bounds)."""
    with TestClient(app) as c:
        # Min bounds
        min_payload = {
            "age": 30,
            "male": 0,
            "currentSmoker": 0,
            "cigsPerDay": 0.0,
            "BPMeds": 0.0,
            "prevalentStroke": 0,
            "prevalentHyp": 0,
            "diabetes": 0,
            "totChol": 100.0,
            "sysBP": 80.0,
            "diaBP": 40.0,
            "BMI": 12.0,
            "heartRate": 40.0,
            "glucose": 40.0
        }
        resp_min = c.post("/predict", json=min_payload)
        assert resp_min.status_code == 200
        assert resp_min.json()["probability"] >= 0.0

        # Max bounds
        max_payload = {
            "age": 80,
            "male": 1,
            "currentSmoker": 1,
            "cigsPerDay": 80.0,
            "BPMeds": 1.0,
            "prevalentStroke": 1,
            "prevalentHyp": 1,
            "diabetes": 1,
            "totChol": 700.0,
            "sysBP": 300.0,
            "diaBP": 160.0,
            "BMI": 60.0,
            "heartRate": 200.0,
            "glucose": 500.0
        }
        resp_max = c.post("/predict", json=max_payload)
        assert resp_max.status_code == 200
        assert resp_max.json()["probability"] <= 1.0


def test_missing_optional_fields():
    """Verify that optional clinical fields are gracefully handled by median imputation inside pipeline."""
    with TestClient(app) as c:
        payload_with_omissions = {
            "age": 45,
            "male": 0,
            "currentSmoker": 0,
            "prevalentStroke": 0,
            "prevalentHyp": 0,
            "diabetes": 0,
            "sysBP": 120.0,
            "diaBP": 80.0
            # omitted: totChol, BMI, heartRate, glucose, BPMeds, cigsPerDay
        }
        response = c.post("/predict", json=payload_with_omissions)
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0


def test_invalid_ranges_fail_fast():
    """Verify that unrealistic out-of-range inputs trigger 422 Unprocessable Entity."""
    with TestClient(app) as c:
        # Age too low (<30)
        resp1 = c.post("/predict", json={"age": 20, "male": 1, "currentSmoker": 0, "sysBP": 120, "diaBP": 80, "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0})
        assert resp1.status_code == 422

        # sysBP unrealistic (>300)
        resp2 = c.post("/predict", json={"age": 50, "male": 1, "currentSmoker": 0, "sysBP": 450, "diaBP": 80, "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0})
        assert resp2.status_code == 422

        # Current smoker 0 but cigsPerDay > 0
        resp3 = c.post("/predict", json={"age": 50, "male": 1, "currentSmoker": 0, "cigsPerDay": 20, "sysBP": 120, "diaBP": 80, "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0})
        assert resp3.status_code == 422

