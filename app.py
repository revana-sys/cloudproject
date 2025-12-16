from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI(title="ML Model API")

# Debug: print current working directory and files
print("Current working dir:", os.getcwd())
print("Files:", os.listdir())

import joblib
import os

MODEL_PATH = "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("📦 Model loaded successfully")
except Exception as e:
    print(f"⚠️ Failed to load model ({e}), retraining...")
    # Insert your model training code here
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print("💾 Model retrained and saved")


# Pydantic model for input data
class InputData(BaseModel):
    age: int
    condition_severity: int
    pain_level: int
    blood_pressure: float
    heart_rate: float
    temperature: float
    consciousness: int
    breathing_difficulty: int
    bleeding: int
    fracture: int
    allergies: int
    pre_existing_conditions: int
    priority_score: int

# Root endpoint
@app.get("/")
def root():
    return {"message": "ML API is running"}

# Health endpoint
@app.get("/healthz")
def healthz():
    return {"status": "ok"}
@app.get("/health")
def health_check():
    return {"status": "ok"}
# Predict endpoint
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Ensure correct column order
        df = df[
            [
                "age", "condition_severity", "pain_level", "blood_pressure",
                "heart_rate", "temperature", "consciousness",
                "breathing_difficulty", "bleeding", "fracture",
                "allergies", "pre_existing_conditions", "priority_score"
            ]
        ]

        # Make prediction
        prediction = int(model.predict(df)[0])

        # Return response
        return {
            "emergency": prediction,
            "label": "EMERGENCY" if prediction == 1 else "NORMAL"
        }

    except Exception as e:
        # Catch all other exceptions
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
