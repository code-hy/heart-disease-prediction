from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import uvicorn

# --- 1. INITIALIZE THE APP ---
# This MUST be at the global level (no indentation)
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# --- 2. LOAD THE MODEL ---
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file 'model.pkl' not found. Please run 'python train.py' first."
    )

# Load the model once when the app starts
model = joblib.load(MODEL_PATH)


# --- 3. DEFINE INPUT DATA STRUCTURE ---
class HeartData(BaseModel):
    age: int = Field(..., example=52, description="Age in years")
    sex: int = Field(..., example=1, ge=0, le=1, description="Sex: 1=male, 0=female")
    cp: int = Field(..., example=0, description="Chest Pain Type (0-3)")
    trestbps: int = Field(..., example=125, description="Resting Blood Pressure")
    chol: int = Field(..., example=212, description="Serum Cholesterol")
    fbs: int = Field(..., example=0, description="Fasting Blood Sugar > 120 mg/dl (1=true)")
    restecg: int = Field(..., example=1, description="Resting ECG results")
    thalach: int = Field(..., example=168, description="Max heart rate achieved")
    exang: int = Field(..., example=0, description="Exercise induced angina (1=yes)")
    oldpeak: float = Field(..., example=1.0, description="ST depression induced by exercise")
    slope: int = Field(..., example=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., example=2, description="Number of major vessels (0-3)")
    thal: int = Field(..., example=3, description="Thalassemia: 3=normal, 6=fixed defect, 7=reversable defect")

    class Config:
        schema_extra = {
            "example": {
                "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
                "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
                "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
            }
        }


# --- 4. API ENDPOINTS ---

@app.get("/")
def home():
    """Root endpoint to check if the service is running."""
    return {
        "message": "Heart Disease Prediction Service",
        "status": "Running",
        "docs_link": "/docs"
    }


@app.post("/predict")
def predict(data: HeartData):
    """
    Accepts heart health data and returns a prediction.
    """
    try:
        # Extract features in the exact order the model was trained on
        features = [
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
            data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
            data.ca, data.thal
        ]

        # Reshape for the model (Single sample)
        input_data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk": "High" if prediction == 1 else "Low"
        }

        return result

    except Exception as e:
        # Log the error in a real app, here we just return it
        raise HTTPException(status_code=500, detail=str(e))


# --- 5. LOCAL RUNNING (Optional) ---
# This block allows you to run the file directly with `python predict.py`
# instead of using the `uvicorn` command.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)