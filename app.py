#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import uvicorn
import logging
from random import uniform
import time
import hashlib

# Load the pre-trained SVC model
try:
    model = joblib.load("data/svc_model.pkl")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the pre-trained model")

# Create the FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="An API for classifying iris flowers",
    version="1.0.0",
    contact={"name": "Your Name", "email": "your.email@example.com"},
)

# Define Pydantic models for input validation
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BatchIrisFeatures(BaseModel):
    instances: list[IrisFeatures]

# Generate a unique code signature
code_signature = hashlib.md5("Your unique string here".encode()).hexdigest()

# Define API endpoints
@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]])
        prediction = model.predict(input_data)[0]
        return {"prediction": str(prediction), "code_signature": code_signature}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(batch: BatchIrisFeatures):
    try:
        instances = [np.array([[
            instance.sepal_length,
            instance.sepal_width,
            instance.petal_length,
            instance.petal_width,
        ]]) for instance in batch.instances]
        predictions = model.predict(np.concatenate(instances))
        return {"predictions": [str(p) for p in predictions], "code_signature": code_signature}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict_random")
def predict_random():
    try:
        features = {
            "sepal_length": uniform(4.3, 7.9),
            "sepal_width": uniform(2.0, 4.4),
            "petal_length": uniform(1.0, 6.9),
            "petal_width": uniform(0.1, 2.5),
        }
        input_data = np.array([[
            features["sepal_length"],
            features["sepal_width"],
            features["petal_length"],
            features["petal_width"],
        ]])
        prediction = model.predict(input_data)[0]
        return {
            "features": features,
            "prediction": str(prediction),
            "code_signature": code_signature,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return "Healthy"

@app.get("/model_info")
def model_info():
    return {
        "model_type": "Support Vector Classifier",
        "model_version": "1.0",
        "model_description": "A Support Vector Classifier trained on the Iris dataset for species classification.",
        "code_signature": code_signature,
    }

@app.get("/simulate_workload")
def simulate_workload(seconds: int = 1):
    try:
        time.sleep(seconds)
        return f"Workload simulated for {seconds} seconds."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
