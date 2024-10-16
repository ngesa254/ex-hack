import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from joblib import load
import logging
import random
import time
import hashlib

# Load the pre-trained SVC model
svc_model = load("svc_model.pkl")

# Create the FastAPI application instance
app = FastAPI(
    title="Iris Classification API",
    description="API for classifying iris flower species based on sepal and petal measurements",
    version="1.0",
    contact={
        "name": "Your Name",
        "email": "your.email@example.com",
    },
)

# Define the Pydantic models for input data validation
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BatchIrisFeatures(BaseModel):
    data: list[IrisFeatures]

# Generate a unique code signature
with open("iris_app.py", "rb") as f:
    code_signature = hashlib.sha256(f.read()).hexdigest()

# API route for single prediction
@app.post("/predict")
def predict(iris_features: IrisFeatures):
    try:
        features = np.array([
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width,
        ]).reshape(1, -1)
        prediction = svc_model.predict(features)[0]
        species = ["setosa", "versicolor", "virginica"][prediction]
        return {"species": species}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API route for batch prediction
@app.post("/predict_batch")
def predict_batch(iris_batch: BatchIrisFeatures):
    try:
        predictions = []
        for iris_features in iris_batch.data:
            features = np.array([
                iris_features.sepal_length,
                iris_features.sepal_width,
                iris_features.petal_length,
                iris_features.petal_width,
            ]).reshape(1, -1)
            prediction = svc_model.predict(features)[0]
            species = ["setosa", "versicolor", "virginica"][prediction]
            predictions.append({"species": species})
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API route for random prediction
@app.get("/predict_random")
def predict_random():
    try:
        sepal_length = random.uniform(4.3, 7.9)
        sepal_width = random.uniform(2.0, 4.4)
        petal_length = random.uniform(1.0, 6.9)
        petal_width = random.uniform(0.1, 2.5)
        features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        prediction = svc_model.predict(features)[0]
        species = ["setosa", "versicolor", "virginica"][prediction]
        return {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
            "species": species,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API route for health check
@app.get("/health")
def health_check():
    return f"Healthy - {code_signature}"

# API route for model information
@app.get("/model_info")
def model_info():
    return {
        "model_type": "Support Vector Classifier",
        "model_version": "1.0",
        "model_description": "Iris species classification model based on sepal and petal measurements",
        "code_signature": code_signature,
    }

# API route for simulating workload
@app.post("/simulate_workload")
def simulate_workload(request_body: dict):
    try:
        delay_in_seconds = request_body.get("delay_in_seconds", 0)
        time.sleep(delay_in_seconds)
        return f"Workload simulated for {delay_in_seconds} seconds"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Start the Uvicorn server if the script is run directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
