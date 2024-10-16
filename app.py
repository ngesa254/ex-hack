from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import uvicorn
import logging
import random
import time
import hashlib

# Load the model 
try:
    model = joblib.load('svc_model.pkl')
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    raise RuntimeError("Could not load the model. Ensure 'svc_model.pkl' is available.")

# Create the FastAPI app
app = FastAPI(
    title="Iris Flower Classification API",
    description="A simple API that classifies iris flowers based on sepal and petal measurements.",
    version="1.0.0",
    contact={
        "name": "Samurai",
    },
)

# Define the request schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BatchIrisFeatures(BaseModel):
    features: List[IrisFeatures]

# Unique identifier for the code
CODE_SIGNATURE = hashlib.md5("IrisAPI_v1.0.0_Hackathon".encode()).hexdigest()

# Endpoint to predict iris species
@app.post("/predict", summary="Predict Iris Species", tags=["Prediction"])
async def predict(features: IrisFeatures):
    try:
        # Log incoming request
        logging.info(f"Received request with features: {features}")
        
        # Prepare the data for prediction
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Perform the prediction
        prediction = model.predict(input_data)
        prediction_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        
        # Return the response with meaningful prediction
        return {
            "prediction": prediction_map[int(prediction[0])],
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Batch prediction endpoint
@app.post("/predict_batch", summary="Batch Predict Iris Species", tags=["Prediction"])
async def predict_batch(batch_features: BatchIrisFeatures):
    try:
        # Log incoming request
        logging.info(f"Received batch request with {len(batch_features.features)} sets of features")
        
        # Prepare the data for batch prediction
        input_data = np.array([
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ] for features in batch_features.features
        ])
        
        # Perform the prediction
        predictions = model.predict(input_data)
        prediction_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        
        # Return the response with meaningful predictions
        return {
            "predictions": [prediction_map[int(pred)] for pred in predictions],
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logging.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

# Random prediction endpoint
@app.get("/predict_random", summary="Get Random Iris Prediction", tags=["Prediction"])
async def predict_random():
    try:
        # Generate random features within typical ranges for iris flowers
        random_features = IrisFeatures(
            sepal_length=random.uniform(4.0, 8.0),
            sepal_width=random.uniform(2.0, 4.5),
            petal_length=random.uniform(1.0, 7.0),
            petal_width=random.uniform(0.1, 2.5)
        )
        
        # Log generated random features
        logging.info(f"Generated random features: {random_features}")
        
        # Prepare the data for prediction
        input_data = np.array([[
            random_features.sepal_length,
            random_features.sepal_width,
            random_features.petal_length,
            random_features.petal_width
        ]])
        
        # Perform the prediction
        prediction = model.predict(input_data)
        prediction_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        
        # Return the response with meaningful prediction
        return {
            "random_features": random_features.dict(),
            "prediction": prediction_map[int(prediction[0])],
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logging.error(f"Random prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Random prediction failed: {e}")

# Health check endpoint
@app.get("/health", summary="Health Check", tags=["Health"])
async def health_check():
    return {"status": "Healthy", "code_signature": CODE_SIGNATURE}

# Endpoint to get model information
@app.get("/model_info", summary="Get Model Information", tags=["Info"])
async def model_info():
    return {
        "model": "Support Vector Classifier (SVC)",
        "version": "1.0",
        "description": "A model trained on the Iris dataset to classify iris flower species.",
        "code_signature": CODE_SIGNATURE
    }

# Endpoint to simulate workload for testing latency
@app.get("/simulate_workload", summary="Simulate Workload", tags=["Testing"])
async def simulate_workload(seconds: Optional[int] = 1):
    try:
        # Log the workload simulation request
        logging.info(f"Simulating workload for {seconds} seconds")
        
        # Simulate some workload
        time.sleep(seconds)
        
        # Return a success response
        return {
            "message": f"Successfully simulated workload for {seconds} seconds",
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logging.error(f"Workload simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workload simulation failed: {e}")

# Run the app
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)