import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn import svm

# Data models
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BatchIrisFeatures(BaseModel):
    features: list[IrisFeatures]

# Utility functions
def load_model():
    model = joblib.load("svc_model.pkl")
    return model

def predict_species(model, features):
    feature_array = np.array([features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]).reshape(1, -1)
    predicted_class = model.predict(feature_array)[0]
    species_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predicted_species = species_mapping[predicted_class]
    return predicted_species

# FastAPI application
app = FastAPI()
CODE_SIGNATURE = "YOUR_UNIQUE_CODE_SIGNATURE"
model = load_model()

@app.post("/predict")
def predict_iris_species(iris_features: IrisFeatures):
    predicted_species = predict_species(model, iris_features)
    return {"predicted_species": predicted_species, "code_signature": CODE_SIGNATURE}

@app.post("/predict_batch")
def predict_batch_iris_species(batch_features: BatchIrisFeatures):
    predictions = []
    for features in batch_features.features:
        predicted_species = predict_species(model, features)
        predictions.append(predicted_species)
    return {"predicted_species": predictions, "code_signature": CODE_SIGNATURE}

@app.get("/predict_random")
def predict_random_iris_species():
    import random
    sepal_length = random.uniform(4.3, 7.9)
    sepal_width = random.uniform(2.0, 4.4)
    petal_length = random.uniform(1.0, 6.9)
    petal_width = random.uniform(0.1, 2.5)
    random_features = IrisFeatures(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width)
    predicted_species = predict_species(model, random_features)
    return {"predicted_species": predicted_species, "code_signature": CODE_SIGNATURE}

@app.get("/health")
def health_check():
    return {"message": "OK", "code_signature": CODE_SIGNATURE}

@app.get("/model_info")
def get_model_info():
    model_type = type(model).__name__
    model_params = model.get_params()
    return {"model_type": model_type, "model_params": model_params, "code_signature": CODE_SIGNATURE}

@app.get("/simulate_workload")
def simulate_workload():
    # Implement any logic required for simulating workload or testing latency
    return {"message": "Workload simulated", "code_signature": CODE_SIGNATURE}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
