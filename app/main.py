from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from fraud_detection.predict import predict_proba

app = FastAPI(title="fraud-detection API")

class Features(BaseModel):
    features: list[list[float]]

@app.post("/predict")
def predict(payload: Features):
    X = np.array(payload.features)
    probs = predict_proba("models/rf.joblib", X)
    return {"probabilities": probs.tolist()}
