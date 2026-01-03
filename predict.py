"""Simple inference helper"""
import numpy as np
from .model import load_model

def predict_proba(model_path: str, X: np.ndarray) -> np.ndarray:
    clf = load_model(model_path)
    return clf.predict_proba(X)
