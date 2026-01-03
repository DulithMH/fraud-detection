"""Train and save a RandomForest classifier with class-weighting and simple API."""
from typing import Any, Optional
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def make_model(random_state: int = 42, n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, class_weight="balanced_subsample")

def train(X: np.ndarray, y: np.ndarray, params: Optional[dict] = None) -> Any:
    clf = make_model()
    if params:
        clf.set_params(**params)
    clf.fit(X, y)
    return clf

def save_model(clf: Any, path: str) -> None:
    joblib.dump(clf, path)

def load_model(path: str) -> Any:
    return joblib.load(path)
