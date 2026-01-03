# fraud-detection

This repository contains an ensemble-based RandomForest classifier tailored for detecting fraudulent activity in highly imbalanced financial datasets.

## Goals
- Reproducible training and evaluation
- Clear API for inference
- CI and tests to ensure correctness
- Example deployment using FastAPI

## Quickstart
1. Create environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run tests:
   ```bash
   pytest
   ```
3. Train a model (example - replace with your data pipeline):
   ```python
   from fraud_detection.model import train, save_model
   # X, y = load your data
   clf = train(X, y)
   save_model(clf, "models/rf.joblib")
   ```
4. Serve the model with FastAPI:
   ```bash
   uvicorn app.main:app --reload
   ```

## Contributing
- Open issues for bugs or feature requests.
- Follow the coding style (Black + flake8).
- Add tests for new functionality.

## License
Add a license file (e.g. MIT) to clarify usage.
