import numpy as np
from fraud_detection.model import train, save_model, load_model
import tempfile
def test_train_save_load():
    X = np.random.RandomState(0).randint(0,2,size=(100, 5))
    y = np.random.RandomState(1).randint(0,2,size=(100,))
    clf = train(X, y)
    f = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    save_model(clf, f.name)
    loaded = load_model(f.name)
    preds = loaded.predict(X[:5])
    assert preds.shape[0] == 5
