import pytest
import numpy as np
from ml.model import train_model, inference, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier

# Dummy data for tests
X_dummy = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
y_dummy = np.array([0, 1, 0, 1])


def test_train_model():
    model = train_model(X_dummy, y_dummy)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier"

def test_inference():
    model = train_model(X_dummy, y_dummy)
    preds = inference(model, X_dummy)
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array"
    assert len(preds) == len(y_dummy), "Predictions should match the number of input samples"

def test_compute_model_metrics():
    precision, recall, f1 = compute_model_metrics(y_dummy, y_dummy)
    assert precision == 1.0, "Precision perfecto score: 1.0 "
    assert recall == 1.0, "Recall perfect score: 1.0"
    assert f1 == 1.0, "F1 perfect socre: 1.0"