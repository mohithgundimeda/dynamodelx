import pytest
import numpy as np
import torch

from dynamodelx import UFA  # works because your src/dynamodelx is installed or in PYTHONPATH

# ---------------------------
# Fixtures
# ---------------------------
@pytest.fixture
def sample_data():
    X = np.random.rand(100, 10).astype(np.float32)  # 100 samples, 10 features
    y_class = np.random.randint(0, 2, size=(100, 1))  # Binary classification
    y_reg = np.random.rand(100, 1).astype(np.float32)  # Regression
    return X, y_class, y_reg

@pytest.fixture
def ufa_classifier():
    return UFA(task="classification", model_size="small", input_dim=10, output_dim=1, loss="binary_cross_entropy")

@pytest.fixture
def ufa_regressor():
    return UFA(task="regression", model_size="small", input_dim=10, output_dim=1, loss="mean_square_loss")


# ---------------------------
# Tests
# ---------------------------
def test_initialization_classifier(ufa_classifier):
    assert ufa_classifier.task == "classification"
    assert ufa_classifier.model_size == "small"
    assert ufa_classifier.input_dim == 10
    assert ufa_classifier.output_dim == 1
    assert isinstance(ufa_classifier.model, torch.nn.Module)

def test_initialization_regressor(ufa_regressor):
    assert ufa_regressor.task == "regression"
    assert ufa_regressor.loss == "mean_square_loss"
    assert isinstance(ufa_regressor.model, torch.nn.Module)

def test_predict_shapes(ufa_classifier, sample_data):
    X, y_class, _ = sample_data
    ufa_classifier.train(X, y_class, epochs=1, learning_rate=0.01, batch_size=16)
    preds = ufa_classifier.predict(X)
    assert preds.shape == y_class.squeeze(axis=1).shape

def test_train_output(ufa_regressor, sample_data):
    X, _, y_reg = sample_data
    results = ufa_regressor.train(X, y_reg, epochs=1, learning_rate=0.01, batch_size=16)
    assert "train" in results
    assert "validation" in results
    assert "train_loss" in results["train"]
    assert len(results["train"]["train_loss"]) == 1

def test_invalid_task():
    with pytest.raises(ValueError):
        UFA(task="invalid", model_size="small", input_dim=10, output_dim=1, loss="binary_cross_entropy")

def test_invalid_model_size():
    with pytest.raises(ValueError):
        UFA(task="classification", model_size="giant", input_dim=10, output_dim=1, loss="binary_cross_entropy")

def test_empty_dataset(ufa_regressor):
    X = np.empty((0, 10), dtype=np.float32)
    y = np.empty((0, 1), dtype=np.float32)
    with pytest.raises(RuntimeError):
        ufa_regressor.train(X, y, epochs=1, learning_rate=0.01, batch_size=16)
