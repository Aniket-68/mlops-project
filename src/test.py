import pytest
import pickle
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path
import subprocess
import sys

# Fixture to load the California Housing dataset
@pytest.fixture
def california_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test

# Fixture to load the trained model
@pytest.fixture
def model():
    model_path = Path("models/model.pkl")
    assert model_path.exists(), f"Model file {model_path} does not exist"
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Test that the model can be loaded
def test_model_loading(model):
    assert model is not None, "Failed to load the model"

# Test that the model's mean squared error is within a reasonable threshold
def test_model_performance(model, california_data):
    X_test, y_test = california_data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    # Assuming a reasonable MSE threshold for a linear regression model
    assert mse < 1.0, f"Model MSE {mse} is too high (expected < 1.0)"

# Test that the model produces consistent predictions
def test_prediction_consistency(model, california_data):
    X_test, _ = california_data
    # Use a subset of test data for consistency check
    X_sample = X_test[:10]
    y_pred_1 = model.predict(X_sample)
    y_pred_2 = model.predict(X_sample)
    assert np.array_equal(y_pred_1, y_pred_2), "Model predictions are not consistent"

# Test to ensure code passes Flake8 linting
def test_flake8_linting():
    files_to_lint = ["src/train.py", "test.py"]
    try:
        result = subprocess.run(
            [sys.executable, "-m", "flake8"] + files_to_lint,
            check=True,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Flake8 linting passed"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Flake8 linting failed:\n{e.stdout}")