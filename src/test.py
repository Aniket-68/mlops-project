import pytest
from sklearn.metrics import mean_squared_error

def test_dummy():
    assert mean_squared_error([1,2], [1,2]) == 0
    assert mean_squared_error([1,2], [1,2]) == 0