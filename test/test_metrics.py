import numpy as np
import pytest

import mlp.metrics as metrics


@pytest.mark.parametrize('y_pred, y_true, expected_output', [
    (
        np.array(['a', 'b', 'c', 'a', 'b', 'c']),
        np.array(['a', 'b', 'c', 'a', 'b', 'c']),
        1.0,
    ),
    (
        np.array(['a', 'b', 'c', 'a', 'b']),
        np.array(['a', 'b', 'c', 'a', 'c']),
        0.8,
    ),
    (
        np.array([1, 2, 3, 4]),
        np.array([1, 2, 3, 6]),
        0.75,
    ),
    (
        np.array([1.0, 2.0, 3.0, 4.1]),
        np.array([1.1, 2.1, 3.0, 6.0]),
        0.25,
    ),
    (
        np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
        0.5,
    ),
])
def test_accuracy(y_pred, y_true, expected_output):
    output = metrics.accuracy(y_pred, y_true)
    assert pytest.approx(expected_output) == output


@pytest.mark.parametrize('y_pred, y_true', [
    (
        np.array([1, 2, 3]),
        np.array([1, 2, 3, 4]),
    ),
])
def test_accuracy_raises_error(y_pred, y_true):
    with pytest.raises(ValueError):
        metrics.accuracy(y_pred, y_true)


@pytest.mark.parametrize('y_pred, y_true, expected_output', [
    (
        np.array([
            [0.1, 0.2, 0.76],
            [0.04, 0.01, 0.9],
        ]),
        np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]),
        1.0,
    ),
    (
        np.array([
            [0.1, 0.1, 0.2],
            [0.8, 0.4, 0.6],
        ]),
        np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        0.5,
    ),
    (
        np.array([
            [0.1, 0.2, 0.76],
            [0.04, 0.01, 0.9],
            [0.1, 0.1, 0.2],
            [0.8, 0.4, 0.6],
        ]),
        np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        0.75,
    ),
])
def test__accuracy(y_pred, y_true, expected_output):
    assert metrics._accuracy._name in metrics._METRICS_DICT

    internal_output, compare_output = metrics._accuracy(y_pred, y_true)
    assert pytest.approx(expected_output) == internal_output
    assert pytest.approx(expected_output) == compare_output


@pytest.mark.parametrize('y_pred, y_true, expected_output', [
    (
        np.array([
            [0.1, 0.2, 0.76],
            [0.04, 0.01, 0.9],
        ]),
        np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]),
        0.01988333,
    ),
    (
        np.array([
            [0.1, 0.1, 0.2],
            [0.8, 0.4, 0.6],
        ]),
        np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        0.23666666,
    ),
    (
        np.array([
            [0.1, 0.2, 0.76],
            [0.04, 0.01, 0.9],
            [0.1, 0.1, 0.2],
            [0.8, 0.4, 0.6],
        ]),
        np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        0.128275,
    ),
    (
        np.array([1.1, 2.44, 3.0]),
        np.array([1.0, 2.0, 2.78]),
        0.084,
    ),
])
def test_mean_squared_error(y_pred, y_true, expected_output):
    assert metrics._mean_squared_error._name in metrics._METRICS_DICT

    internal_output, compare_output = metrics._mean_squared_error(y_pred, y_true)
    external_output = metrics.mean_squared_error(y_pred, y_true)

    assert pytest.approx(expected_output) == internal_output
    assert pytest.approx(-expected_output) == compare_output

    assert pytest.approx(expected_output) == external_output


@pytest.mark.parametrize('y_pred, y_true', [
    (
        np.array([1.1, 2.04, 3.0]),
        np.array([1.0, 2.0, 2.98, 4.1]),
    ),
])
def test_mean_squared_error_raises_error(y_pred, y_true):
    with pytest.raises(ValueError):
        metrics.mean_squared_error(y_pred, y_true)
