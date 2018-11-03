import numpy as np
import pytest

import mlp.metrics as metrics


@pytest.mark.parametrize('y_pred, y_true, expected_output', [
    (
        np.array([
            [-1.0],
            [0.8],
            [0.1],
        ]),
        np.array([
            [-1.0],
            [1.0],
            [0.0],
        ]),
        1.0,
    ),
    (
        np.array([
            [7.1],
            [0.9],
        ]),
        np.array([
            [7.0],
            [0.0],
        ]),
        0.5,
    ),
    (
        np.array([
            [-1.0],
            [0.8],
            [0.1],
            [7.1],
            [0.9],
        ]),
        np.array([
            [-1.0],
            [1.0],
            [0.0],
            [7.0],
            [0.0],
        ]),
        0.8,
    ),
])
def test_binary_accuracy(y_pred, y_true, expected_output):
    assert metrics._binary_accuracy._name in metrics._METRICS_DICT

    internal_output, compare_output = metrics._binary_accuracy(y_pred, y_true)
    external_output = metrics.binary_accuracy(y_pred, y_true)

    assert internal_output >= 0.0
    assert internal_output <= 1.0

    assert external_output == pytest.approx(internal_output)
    assert compare_output == pytest.approx(internal_output)

    assert expected_output == pytest.approx(internal_output)


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
def test_categorical_accuracy(y_pred, y_true, expected_output):
    assert metrics._categorical_accuracy._name in metrics._METRICS_DICT

    internal_output, compare_output = metrics._categorical_accuracy(y_pred, y_true)
    external_output = metrics.categorical_accuracy(y_pred, y_true)

    assert internal_output >= 0.0
    assert internal_output <= 1.0

    assert external_output == pytest.approx(internal_output)
    assert compare_output == pytest.approx(internal_output)

    assert expected_output == pytest.approx(internal_output)


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
])
def test_mean_squared_error(y_pred, y_true, expected_output):
    assert metrics._mean_squared_error._name in metrics._METRICS_DICT

    internal_output, compare_output = metrics._mean_squared_error(y_pred, y_true)
    external_output = metrics.mean_squared_error(y_pred, y_true)

    assert internal_output >= 0.0
    assert internal_output <= 1.0

    assert external_output == pytest.approx(internal_output)
    assert compare_output == pytest.approx(-internal_output)

    assert expected_output == pytest.approx(internal_output)
