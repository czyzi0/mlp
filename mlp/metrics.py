"""Module with metrics function definitions.

"""

from typing import Tuple

import numpy as np

from .utils import argmax


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate accuracy.

    Args:
        y_pred: Predicted labels (2d).
        y_true: Ground truth (correct) labels (2d).

    Returns:
        Accuracy score.

    """
    acc, _ = _accuracy(np.array(y_pred, ndmin=2), np.array(y_true, ndmin=2))
    return acc


def _accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Calculate accuracy.

    For internal use only.

    Args:
        y_pred: Predicted labels (2d).
        y_true: Ground truth (correct) labels (2d).

    Returns:
        Tuple of proper accuracy score and score for comparison (greater means
        better score).

    """
    correct_num = sum(
        np.allclose(v_pred, v_true) for v_pred, v_true in zip(argmax(y_pred), y_true))
    acc = correct_num / len(y_pred)
    return acc, acc
_accuracy._name = 'accuracy'


def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate mean squared error.

    Args:
        y_pred: Predicted labels (2d).
        y_true: Ground truth (correct) labels (2d).

    Returns:
        Mean squared error score.

    """
    mse, _ = _mean_squared_error(np.array(y_pred, ndmin=2), np.array(y_true, ndmin=2))
    return mse


def _mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Calculate mean squared error.

    For internal use only.

    Args:
        y_pred: Predicted labels (2d).
        y_true: Ground truth (correct) labels (2d).

    Returns:
        Tuple of proper mean squared error score and score for comparison
        (greater means better score).

    """
    mse = np.mean(np.square(y_pred - y_true))
    return mse, - mse
_mean_squared_error._name = 'mean_squared_error'


_METRICS_DICT = {
    _accuracy._name: _accuracy,
    _mean_squared_error._name: _mean_squared_error,
}
