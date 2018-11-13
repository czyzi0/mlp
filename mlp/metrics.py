"""Module with metrics function definitions.

"""

from typing import Tuple

import numpy as np

from .utils import argmax


class ConfusionMatrix:

    def __init__(self):
        pass


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate accuracy.

    Args:
        y_pred: Predicted labels (2d).
        y_true: Ground truth (correct) labels (2d).

    Returns:
        Accuracy score.

    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f'shapes of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) are different')
    if np.issubdtype(y_pred.dtype, np.number) and np.issubdtype(y_true.dtype, np.number):
        correct_n = sum(np.allclose(v_pred, v_true) for v_pred, v_true in zip(y_pred, y_true))
    else:
        correct_n = sum(y_pred == y_true)
    acc = correct_n / len(y_pred)
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
    acc = accuracy(argmax(y_pred), y_true)
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
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f'shapes of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) are different')
    mse = np.mean(np.square(y_pred - y_true))
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
    mse = mean_squared_error(y_pred, y_true)
    return mse, -mse
_mean_squared_error._name = 'mean_squared_error'


_METRICS_DICT = {
    _accuracy._name: _accuracy,
    _mean_squared_error._name: _mean_squared_error,
}
