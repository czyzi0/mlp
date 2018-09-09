"""Module with metrics function definitions.

"""

from typing import Tuple

import numpy as np

from .utils import one_hot


def binary_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate binary accuracy.

    Args:
        y_pred: Predicted labels (2d).
        y_true: Ground truth (correct) labels (2d).

    Returns:
        Binary accuracy score.

    """
    binary_acc, _ = _binary_accuracy(np.array(y_pred, ndmin=2), np.array(y_true, ndmin=2))
    return binary_acc


def _binary_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Calculate binary accuracy.

    For internal use only.

    Args:
        y_pred: Predicted labels (2d).
        y_true: Ground truth (correct) labels (2d).

    Returns:
        Tuple of proper binary accuracy score and score for comparison (greater
        means better score).

    """
    correct_num = sum(
        np.allclose(v_pred, v_true) for v_pred, v_true in zip(np.round(y_pred), y_true))
    binary_acc = correct_num / len(y_pred)
    return binary_acc, binary_acc
_binary_accuracy._name = 'binary_accuracy'


def categorical_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate categorical accuracy.

    Args:
        y_pred: Predicted labels (2d).
        y_true: Ground truth (correct) labels (2d).

    Returns:
        Categorical accuracy score.

    """
    categorical_acc, _ = _categorical_accuracy(np.array(y_pred, ndmin=2), np.array(y_true, ndmin=2))
    return categorical_acc


def _categorical_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Calculate categorical accuracy.

    For internal use only.

    Args:
        y_pred: Predicted labels (2d).
        y_true: Ground truth (correct) labels (2d).

    Returns:
        Tuple of proper categorical accuracy score and score for comparison
        (greater means better score).

    """
    correct_num = sum(
        np.allclose(v_pred, v_true) for v_pred, v_true in zip(one_hot(y_pred), y_true))
    categorical_acc = correct_num / len(y_pred)
    return categorical_acc, categorical_acc
_categorical_accuracy._name = 'categorical_accuracy'


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
    return mse, -mse
_mean_squared_error._name = 'mean_squared_error'


_METRICS_DICT = {
    _binary_accuracy._name: _binary_accuracy,
    _categorical_accuracy._name: _categorical_accuracy,
    _mean_squared_error._name: _mean_squared_error,
}
