"""Module with metrics function definitions."""

from typing import Tuple

import numpy as np

from .utils import argmax


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculates accuracy.

    Args:
        y_pred: Predicted labels. To get labels from `MultilayerPerceptron`
                output you can use `mlp.utils.argmax`.
        y_true: Ground truth (correct) labels.

    Returns:
        Accuracy score.

    """
    if y_pred.shape != y_true.shape:
        raise ValueError('shapes of y_pred and y_true are different')
    if np.issubdtype(y_pred.dtype, np.number) and np.issubdtype(y_true.dtype, np.number):
        correct_n = sum(np.allclose(v_pred, v_true) for v_pred, v_true in zip(y_pred, y_true))
    else:
        correct_n = sum(y_pred == y_true)
    acc = correct_n / len(y_pred)
    return acc


def _accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Calculates accuracy.

    For internal use only.

    Args:
        y_pred: MultilayerPerceptron outputs.
        y_true: Ground truth (correct) labels as one-hot vectors.

    Returns:
        Tuple of proper accuracy score and score for comparison (greater means
        better score).

    """
    acc = accuracy(argmax(y_pred), y_true)
    return acc, acc


def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculates mean squared error.

    Args:
        y_pred: Predicted values.
        y_true: Ground truth (correct) values.

    Returns:
        Mean squared error score.

    """
    if y_pred.shape != y_true.shape:
        raise ValueError('shapes of y_pred and y_true are different')
    mse = np.mean(np.square(y_pred - y_true))
    return mse


def _mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Calculate mean squared error.

    For internal use only.

    Args:
        y_pred: MultilayerPerceptron outputs.
        y_true: Ground truth (correct) values.

    Returns:
        Tuple of proper mean squared error score and score for comparison
        (greater means better score).

    """
    mse = mean_squared_error(y_pred, y_true)
    return mse, -mse


_METRICS_DICT = {
    'accuracy': _accuracy,
    'acc': _accuracy,
    'mean_squared_error': _mean_squared_error,
    'mse': _mean_squared_error}
