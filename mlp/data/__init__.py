"""Module with data sets connected functions."""

import pathlib
from typing import Tuple

import numpy as np


def load_iris() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads Iris data set.

    https://archive.ics.uci.edu/ml/datasets/iris

    Returns:
        Tuple of four numpy arrays: `train_x`, `train_y`, `test_x`, `test_y`.

    """
    iris_path = str(pathlib.Path(__file__).parent / 'iris.npz')
    return _load_data(iris_path)


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads MNIST data set.

    http://yann.lecun.com/exdb/mnist/

    Returns:
        Tuple of four numpy arrays: `train_x`, `train_y`, `test_x`, `test_y`.

    """
    mnist_path = str(pathlib.Path(__file__).parent / 'mnist.npz')
    return _load_data(mnist_path)


def _load_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads data from previously prepared `npz` file.

    Args:
        path: Path to data file.

    Returns:
        Tuple of four numpy arrays: `train_x`, `train_y`, `test_x`, `test_y`.

    """
    data = np.load(path)
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    return train_x, train_y, test_x, test_y
