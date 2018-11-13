"""

"""

import pathlib
from typing import Tuple

import numpy as np


def load_iris() -> Tuple[np.ndarray, np.ndarray]:
    """

    """
    iris_path = str(pathlib.Path(__file__).parent / 'iris.npz')
    return _load_data(iris_path)


def load_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """

    """
    mnist_path = str(pathlib.Path(__file__).parent / 'mnist.npz')
    return _load_data(mnist_path)


def _load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """

    """
    data = np.load(path)
    x = data['x']
    y = data['y']
    return x, y
