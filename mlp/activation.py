"""Module with activation function definitions."""

import numpy as np


def _identity(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Identity activation function.

    Args:
        x: Input for function.
        derivative: Flag for calculating function derivative.

    Returns:
        Result.

    """
    if derivative:
        return np.ones_like(x)
    return x


def _relu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """ReLu activation function.

    Args:
        x: Input for function.
        derivative: Flag for calculating function derivative.

    Returns:
        Result.

    """
    # pylint: disable=no-member
    if derivative:
        return np.heaviside(x, 0)
    return np.maximum(x, 0)


def _sigmoid(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Sigmoid activation function.

    Args:
        x: Input for function.
        derivative: Flag for calculating function derivative.

    Returns:
        Result.

    """
    x = np.maximum(x, -100.0)
    sigmoid = 1 / (1 + np.exp(-x))
    if derivative:
        return sigmoid * (1 - sigmoid)
    return sigmoid


def _softplus(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Softplus activation function.

    Args:
        x: Input for function.
        derivative: Flag for calculating function derivative.

    Returns:
        Result.

    """
    x = np.maximum(x, -100.0)
    if derivative:
        return 1 / (1 + np.exp(-x))
    return x + np.log(1 + np.exp(-x))


def _tanh(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Hyperbolic tangent activation function.

    Args:
        x: Input for function.
        derivative: Flag for calculating function derivative.

    Returns:
        Result.

    """
    tanh = np.tanh(x)
    if derivative:
        return 1 - tanh * tanh
    return tanh


_ACTIVATION_DICT = {
    'identity': _identity,
    'relu': _relu,
    'sigmoid': _sigmoid,
    'softplus': _softplus,
    'tanh': _tanh,
}
