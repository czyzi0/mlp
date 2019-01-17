import numpy as np
import pytest

import mlp.activations as activations


@pytest.mark.parametrize('x, xpcd_y, xpcd_derivative', [
    (
        np.array([
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0]]),
        np.array([
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0]]),
        np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]]),
    ),
])
def test__identity(x, xpcd_y, xpcd_derivative):
    assert 'identity' in activations._ACTIVATION_DICT

    y = activations._identity(x)
    assert np.allclose(xpcd_y, y)

    derivative = activations._identity(x, derivative=True)
    assert np.allclose(xpcd_derivative, derivative)


@pytest.mark.parametrize('x, xpcd_y, xpcd_derivative', [
    (
        np.array([[-1.0, 0.0, 0.00001, 1.0]]),
        np.array([[0.0, 0.0, 0.00001, 1.0]]),
        np.array([[0.0, 0.0, 1.0, 1.0]])
    ),
])
def test__relu(x, xpcd_y, xpcd_derivative):
    assert 'relu' in activations._ACTIVATION_DICT

    y = activations._relu(x)
    assert np.allclose(xpcd_y, y)

    derivative = activations._relu(x, derivative=True)
    assert np.allclose(xpcd_derivative, derivative)


@pytest.mark.parametrize('x, xpcd_y, xpcd_derivative', [
    (
        np.array([
            [0.0, 1.0],
            [100.0, 1000.0],
            [-1.0, -100.0],
            [-1000.0, 0.0]]),
        np.array([
            [0.5, 0.7310586],
            [1.0, 1.0],
            [0.2689414, 0.0],
            [0.0, 0.5]]),
        np.array([
            [0.25, 0.1966119],
            [0.0, 0.0],
            [0.1966119, 0.0],
            [0.0, 0.25]])
    ),
])
def test__sigmoid(x, xpcd_y, xpcd_derivative):
    assert 'sigmoid' in activations._ACTIVATION_DICT

    y = activations._sigmoid(x)
    assert np.allclose(xpcd_y, y)

    derivative = activations._sigmoid(x, derivative=True)
    assert np.allclose(xpcd_derivative, derivative)


@pytest.mark.parametrize('x, xpcd_y, xpcd_derivative', [
    (
        np.array([
            [0.0, 1.0, 100.0, 1000.0],
            [0.0, -1.0, -100.0, -1000.0]]),
        np.array([
            [0.6931472, 1.3132617, 100.0, 1000.0],
            [0.6931472, 0.3132617, 0.0, 0.0]]),
        np.array([
            [0.5, 0.7310586, 1.0, 1.0],
            [0.5, 0.2689414, 0.0, 0.0]])
    ),
])
def test__softplus(x, xpcd_y, xpcd_derivative):
    assert 'softplus' in activations._ACTIVATION_DICT

    y = activations._softplus(x)
    assert np.allclose(xpcd_y, y)

    derivative = activations._softplus(x, derivative=True)
    assert np.allclose(xpcd_derivative, derivative)


@pytest.mark.parametrize('x, xpcd_y, xpcd_derivative', [
    (
        np.array([
            [0.0, 1.0, 1000.0],
            [0.0, -1.0, -1000.0]]),
        np.array([
            [0.0, 0.7615942, 1.0],
            [0.0, -0.7615942, -1.0]]),
        np.array([
            [1.0, 0.4199743, 0.0],
            [1.0, 0.4199743, 0.0]])
    ),
])
def test__tanh(x, xpcd_y, xpcd_derivative):
    assert 'tanh' in activations._ACTIVATION_DICT

    y = activations._tanh(x)
    assert np.allclose(xpcd_y, y)

    derivative = activations._tanh(x, derivative=True)
    assert np.allclose(xpcd_derivative, derivative)
