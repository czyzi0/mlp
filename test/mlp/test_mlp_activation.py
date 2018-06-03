import numpy as np
import pytest

import mlp.activation as activation


@pytest.mark.parametrize('x, expected_y, expected_derivative', [
    (
        np.array([
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
        ]),
        np.array([
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
        ]),
        np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]),
    ),
])
def test__identity(x, expected_y, expected_derivative):
    assert activation._identity._name in activation._ACTIVATION_DICT

    y = activation._identity(x)
    assert np.allclose(expected_y, y)

    derivative = activation._identity(x, derivative=True)
    assert np.allclose(expected_derivative, derivative)


@pytest.mark.parametrize('x, expected_y, expected_derivative', [
    (
        np.array([
            [-1.0, 0.0, 0.00001, 1.0],
        ]),
        np.array([
            [0.0, 0.0, 0.00001, 1.0],
        ]),
        np.array([
            [0.0, 0.0, 1.0, 1.0],
        ])
    ),
])
def test__relu(x, expected_y, expected_derivative):
    assert activation._relu._name in activation._ACTIVATION_DICT

    y = activation._relu(x)
    assert np.allclose(expected_y, y)

    derivative = activation._relu(x, derivative=True)
    assert np.allclose(expected_derivative, derivative)


@pytest.mark.parametrize('x, expected_y, expected_derivative', [
    (
        np.array([
            [0.0, 1.0],
            [100.0, 1000.0],
            [-1.0, -100.0],
            [-1000.0, 0.0],
        ]),
        np.array([
            [0.5, 0.7310586],
            [1.0, 1.0],
            [0.2689414, 0.0],
            [0.0, 0.5],
        ]),
        np.array([
            [0.25, 0.1966119],
            [0.0, 0.0],
            [0.1966119, 0.0],
            [0.0, 0.25],
        ])
    ),
])
def test__sigmoid(x, expected_y, expected_derivative):
    assert activation._sigmoid._name in activation._ACTIVATION_DICT

    y = activation._sigmoid(x)
    assert np.allclose(expected_y, y)

    derivative = activation._sigmoid(x, derivative=True)
    assert np.allclose(expected_derivative, derivative)


@pytest.mark.parametrize('x, expected_y, expected_derivative', [
    (
        np.array([
            [0.0, 1.0, 100.0, 1000.0],
            [0.0, -1.0, -100.0, -1000.0],
        ]),
        np.array([
            [0.6931472, 1.3132617, 100.0, 1000.0],
            [0.6931472, 0.3132617, 0.0, 0.0],
        ]),
        np.array([
            [0.5, 0.7310586, 1.0, 1.0],
            [0.5, 0.2689414, 0.0, 0.0],
        ])
    ),
])
def test__softplus(x, expected_y, expected_derivative):
    assert activation._softplus._name in activation._ACTIVATION_DICT

    y = activation._softplus(x)
    assert np.allclose(expected_y, y)

    derivative = activation._softplus(x, derivative=True)
    assert np.allclose(expected_derivative, derivative)


@pytest.mark.parametrize('x, expected_y, expected_derivative', [
    (
        np.array([
            [0.0, 1.0, 1000.0],
            [0.0, -1.0, -1000.0],
        ]),
        np.array([
            [0.0, 0.7615942, 1.0],
            [0.0, -0.7615942, -1.0],
        ]),
        np.array([
            [1.0, 0.4199743, 0.0],
            [1.0, 0.4199743, 0.0],
        ])
    ),
])
def test__tanh(x, expected_y, expected_derivative):
    assert activation._tanh._name in activation._ACTIVATION_DICT

    y = activation._tanh(x)
    assert np.allclose(expected_y, y)

    derivative = activation._tanh(x, derivative=True)
    assert np.allclose(expected_derivative, derivative)
