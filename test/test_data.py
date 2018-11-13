import pytest

import mlp.data as data


def test_iris():
    x, y = data.load_iris()
    assert (150, 4) == x.shape
    assert (150,) == y.shape


def test_mnist():
    x, y = data.load_mnist()
    assert (70000, 784) == x.shape
    assert (70000,) == y.shape
