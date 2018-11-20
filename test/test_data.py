import pytest

import mlp.data as data


def test_iris():
    train_x, train_y, test_x, test_y = data.load_iris()
    assert (120, 4) == train_x.shape
    assert (120,) == train_y.shape
    assert (30, 4) == test_x.shape
    assert (30,) == test_y.shape


def test_mnist():
    train_x, train_y, test_x, test_y = data.load_mnist()
    assert (60000, 784) == train_x.shape
    assert (60000,) == train_y.shape
    assert (10000, 784) == test_x.shape
    assert (10000,) == test_y.shape
