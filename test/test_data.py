import pytest

import mlp.data as data


def test_iris(tmpdir):
    assert data.download_iris(tmpdir)
    assert not data.download_iris(tmpdir)
    x, y = data.load_iris(tmpdir)
    assert (150, 4) == x.shape
    assert (150,) == y.shape
