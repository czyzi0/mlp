import numpy as np
import pytest

import mlp.utils as utils


@pytest.mark.parametrize('x, expected_y', [
    (
        np.array([
            [0.8, 0.0, 0.1],
            [0.0, 0.1, 0.9],
            [0.1, 0.2, 0.1],
        ]),
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
    ),
    (
        np.array([
            [0.8, 0.7, 0.9],
            [0.1, 0.1, 0.0],
            [0.2, 0.2, 0.2],
            [23.0, 0.0, -2.0],
        ]),
        np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
    )
])
def test_one_hot(x, expected_y):
    y = utils.one_hot(x)
    assert np.allclose(expected_y, y)


@pytest.mark.parametrize('array, chunk_size, expected_chunks', [
    (
        np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
            [8.0, 8.0],
        ]),
        3,
        [
            np.array([
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]),
            np.array([
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],
            ]),
            np.array([
                [7.0, 7.0],
                [8.0, 8.0],
            ]),
        ]
    ),
    (
        np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
            [8.0, 8.0],
        ]),
        2,
        [
            np.array([
                [1.0, 1.0],
                [2.0, 2.0],
            ]),
            np.array([
                [3.0, 3.0],
                [4.0, 4.0],
            ]),
            np.array([
                [5.0, 5.0],
                [6.0, 6.0],
            ]),
            np.array([
                [7.0, 7.0],
                [8.0, 8.0],
            ]),
        ]
    ),
])
def test__chunked(array, chunk_size, expected_chunks):
    chunks = list(utils._chunked(array, chunk_size=chunk_size))
    for expected_chunk, chunk in zip(expected_chunks, chunks):
        assert np.allclose(expected_chunk, chunk)


@pytest.mark.parametrize('array_shape', [
    ((12, 4500)),
    ((100, 1)),
    ((50, 4)),
    ((1500, 12)),
    ((60000, 768)),
])
def test__unison_shuffle(array_shape):
    array1 = np.arange(array_shape[0] * array_shape[1]).reshape(array_shape)
    array2 = np.arange(array_shape[0] * array_shape[1]).reshape(array_shape)

    array1, array2 = utils._unison_shuffle(array1, array2)

    assert np.allclose(array1, array2)


@pytest.mark.parametrize('iterable', [
    (list(range(100))),
    (np.arange(1500)),
])
def test__progress_bar(iterable):
    for item1, item2 in zip(iterable, utils._progress_bar(iterable, verbose=False)):
        assert item1 == item2
