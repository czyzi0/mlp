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
def test_argmax(x, expected_y):
    y = utils.argmax(x)
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
def test_chunked(array, chunk_size, expected_chunks):
    chunks = list(utils.chunked(array, chunk_size=chunk_size))
    for expected_chunk, chunk in zip(expected_chunks, chunks):
        assert np.allclose(expected_chunk, chunk)


@pytest.mark.parametrize('array_shape', [
    ((12, 4500)),
    ((100, 1)),
    ((50, 4)),
    ((1500, 12)),
    ((60000, 768)),
])
def test_unison_shuffle(array_shape):
    array1 = np.arange(array_shape[0] * array_shape[1]).reshape(array_shape)
    array2 = np.arange(array_shape[0] * array_shape[1]).reshape(array_shape)

    array1, array2 = utils.unison_shuffle(array1, array2)

    assert np.allclose(array1, array2)


@pytest.mark.parametrize('array1, array2', [
    (
        np.array([1, 2, 3, 4, 5]),
        np.array([1, 2, 3, 4, 5, 6]),
    ),
])
def test_unison_shuffle_raises_error(array1, array2):
    with pytest.raises(ValueError):
        utils.unison_shuffle(array1, array2)


@pytest.mark.parametrize('iterable, total, verbose', [
    (list(range(100)), 100, False),
    (list(range(100)), 100, True),
    (np.arange(1500), 1500, False),
    (range(300), 300, True),
    (range(300), 300, False),
    (list(range(50)), 50, True),
])
def test_progress_bar(iterable, total, verbose):
    for item1, item2 in zip(iterable, utils.progress_bar(iterable, total=total, verbose=verbose)):
        assert item1 == item2
