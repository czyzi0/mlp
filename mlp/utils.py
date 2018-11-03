"""Module with various utility tools.

"""

import json
import sys
import time
from typing import Any, Generator, Iterable, Optional, Tuple

import numpy as np


class PrincipalComponentAnalysis:
    """Principal Component Analysis.

    Attributes:
        _model: 

    """

    def __init__(self):
        self._model = np.identity(1)

    def train(self, train_vectors_in, dims):
        train_vecs_in = np.array(train_vectors_in, ndmin=2).T

        covariance_matrix = np.cov(train_vecs_in)

        eigen_values, eigen_vecs = np.linalg.eig(covariance_matrix)
        eigen_pairs = [
            (np.abs(eigen_values[i]), np.atleast_2d(eigen_vecs[:, i]).T)
            for i in range(len(eigen_values))
        ]
        eigen_pairs.sort(key=lambda x: x[0], reverse=True)

        self._model = np.hstack([eigen_pairs[i][1] for i in range(dims)])

    def transform(self, vectors_in):
        vecs_pred = np.matmul(self._model.T, np.array(vectors_in, ndmin=2).T).T
        if isinstance(vectors_in, type([])):
            return vecs_pred.tolist()
        return vecs_pred

    def save(self, file_path):
        model = self._model.tolist()
        with open(file_path, 'w') as file_:
            json.dump(model, file_, indent=4, sort_keys=True)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as file_:
            model = json.loads(file_.read())
            pca = PrincipalComponentAnalysis()
            # pylint: disable=protected-access
            pca._model = np.array(model, ndmin=2)
            return pca


def argmax(x: np.ndarray) -> np.ndarray:
    """One hot given vectors.

    Function searches for maximal value in each of vectors and sets it to 1 and
    the rest to 0.

    Args:
        x: Vectors to one hot.

    Returns:
        One-hotted vectors.

    """
    x = np.array(x, ndmin=2)
    indices = np.argmax(x, axis=1)

    y = np.zeros_like(x)
    y[np.arange(len(indices)), indices] = 1.0

    return y


def chunked(array: np.ndarray, chunk_size: int) -> Generator[np.ndarray, None, None]:
    """Break array into chunks of length chunk_size.

    Args:
        array: Array to break.
        chunk_size: Length of chunk.

    Yields:
        Chunks of length chunk_size.

    """
    for i in range(0, len(array), chunk_size):
        yield array[i:i + chunk_size]


def unison_shuffle(array1: np.ndarray, array2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle two arrays in unison.

    Args:
        array1: First array to shuffle.
        array2: Second array to shuffle.

    Returns:
        Tuple of shuffled in unison arrays.

    """
    permutation = np.random.permutation(len(array1))
    return array1[permutation], array2[permutation]


def progress_bar(
        iterable: Iterable[Any],
        total: Optional[int] = None,
        step: int = 1,
        verbose: bool = True,
        n_cols: int = 30
    ) -> Generator[Any, None, None]:
    """Wrap given iterable and print progress bar.

    Args:
        iterable: Iterable to wrap.
        total: Total number of values in iterable. If not given function will
               try to find out length of iterable by itself.
        step: Step for one iteration.
        verbose: If set to False wrapper won't print anything.
        n_cols: Width of progress bar.

    Yields:
        Consecutive values from given iterable.

    """
    if verbose:
        state = 0
        if total is None:
            total = len(iterable)

    for item in iterable:
        yield item
        if verbose:
            state = min(state + step, total)
            bar_length = int(n_cols * state / total)
            # pylint: disable=blacklisted-name
            bar = '=' * bar_length + '.' * (n_cols - bar_length)
            print(f'\r{state}/{total} [{bar}]', file=sys.stderr, end='')

    if verbose:
        print(file=sys.stderr)


def spinner(
        iterable: Iterable[Any],
        message: str = '',
        verbose: bool = True
    ) -> Generator[Any, None, None]:
    """Wrap given iterable and print spinner.

    Args:
        iterable: Iterable to wrap.
        message: Message to print before spinner.
        verbose: If set False wrapper won't print anything.

    Yields:
        Consecutive values from given iterable.

    """
    if verbose:
        markers = '|/-\\'
        print(f'{message} ', file=sys.stderr, end='')

    for item in iterable:
        yield item
        if verbose:
            current_marker = int(10 * time.time()) % len(markers)
            print(f'\b{markers[current_marker]}', file=sys.stderr, end='')

    if verbose:
        print('\bdone', file=sys.stderr)
