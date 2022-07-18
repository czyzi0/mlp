"""Module with various utility tools."""

import sys
from typing import Any, Generator, Iterable, Tuple

import numpy as np


def argmax(x: np.ndarray) -> np.ndarray:
    """Transforms vectors to one-hot vectors with 1 at hightes values.

    Function searches for maximal value in each of vectors and sets it to 1 and
    the rest to 0.

    Args:
        x: Vectors to process.

    Returns:
        One-hotted vectors.

    """
    indices = np.argmax(x, axis=1)

    y = np.zeros_like(x)
    y[np.arange(len(indices)), indices] = 1.0

    return y


def chunked(array: np.ndarray, chunk_size: int) -> Generator[np.ndarray, None, None]:
    """Breaks array into chunks of length chunk_size.

    Args:
        array: Array to break.
        chunk_size: Length of chunk.

    Yields:
        Chunks of length `chunk_size`.

    """
    for i in range(0, len(array), chunk_size):
        yield array[i:i + chunk_size]


def unison_shuffle(array1: np.ndarray, array2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffles two arrays in unison.

    Args:
        array1: First array to shuffle.
        array2: Second array to shuffle.

    Returns:
        Tuple of shuffled in unison arrays.

    """
    if array1.shape[0] != array2.shape[0]:
        raise ValueError('lengths of array1 and array2 are different')
    permutation = np.random.permutation(len(array1))
    return array1[permutation], array2[permutation]


def progress_bar(
        iterable: Iterable[Any],
        total: int,
        step: int = 1,
        verbose: bool = True,
        n_cols: int = 30
    ) -> Generator[Any, None, None]:
    """Wraps given iterable and prints progress bar.

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
