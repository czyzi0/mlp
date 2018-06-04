"""Module with various utility functions.

"""

import sys
import time
from typing import Any, Generator, Iterable, Optional, Tuple

import numpy as np


def one_hot(x: np.ndarray) -> np.ndarray:
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
