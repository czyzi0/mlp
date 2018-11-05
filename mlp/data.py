"""

"""

import csv
import pathlib
from typing import Tuple

import numpy as np
import requests

from .utils import spinner


def download_iris(data_dir: str, verbose: bool = False) -> bool:
    """

    """
    iris_file_path = pathlib.Path(data_dir) / 'iris.data'
    if not iris_file_path.exists():
        response = requests.get(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            stream=True
        )
        with open(iris_file_path, 'wb') as iris_file:
            for data_chunk in spinner(
                    response.iter_content(),
                    message='Downloading Iris data: ',
                    verbose=verbose
                ):
                iris_file.write(data_chunk)
        return True
    return False


def load_iris(
        data_dir: pathlib.Path, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """

    """
    iris_file_path = pathlib.Path(data_dir) / 'iris.data'
    with open(iris_file_path, 'r') as iris_file:
        reader = csv.reader(iris_file, delimiter=',')
        x, y = [], []
        for row in spinner(
                (row for row in reader if row),
                message='Loading Iris data: ',
                verbose=verbose
            ):
            x.append([float(number) for number in row[:4]])
            y.append(row[4])
    x = np.array(x)
    y = np.array(y)
    return x, y


def download_mnist(data_dir: pathlib.Path, verbose: bool = False) -> bool:
    """

    """
    pass


def load_mnist(
        data_dir: pathlib.Path, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """

    """
    pass
