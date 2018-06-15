"""

"""

import argparse
import pathlib
import sys
from typing import List

import numpy as np
import requests

import mlp


IRIS_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
IRIS_FILE_PATH = pathlib.Path(__file__).parent / 'data' / 'iris.data'


def parse_args(args: List[str]) -> argparse.Namespace:
    """

    """
    parser = argparse.ArgumentParser(description='Train new model or test existing model on Iris data.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--input-model-path', type=pathlib.Path, help='path to the model to be tested')
    group.add_argument('-o', '--output-model-path', type=pathlib.Path, help='path to file to save trained model in')

    return parser.parse_args(args)


def download_iris(file_path: pathlib.Path) -> None:
    """

    """
    response = requests.get(IRIS_URL, stream=True)
    with open(file_path, 'wb') as iris_file:
        for data_chunk in mlp.utils.spinner(response.iter_content(), message='Downloading Iris data: '):
            iris_file.write(data_chunk)


def load_iris(file_path: pathlib.Path) -> np.ndarray:
    """

    """
    pass


def main(args: List[str]) -> None:
    """

    """
    args = parse_args(args)

    if not IRIS_FILE_PATH.exists():
        download_iris(IRIS_FILE_PATH)

    load_iris()
    print(f'Loading Iris data from {IRIS_FILE_PATH.absolute()}')


if __name__ == '__main__':
    main(sys.argv[1:])
