import argparse
import pathlib

import requests

import mlp


def parse_args():
    pass


def download_iris(target_file_path: pathlib.Path) -> None:
    IRIS_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    response = requests.get(IRIS_URL, stream=True)
    with open(target_file_path, 'wb') as iris_file:
        for data_chunk in mlp.utils.spinner(response.iter_content(), message='Downloading data: '):
            iris_file.write(data_chunk)


if __name__ == '__main__':
    iris_file_path = pathlib.Path(__file__).parent / 'data' / 'iris.data'
    download_iris(iris_file_path)
