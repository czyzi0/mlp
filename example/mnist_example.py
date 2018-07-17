"""Classification example using MNIST data set.

"""

import argparse
import gzip
import pathlib
from typing import Iterable, List, Tuple

import numpy as np
import requests

import mlp


URL = 'http://yann.lecun.com/exdb/mnist/'
MNIST_DIR_PATH = pathlib.Path(__file__).parent / 'data'
FILE_NAMES = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
]


def parse_args() -> argparse.Namespace:
    """Parse arguments.

    Returns:
        Parsed arguments.

    """
    parser = argparse.ArgumentParser(description='Train new model or evaluate existing model on MNIST data.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input-model-path', type=pathlib.Path, help='path to the model to be evaluated')
    group.add_argument('-o', '--output-model-path', type=pathlib.Path, help='path to file to save trained model in')

    return parser.parse_args()


def download_mnist() -> None:
    """Download MNIST data.loading mnist data wrong magic numbe

    Args: 

    """
    if not all([(MNIST_DIR_PATH / file_name).exists() for file_name in FILE_NAMES]):
        for file_name in FILE_NAMES:
            response = requests.get(URL + file_name, stream=True)
            with open(MNIST_DIR_PATH / file_name, 'wb') as mnist_file:
                for data_chunk in mlp.utils.spinner(
                        response.iter_content(),
                        message=f'Downloading MNIST data file ({file_name}): '
                    ):
                    mnist_file.write(data_chunk)


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST data.

    Returns:
        Two numpy arrays one with inputs (x) and one with expected outputs (y).
        Array shapes are (n, 784) and (n, 10).

    """

    def _number2vec(number: int) -> List[float]:
        vec = [0.0] * 10
        vec[number] = 1.0
        return vec

    def _load_data_set(images_file_name: str, labels_file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        with gzip.open(MNIST_DIR_PATH / images_file_name, 'rb') as images, \
                gzip.open(MNIST_DIR_PATH / labels_file_name, 'rb') as labels:

            images_magic_number = int.from_bytes(images.read(4), byteorder='big')
            if images_magic_number != 2051:
                raise ValueError(f'magic number for images ({images_magic_number}) is not 2051')
            labels_magic_number = int.from_bytes(labels.read(4), byteorder='big')
            if labels_magic_number != 2049:
                raise ValueError(f'magic number for labels ({labels_magic_number}) is not 2049')

            n_images = int.from_bytes(images.read(4), byteorder='big')
            n_labels = int.from_bytes(labels.read(4), byteorder='big')
            if n_images != n_labels:
                raise ValueError(f'number of images ({n_images}) and number of labels ({n_labels}) are different')

            n_rows = int.from_bytes(images.read(4), byteorder='big')
            n_columns = int.from_bytes(images.read(4), byteorder='big')

            x = np.frombuffer(
                images.read(n_images * n_rows * n_columns),
                dtype=np.uint8
            ).astype(np.float32).reshape(n_images, n_rows * n_columns) / 255
            y = np.array([
                _number2vec(number)
                for number in np.frombuffer(labels.read(n_labels), dtype=np.uint8).astype(np.int64)
            ])
        return x, y

    train_x, train_y = _load_data_set(FILE_NAMES[0], FILE_NAMES[1])
    test_x, test_y = _load_data_set(FILE_NAMES[0], FILE_NAMES[1])

    return train_x, train_y, test_x, test_y

def main() -> None:
    """

    """
    args = parse_args()

    # Download data
    download_mnist()

    # Load data
    print('Loading MNIST data: ', end='')
    train_x, train_y, test_x, test_y = load_mnist()
    print('done\n')

    # Load or create estimator
    if args.input_model_path:
        estimator = mlp.MultilayerPerceptron.load(args.input_model_path)
    elif args.output_model_path:
        estimator = mlp.MultilayerPerceptron(inputs=784, units=[300, 50, 10])
    print(f'Model summary:\n{estimator}\n')

    # Train estimator if it is new
    if args.output_model_path:
        print('Training:')
        estimator.train(
            train_x[:10000],
            train_y[:10000],
            batch_size=32,
            epochs=5,
            learning_rate=0.01,
            momentum=0.6,
            shuffle=True,
            metrics_name='categorical_accuracy',
            early_stopping=False
        )
        estimator.train(
            train_x[:55000],
            train_y[:55000],
            validation_x=train_x[55000:],
            validation_y=train_y[55000:],
            batch_size=20,
            epochs=10,
            learning_rate=0.01,
            momentum=0.7,
            shuffle=True,
            metrics_name='categorical_accuracy',
            early_stopping=True
        )
        print()

    # Test estimator
    train_acc = estimator.evaluate(train_x[:55000], train_y[:55000], metrics_name='categorical_accuracy')
    train_mse = estimator.evaluate(train_x[:55000], train_y[:55000], metrics_name='mean_squared_error')
    val_acc = estimator.evaluate(train_x[55000:], train_y[55000:], metrics_name='categorical_accuracy')
    val_mse = estimator.evaluate(train_x[55000:], train_y[55000:], metrics_name='mean_squared_error')
    test_acc = estimator.evaluate(test_x, test_y, metrics_name='categorical_accuracy')
    test_mse = estimator.evaluate(test_x, test_y, metrics_name='mean_squared_error')
    print(
        f'Evaluation results:\n'
        f'  train accuracy:      {train_acc:.4f}\n'
        f'  train mse:           {train_mse:.4f}\n'
        f'  validation accuracy: {val_acc:.4f}\n'
        f'  validation mse:      {val_mse:.4f}\n'
        f'  test accuracy:       {test_acc:.4f}\n'
        f'  test mse:            {test_mse:.4f}\n'
    )

    # Save model if it is new
    if args.output_model_path:
        estimator.save(args.output_model_path)
        print(f'Saved model to {args.output_model_path.absolute()}')


if __name__ == '__main__':
    main()
