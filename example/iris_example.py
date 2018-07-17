"""Classification example using Iris data set.

"""

import argparse
import csv
import pathlib
from typing import Tuple

import numpy as np
import requests

import mlp


def parse_args() -> argparse.Namespace:
    """Parse arguments.

    Returns:
        Parsed arguments.

    """
    parser = argparse.ArgumentParser(description='Train new model or evaluate existing model on Iris data.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input-model-path', type=pathlib.Path, help='path to the model to be tested')
    group.add_argument('-o', '--output-model-path', type=pathlib.Path, help='path to file to save trained model in')

    return parser.parse_args()


def download_iris(file_path) -> None:
    """Download Iris data.

    Args:
        file_path: File path to which data is saved.

    """
    # pylint: disable=no-member
    if not file_path.exists():
        response = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', stream=True)
        with open(file_path, 'wb') as iris_file:
            for data_chunk in mlp.utils.spinner(response.iter_content(), message='Downloading Iris data: '):
                iris_file.write(data_chunk)


def load_iris(file_path) -> Tuple[np.ndarray, np.ndarray]:
    """Load Iris data.

    Args:
        file_path: Path to data file.

    Returns:
        Two numpy arrays one with inputs(x) and one with expected outputs(y).
        Array shapes are (n, 4) and (n, 3).

    """
    NAME2VEC = {
        'Iris-setosa': [1.0, 0.0, 0.0],
        'Iris-versicolor': [0.0, 1.0, 0.0],
        'Iris-virginica': [0.0, 0.0, 1.0],
    }

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        x, y = [], []
        for row in mlp.utils.spinner((row for row in reader if row), message='Loading Iris data: '):
            x.append([float(number) for number in row[:4]])
            y.append(NAME2VEC[row[4]])
        x = np.array(x)
        y = np.array(y)
    return x, y


def main() -> None:
    """Run example

    """
    args = parse_args()

    IRIS_FILE_PATH = pathlib.Path(__file__).parent / 'data' / 'iris.data'

    # Download data
    download_iris(IRIS_FILE_PATH)

    # Load data
    x, y = load_iris(IRIS_FILE_PATH)

    # Split data into train and test set
    train_x = np.vstack([x[0:40], x[50:90], x[100:140]])
    train_y = np.vstack([y[0:40], y[50:90], y[100:140]])
    test_x = np.vstack([x[40:50], x[90:100], x[140:150]])
    test_y = np.vstack([y[40:50], y[90:100], y[140:150]])
    print()

    # Load or create estimator
    if args.input_model_path:
        estimator = mlp.MultilayerPerceptron.load(args.input_model_path)
    elif args.output_model_path:
        estimator = mlp.MultilayerPerceptron(inputs=4, units=[5, 3])
    print(f'Model summary:\n{estimator}\n')

    # Train estimator if is new
    if args.output_model_path:
        print(f'Training:')
        estimator.train(
            train_x,
            train_y,
            batch_size=1,
            epochs=10,
            learning_rate=0.1,
            momentum=0.7,
            shuffle=True,
            metrics_name='categorical_accuracy',
            early_stopping=True
        )
        print()

    # Test estimator
    train_acc = estimator.evaluate(train_x, train_y, metrics_name='categorical_accuracy')
    train_mse = estimator.evaluate(train_x, train_y, metrics_name='mean_squared_error')
    test_acc = estimator.evaluate(test_x, test_y, metrics_name='categorical_accuracy')
    test_mse = estimator.evaluate(test_x, test_y, metrics_name='mean_squared_error')
    print(
        f'Evaluation results:\n'
        f'  train accuracy: {train_acc:.4f}\n'
        f'  train mse:      {train_mse:.4f}\n'
        f'  test accuracy:  {test_acc:.4f}\n'
        f'  test mse:       {test_mse:.4f}\n'
    )

    # Save model if is new
    if args.output_model_path:
        estimator.save(args.output_model_path)
        print(f'Saved model to {args.output_model_path.absolute()}')


if __name__ == '__main__':
    main()
