"""Classification example using Iris data set."""

import argparse
import pathlib

import numpy as np

from .. import MultilayerPerceptron
from ..data import load_iris
from ..metrics import accuracy, mean_squared_error
from ..utils import argmax


def parse_args() -> argparse.Namespace:
    """Parses arguments.

    Returns:
        Parsed arguments.

    """
    parser = argparse.ArgumentParser(description='Train and evaluate model on Iris data.')
    parser.add_argument('-o', '--output', type=pathlib.Path, help='path to save model at')
    return parser.parse_args()


def main() -> None:
    """Runs example."""
    args = parse_args()

    train_x, train_y, test_x, test_y = load_iris()

    encode_dict = {
        'setosa': [1.0, 0.0, 0.0],
        'versicolor': [0.0, 1.0, 0.0],
        'virginica': [0.0, 0.0, 1.0]}
    train_y = np.array([encode_dict[label] for label in train_y])
    test_y = np.array([encode_dict[label] for label in test_y])

    model = MultilayerPerceptron(inputs=4, units=[5, 3])
    print(f'Model summary:\n{model}\n')

    model.train(
        train_x, train_y,
        batch_size=1, epochs=10, learning_rate=0.1, momentum=0.7,
        shuffle=True, metrics='acc', early_stopping=True)
    print()

    # Evaluate model
    train_y_pred = model.predict(train_x)
    train_acc = accuracy(argmax(train_y_pred), train_y)
    train_mse = mean_squared_error(train_y_pred, train_y)

    test_y_pred = model.predict(test_x)
    test_acc = accuracy(argmax(test_y_pred), test_y)
    test_mse = mean_squared_error(test_y_pred, test_y)

    print(
        f'Evaluation results:\n'
        f'  train accuracy: {train_acc:.4f}\n'
        f'  train mse:      {train_mse:.4f}\n'
        f'  test accuracy:  {test_acc:.4f}\n'
        f'  test mse:       {test_mse:.4f}\n')

    # Save model
    if args.output:
        model.save(args.output)
        print(f'Saved model to {args.output.absolute()}')


if __name__ == '__main__':
    main()
