"""Classification example using Iris data set."""

import argparse
import pathlib
from typing import List

import numpy as np

from .. import MultilayerPerceptron
from ..data import load_mnist
from ..metrics import accuracy, mean_squared_error
from ..utils import argmax


def parse_args() -> argparse.Namespace:
    """Parses arguments.

    Returns:
        Parsed arguments.

    """
    parser = argparse.ArgumentParser(description='Train and evaluate model on MNIST data.')
    parser.add_argument('-o', '--output', type=pathlib.Path, help='path to save model at')
    return parser.parse_args()


def main() -> None:
    """Runs example."""
    # pylint: disable=too-many-locals
    args = parse_args()

    train_x, train_y, test_x, test_y = load_mnist()
    val_x, val_y = test_x[:5000], test_y[:5000]
    test_x, test_y = test_x[5000:], test_y[5000:]

    train_x = train_x / 255
    val_x = val_x / 255
    test_x = test_x / 255

    def _encode(label: int) -> List[float]:
        encoding = [0.0] * 10
        encoding[label] = 1.0
        return encoding

    train_y = np.array([_encode(label) for label in train_y])
    val_y = np.array([_encode(label) for label in val_y])
    test_y = np.array([_encode(label) for label in test_y])

    model = MultilayerPerceptron(inputs=784, units=[300, 50, 10])
    print(f'Model summary:\n{model}\n')

    model.train(
        train_x[:20000], train_y[:20000],
        batch_size=100, epochs=5, learning_rate=0.01, momentum=0.6,
        shuffle=True, metrics='acc', early_stopping=False)
    model.train(
        train_x, train_y, val_x=val_x, val_y=val_y,
        batch_size=50, epochs=10, learning_rate=0.01, momentum=0.6,
        shuffle=True, metrics='acc', early_stopping=True)
    print()

    # Evaluate model
    train_y_pred = model.predict(train_x)
    train_acc = accuracy(argmax(train_y_pred), train_y)
    train_mse = mean_squared_error(train_y_pred, train_y)

    val_y_pred = model.predict(val_x)
    val_acc = accuracy(argmax(val_y_pred), val_y)
    val_mse = mean_squared_error(val_y_pred, val_y)

    test_y_pred = model.predict(test_x)
    test_acc = accuracy(argmax(test_y_pred), test_y)
    test_mse = mean_squared_error(test_y_pred, test_y)

    print(
        f'Evaluation results:\n'
        f'  train accuracy: {train_acc:.4f}\n'
        f'  train mse:      {train_mse:.4f}\n'
        f'  val accuracy:   {val_acc:.4f}\n'
        f'  val mse:        {val_mse:.4f}\n'
        f'  test accuracy:  {test_acc:.4f}\n'
        f'  test mse:       {test_mse:.4f}\n')

    # Save model
    if args.output:
        model.save(args.output)
        print(f'Saved model to {args.output.absolute()}')


if __name__ == '__main__':
    main()
