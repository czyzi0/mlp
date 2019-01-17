"""Module with multilayer perceptron definition.

"""

import json
import pathlib
import time
from copy import deepcopy
from typing import List, Optional

import numpy as np

from .activations import _ACTIVATION_DICT
from .metrics import _METRICS_DICT
from .utils import chunked, progress_bar, unison_shuffle


class MultilayerPerceptron:
    """Multilayer perceptron estimator.

    Args:
        inputs: Number of inputs to multilayer perceptron.
        units: List of numbers of units in consecutive layers of perceptron.
        activations: List of names of activation functions for consecutive
                     layers. If not given all layers will have sigmoid
                     activation functions.

    Attributes:
        _model: Model data for estimator.

    """

    def __init__(
            self, inputs: int, units: List[int], activations: Optional[List[str]] = None
        ) -> None:
        if activations is None:
            activations = ['sigmoid' for _ in range(len(units))]
        else:
            if len(units) != len(activations):
                raise ValueError('length of units and activations are different')
            for activation in activations:
                if activation not in _ACTIVATION_DICT:
                    raise ValueError(f'{activation} not available')

        layer_shapes = zip((inputs, *units)[1:], (inputs, *units)[:-1])
        self._activations: List[str] = activations
        self._layers: List[np.ndarray] = [
            np.random.rand(units_num, inputs_num + 1) * 2 - 1
            for units_num, inputs_num in layer_shapes]

    def __str__(self) -> str:
        """Creates model summary in form of string.

        Returns:
            Model summary in form of string.

        """
        text = f'{"_"*48}\n{"inputs":<16}{"units(outputs)":<16}{"activation":<16}\n{"="*48}'
        for layer, activation in zip(self._layers, self._activations):
            text += f'\n{layer.shape[1]-1:<16}{layer.shape[0]:<16}{activation:<16}\n{"_"*48}'
        return text

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts outputs for given inputs.

        Args:
            x: Inputs for prediction in form of numpy array of vectors.

        Returns:
            Prediction in form of numpy array of vectors.

        """
        self._validate_x(x)

        y = _bias_extend(np.array(x, ndmin=2))
        for layer, activation in zip(self._layers, self._activations):
            y = _bias_extend(_ACTIVATION_DICT[activation](np.matmul(layer, y.T).T))
        y = _bias_reduce(y)
        return y

    def train(
            self,
            train_x: np.ndarray,
            train_y: np.ndarray,
            val_x: Optional[np.ndarray] = None,
            val_y: Optional[np.ndarray] = None,
            batch_size: int = 32,
            epochs: int = 20,
            learning_rate: float = 0.01,
            momentum: float = 0.9,
            shuffle: bool = True,
            metrics: str = 'mse',
            early_stopping: bool = True,
            verbose: bool = True
        ) -> None:
        """Trains estimator on given data.

        Args:
            train_x: Training input data (2d).
            train_y: Training labels (2d).
            val_x: Validation input data (2d).
            val_y: Validation labels (2d).
            batch_size: Number of vectors in single batch.
            epochs: Number of training epochs.
            learning_rate: Learning rate for model.
            momentum: Momentum or model.
            shuffle: Flag for shuffling input data every epoch.
            metrics: Name of metrics to evaluate model with.
            early_stopping: Flag for early stopping.
            verbose: Flag for training verbosity.

        """
        # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
        # Check if metrics is available
        if metrics not in _METRICS_DICT:
            raise ValueError(f'{metrics} is not available')
        # Check if validation data provided
        validate = val_x is not None and val_y is not None
        # Validate data
        self._validate_x(train_x)
        self._validate_y(train_y)
        if validate:
            self._validate_x(val_x)
            self._validate_y(val_y)
        # Print training summary
        if verbose:
            print(f'Train on {len(train_x)} samples', end='')
            if validate and val_x is not None:    # check to suppress mypy error
                print(f' - Validate on {len(val_x)} samples')
            else:
                print()
        # Calculate base scores
        train_score, compare_score = _METRICS_DICT[metrics](self.predict(train_x), train_y)
        if validate:
            val_score, compare_score = _METRICS_DICT[metrics](self.predict(val_x), val_y)
        # Save best model and score
        if early_stopping:
            best_score = compare_score
            best_model = {
                'activations': deepcopy(self._activations),
                'layers': deepcopy(self._layers)}

        deltas = [np.zeros_like(layer) for layer in self._layers]
        for epoch_i in range(epochs):
            if verbose:
                start_time = time.time()
                print(f'Epoch {epoch_i+1}/{epochs}')
            if shuffle:
                train_x, train_y = unison_shuffle(train_x, train_y)

            for batch_x, batch_y in progress_bar(
                    zip(chunked(train_x, batch_size), chunked(train_y, batch_size)),
                    total=len(train_x),
                    step=batch_size,
                    verbose=verbose):
                # Feedforward and save results
                # `inputs` -> layer -> `outputs` -> activation
                inputs, outputs = [_bias_extend(batch_x)], []
                for i, (layer, activation) in enumerate(zip(self._layers, self._activations)):
                    outputs.append(np.matmul(layer, inputs[i].T).T)
                    inputs.append(_bias_extend(_ACTIVATION_DICT[activation](outputs[i])))
                batch_y_pred = _bias_reduce(inputs.pop())
                # Backpropagate
                errors = [
                    (batch_y_pred - batch_y)
                    * _ACTIVATION_DICT[self._activations[-1]](outputs.pop(), derivative=True)]
                for i, (layer, activation) in enumerate(zip(
                        reversed(self._layers), reversed(self._activations[:-1]))):
                    errors.append(
                        np.matmul(_bias_reduce(layer).T, errors[i].T).T
                        * _ACTIVATION_DICT[activation](outputs.pop(), derivative=True))
                errors = list(reversed(errors))
                # Update weigths
                deltas = [
                    -learning_rate * np.matmul(input_.T, error).T + momentum * delta
                    for input_, error, delta in zip(inputs, errors, deltas)]
                for layer, delta in zip(self._layers, deltas):
                    layer += delta

            # Calculate scores
            train_score, compare_score = _METRICS_DICT[metrics](self.predict(train_x), train_y)
            if validate:
                val_score, compare_score = _METRICS_DICT[metrics](self.predict(val_x), val_y)
            # Save best model and score
            was_better = False
            if early_stopping and compare_score >= best_score:
                was_better = True
                best_score = compare_score
                best_model = {
                    'activations': deepcopy(self._activations),
                    'layers': deepcopy(self._layers)}
            # Print epoch summary
            if verbose:
                duration = time.time() - start_time
                # Print results
                if validate:
                    print(
                        f'duration: {duration:.2f}s - train_{metrics}: {train_score:.4f} '
                        f'- val_{metrics}: {val_score:.4f}',
                        end='')
                else:
                    print(f'duration: {duration:.2f}s - train_{metrics}: {train_score:.4f}', end='')
                if early_stopping and was_better:
                    print(' <<')
                else:
                    print()

        # Restore best model
        if early_stopping:
            self._activations = best_model['activations']
            self._layers = best_model['layers']

    def save(self, file_path: pathlib.Path) -> None:
        """Saves model to file.

        Args:
            file_path: Path to file to which model will be saved.

        """
        model = {
            'activations': self._activations,
            'layers': [layer.tolist() for layer in self._layers]}
        with open(pathlib.Path(file_path), 'w') as model_file:
            json.dump(model, model_file, indent=4, sort_keys=True)

    @staticmethod
    def load(file_path: pathlib.Path) -> 'MultilayerPerceptron':
        """Loads model from file.

        Args:
            file_path: Path to model file.

        """
        with open(pathlib.Path(file_path), 'r') as model_file:
            model = json.loads(model_file.read())
            multilayer_perceptron = MultilayerPerceptron(inputs=1, units=[1,])
            multilayer_perceptron._activations = model['activations']
            multilayer_perceptron._layers = [np.array(layer, ndmin=2) for layer in model['layers']]
            return multilayer_perceptron

    def _validate_x(self, x: np.ndarray) -> None:
        """Validates x data regarding model.

        Args:
            x: Data to validate.

        Raises:
            ValueError: In case of failed validation.

        """
        if x.ndim != 2:
            raise ValueError('x should have 2 dims')
        if x.shape[1] != self._layers[0].shape[1] - 1:
            raise ValueError(
                f'invalid x shape (None, {x.shape[1]}), '
                f'should be (None, {self._layers[0].shape[1] - 1})')

    def _validate_y(self, y: np.ndarray) -> None:
        """Validates y data regarding model.

        Args:
            y: Data to validate.

        Raises:
            ValueError: In case of failed validation.

        """
        if y.ndim != 2:
            raise ValueError('y should have 2 dims')
        if y.shape[1] != self._layers[-1].shape[0]:
            raise ValueError(
                f'invalid y shape (None, {y.shape[1]}), '
                f'should be (None, {self._layers[-1].shape[0]})')


def _bias_extend(array: np.ndarray) -> np.ndarray:
    """Extends data with ones.

    Method extends each vector in array with additional one at the end.

    Args:
        array: Array to extend.

    Returns:
        Array extended with ones.

    """
    return np.hstack((array, np.ones((array.shape[0], 1))))


def _bias_reduce(array: np.ndarray) -> np.ndarray:
    """Reduces data.

    Method removes last element from each vector in array.

    Args:
        array: Array to reduce.

    Returns:
        Reduced array.

    """
    return array[:, :-1]
