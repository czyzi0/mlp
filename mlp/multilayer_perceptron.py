"""Module with multilayer perceptron definition.

"""

import json
import pathlib
import time
from copy import deepcopy
from typing import List, Optional

import numpy as np

from .activation import _ACTIVATION_DICT
from .metrics import _METRICS_DICT
from .utils import _chunked, _unison_shuffle, _progress_bar


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

    def __init__(self, inputs: int, units: List[int], activations: Optional[List[str]] = None):
        if not activations:
            activations = ['sigmoid' for _ in range(len(units))]
        layer_shapes = zip((inputs, *units)[1:], (inputs, *units)[:-1])

        self._model = {
            'activations': [_ACTIVATION_DICT[activation_name] for activation_name in activations],
            'layers': [np.random.rand(units_num, inputs_num + 1) * 2 - 1 for units_num, inputs_num in layer_shapes],
        }

    def evaluate(self, x: np.ndarray, y_true: np.ndarray, metrics_name: str) -> float:
        """Evaluate estimator.

        Args:
            x: Inputs for evaluation (2d).
            y_true: Ground truth (correct) labels (2d).
            metrics_name: Evaluation metrics name.

        Returns:
            Score for given metrics.

        """
        y_pred = self.predict(np.array(x, ndmin=2))
        metrics = _METRICS_DICT[metrics_name]
        return metrics(y_pred, np.array(y_true, ndmin=2))[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict for given inputs.

        Args:
            x: Inputs for prediction (2d).

        Returns:
            Estimator's prediction (2d).

        """
        y = self._bias_extend(np.array(x, ndmin=2))
        for layer, activation in zip(self._model['layers'], self._model['activations']):
            y = self._bias_extend(activation(np.matmul(layer, y.T).T))
        y = self._bias_reduce(y)
        return y

    def train(
            self,
            train_x: np.ndarray,
            train_y: np.ndarray,
            validation_x: Optional[np.ndarray] = None,
            validation_y: Optional[np.ndarray] = None,
            batch_size: int = 32,
            epochs: int = 20,
            learning_rate: float = 0.01,
            momentum: float = 0.9,
            shuffle: bool = True,
            metrics_name: str = 'mean_squared_error',
            early_stopping: bool = True,
            verbose: bool = True
        ) -> None:
        """Train estimator on given data.

        Args:
            train_x: Training input data (2d).
            train_y: Training labels (2d).
            validation_x: Validation input data (2d).
            validation_y: Validation labels (2d).
            batch_size: Number of vectors in single batch.
            epochs: Number of training epochs.
            learning_rate: Learning rate for model.
            momentum: Momentum or model.
            shuffle: Flag for shuffling input data every epoch.
            metrics_name: Name of metrics to evaluate model with.
            early_stopping: Flag for early stopping.
            verbose: Flag for training verbosity.

        """
        # pylint: disable=too-many-arguments,too-many-locals,unused-variable
        # Put vectors into numpy arrays and set flag for validation
        train_x = np.array(train_x, ndmin=2)
        train_y = np.array(train_y, ndmin=2)
        validation = validation_x and validation_y
        if validation:
            validation_x = np.array(validation_x, ndmin=2)
            validation_y = np.array(validation_y, ndmin=2)

        delta_weights = [np.zeros_like(layer) for layer in self._model['layers']]

        def _train_batch(batch_x: np.ndarray, batch_y_true: np.ndarray) -> None:
            """Train single batch.

            Args:
                batch_x: Input data for batch.
                batch_y_true: Labels for batch.

            """
            nonlocal delta_weights
            # Feedforward
            layer_args = [self._bias_extend(batch_x)]
            activation_args = []
            for index, (layer, activation) in enumerate(zip(self._model['layers'], self._model['activations'])):
                activation_args.append(np.matmul(layer, layer_args[index].T).T)
                layer_args.append(self._bias_extend(activation(activation_args[index])))
            batch_y_pred = self._bias_reduce(layer_args.pop())
            # Backpropagate
            last_activation = self._model['activations'][-1]
            errors = [(batch_y_pred - batch_y_true) * last_activation(activation_args.pop(), derivative=True)]
            for index, (layer, activation) in enumerate(zip(
                    reversed(self._model['layers']),
                    reversed(self._model['activations'][:-1])
                )):
                errors.append(
                    np.matmul(self._bias_reduce(layer).T, errors[index].T).T *
                    activation(activation_args.pop(), derivative=True)
                )
            errors = list(reversed(errors))
            # Update weigths
            delta_weights = [
                -learning_rate * np.matmul(layer_arg.T, error).T + momentum * delta_weight
                for layer_arg, error, delta_weight in zip(layer_args, errors, delta_weights)
            ]
            for layer, delta_weight in zip(self._model['layers'], delta_weights):
                layer += delta_weight

        metrics = _METRICS_DICT[metrics_name]
        train_metrics, validation_metrics, comparison_metrics = None, None, None

        def _evaluate() -> None:
            """Evaluate model.

            """
            nonlocal train_metrics, validation_metrics, comparison_metrics

            train_metrics, comparison_metrics = metrics(self.predict(train_x), train_y)
            if validation:
                validation_metrics, comparison_metrics = metrics(self.predict(validation_x), validation_y)

        def _print_results() -> None:
            """Print epoch results.

            """
            if validation:
                print(
                    f'duration: {duration:.2f}s - train_{metrics_name}: {train_metrics:.4f} '
                    f'- validation_{metrics}: {validation_metrics:.4f}',
                    end=''
                )
            else:
                print(
                    f'duration: {duration:.2f}s - train_{metrics_name}: {train_metrics:.4f}',
                    end=''
                )
            if early_stopping and was_better:
                print(' <<')
            else:
                print('')

        # Calculate base metrics
        _evaluate()

        if early_stopping:
            best_metrics = comparison_metrics
            best_model = deepcopy(self._model)

        for epoch_index in range(epochs):

            if verbose:
                start_time = time.time()
                print(f'Epoch {epoch_index+1}/{epochs}')

            if shuffle:
                train_x, train_y = _unison_shuffle(train_x, train_y)

            # Train batches
            for batch_x, batch_y in _progress_bar(
                    zip(_chunked(train_x, batch_size), _chunked(train_y, batch_size)),
                    total=len(train_x),
                    step=batch_size,
                    verbose=verbose
                ):
                _train_batch(batch_x, batch_y)

            # Calculate metrics
            _evaluate()

            was_better = False
            if early_stopping and comparison_metrics >= best_metrics:
                was_better = True
                best_metrics = comparison_metrics
                best_model = deepcopy(self._model)

            if verbose:
                duration = time.time() - start_time
                _print_results()

        if early_stopping:
            self._model = best_model

    def save(self, file_path: pathlib.Path) -> None:
        """Save model to file.

        Args:
            file_path: Path to file to which model will be dumped.

        """
        model = {
            'activations': [activation._name for activation in self._model['activations']],
            'layers': [layer.tolist() for layer in self._model['layers']],
        }
        with open(pathlib.Path(file_path), 'w') as model_file:
            json.dump(model, model_file, indent=4, sort_keys=True)

    @staticmethod
    def load(file_path: pathlib.Path) -> 'MultilayerPerceptron':
        """Load model from file.

        Args:
            file_path: Path to model file.

        """
        with open(pathlib.Path(file_path), 'r') as model_file:
            model = json.loads(model_file.read())
            multilayer_perceptron = MultilayerPerceptron(inputs=1, units=[1,])
            multilayer_perceptron._model = {
                'activations': [_ACTIVATION_DICT[activation_name] for activation_name in model['activations']],
                'layers': [np.array(layer, ndmin=2) for layer in model['layers']],
            }
            return multilayer_perceptron

    @staticmethod
    def _bias_extend(array: np.ndarray) -> np.ndarray:
        """Extend data with ones.

        Method extends each vector in array with additional one at the end.

        Args:
            array: Array to extend.

        Returns:
            Array extended with ones.

        """
        return np.hstack((array, np.ones((array.shape[0], 1))))

    @staticmethod
    def _bias_reduce(array: np.ndarray) -> np.ndarray:
        """Reduce data.

        Method removes last element from each vector in array.

        Args:
            array: Array to reduce.

        Returns:
            Reduced array.

        """
        return array[:, :-1]
