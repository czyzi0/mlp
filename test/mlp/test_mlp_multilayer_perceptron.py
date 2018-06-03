import numpy as np
import pytest

from mlp.multilayer_perceptron import MultilayerPerceptron
from mlp.metrics import _METRICS_DICT


class TestMultilayerPerceptron:

    @pytest.mark.parametrize('inputs, units, activations, vecs_num', [
        (4, [3,], ['relu',], 15),
        (5, [3, 2, 10], None, 13),
        (784, [150, 50, 10], ['relu', 'relu', 'tanh'], 29),
        (3, [3, 2], None, 986),
        (2, (10, 1), ['softplus', 'sigmoid'], 111),
    ])
    def test_predict(self, inputs, units, activations, vecs_num):
        multilayer_perceptron = MultilayerPerceptron(inputs=inputs, units=units, activations=activations)
        assert multilayer_perceptron != None

        x = np.random.rand(vecs_num, inputs)
        y = multilayer_perceptron.predict(x)
        assert (vecs_num, units[-1]) == y.shape

    @pytest.mark.parametrize('inputs, units, x, y_true, metrics_name', [
        (
            1,
            [1,],
            np.array([
                [1.0],
                [0.3],
                [0.2],
            ]),
            np.array([
                [1.0],
                [7.0],
                [0.0],
            ]),
            'binary_accuracy',
        ),
        (
            1,
            [2,],
            np.array([
                [1.0],
                [0.3],
                [0.2],
            ]),
            np.array([
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]),
            'categorical_accuracy',
        ),
        (
            1,
            [2,],
            np.array([
                [1.0],
                [0.3],
                [0.2],
            ]),
            np.array([
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]),
            'mean_squared_error',
        ),
    ])
    def test_evaluate(self, inputs, units, x, y_true, metrics_name):
        multilayer_perceptron = MultilayerPerceptron(inputs=inputs, units=units)
        y_pred = multilayer_perceptron.predict(x)

        metrics = _METRICS_DICT[metrics_name]
        expected_output, _ = metrics(y_pred, y_true)

        output = multilayer_perceptron.evaluate(x, y_true, metrics_name=metrics_name)

        assert expected_output == pytest.approx(output)

    @pytest.mark.parametrize('train_y', [
        # XOR
        ([[0], [1], [1], [0]]),
        # AND
        ([[0], [0], [0], [1]]),
        # OR
        ([[0], [1], [1], [1]]),
        # NAND
        ([[1], [1], [1], [0]]),
        # NOR
        ([[1], [0], [0], [0]]),
        # XNOR
        ([[1], [0], [0], [1]]),
    ])
    def test_train_logic_functions(self, train_y):
        multilayer_perceptron = MultilayerPerceptron(inputs=2, units=[5, 1])

        train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 1000)
        train_y = np.array(train_y * 1000)

        multilayer_perceptron.train(train_x, train_y, verbose=False)

        for x, y_true in zip(train_x[:4], train_y[:4]):
            if y_true == [0]:
                assert 0.1 > multilayer_perceptron.predict(np.array(x, ndmin=2))[0][0]
            elif y_true == [1]:
                assert 0.9 < multilayer_perceptron.predict(np.array(x, ndmin=2))[0][0]

    @pytest.mark.parametrize('inputs, units, activations', [
        (3, [2, 5], None),
        (2, [1,], ['relu',]),
    ])
    def test_save_load(self, tmpdir, inputs, units, activations):
        file_path = tmpdir.join('model.json')

        mlp_before = MultilayerPerceptron(inputs=inputs, units=units, activations=activations)
        mlp_before.save(file_path)

        mlp_after = MultilayerPerceptron.load(file_path)

        x = np.random.rand(15, inputs)
        y_before = mlp_before.predict(x)
        y_after = mlp_after.predict(x)

        assert  np.allclose(y_before, y_after)

    @pytest.mark.parametrize('array, expected_output', [
        (
            np.array([
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],
            ]),
            np.array([
                [0.0, 0.0, 0.0, 1.0],
                [2.0, 2.0, 2.0, 1.0],
            ]),
        ),
        (
            np.array([
                [0.0, 0.0],
            ]),
            np.array([
                [0.0, 0.0, 1.0],
            ]),
        ),
    ])
    def test__bias_extend(self, array, expected_output):
        output = MultilayerPerceptron._bias_extend(array)
        assert np.allclose(expected_output, output)

    @pytest.mark.parametrize('array, expected_output', [
        (
            np.array([
                [0.0, 0.0, 0.0, 1.0],
                [2.0, 2.0, 2.0, 1.0],
            ]),
            np.array([
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],
            ]),
        ),
        (
            np.array([
                [0.0, 0.0, 1.0],
            ]),
            np.array([
                [0.0, 0.0],
            ]),
        ),
    ])
    def test__bias_reduce(self, array, expected_output):
        output = MultilayerPerceptron._bias_reduce(array)
        assert np.allclose(expected_output, output)