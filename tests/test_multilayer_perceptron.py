import numpy as np
import pytest

from mlp.multilayer_perceptron import MultilayerPerceptron, _bias_extend, _bias_reduce
from mlp.metrics import _METRICS_DICT


class TestMultilayerPerceptron:

    @pytest.mark.parametrize('inputs, units, activations', [
        (5, [3, 2], ['sigmoid']),
        (5, [3, 2], ['sigmoid', 'test']),
    ])
    def test___init___raises_error(self, inputs, units, activations):
        with pytest.raises(ValueError):
            multilayer_perceptron = MultilayerPerceptron(
                inputs=inputs, units=units, activations=activations)

    @pytest.mark.parametrize('inputs, units, activations, xpcd_model_summary', [
        (
            5, [3, 2], None,
            (
                '________________________________________________\n'
                'inputs          units(outputs)  activation      \n'
                '================================================\n'
                '5               3               sigmoid         \n'
                '________________________________________________\n'
                '3               2               sigmoid         \n'
                '________________________________________________'
            ),
        ),
        (
            87, [16, 11, 3], ['relu', 'softplus', 'identity'],
            (
                '________________________________________________\n'
                'inputs          units(outputs)  activation      \n'
                '================================================\n'
                '87              16              relu            \n'
                '________________________________________________\n'
                '16              11              softplus        \n'
                '________________________________________________\n'
                '11              3               identity        \n'
                '________________________________________________'
            ),
        ),
    ])
    def test___str__(self, inputs, units, activations, xpcd_model_summary):
        multilayer_perceptron = MultilayerPerceptron(
            inputs=inputs, units=units, activations=activations)
        model_summary = multilayer_perceptron.__str__()
        assert xpcd_model_summary == model_summary

    @pytest.mark.parametrize('inputs, units, activations, x_len', [
        (4, [3,], ['relu',], 15),
        (5, [3, 2, 10], None, 13),
        (784, [150, 50, 10], ['relu', 'relu', 'tanh'], 29),
        (3, [3, 2], None, 986),
        (2, (10, 1), ['softplus', 'sigmoid'], 111),
    ])
    def test_predict(self, inputs, units, activations, x_len):
        multilayer_perceptron = MultilayerPerceptron(
            inputs=inputs, units=units, activations=activations)
        assert multilayer_perceptron != None

        x = np.random.rand(x_len, inputs)
        y = multilayer_perceptron.predict(x)
        assert (x_len, units[-1]) == y.shape

    def test_train_linear_regression(self):
        multilayer_perceptron = MultilayerPerceptron(
            inputs=1, units=[1], activations=['identity'])

        x = np.random.uniform(-1.0, 1.0, (1000, 1))
        y = 2 * x + 5 + np.random.uniform(-0.1, 0.1, x.shape)
        train_x, train_y = x[:800], y[:800]
        val_x, val_y = x[800:], y[800:]

        multilayer_perceptron.train(train_x, train_y, verbose=True)
        multilayer_perceptron.train(train_x, train_y, val_x=val_x, val_y=val_y, verbose=True)

        assert np.allclose(np.array([[2.0, 5.0]]), multilayer_perceptron._layers[0], atol=0.1)

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

    def test_train_raises_error(self):
        multilayer_perceptron = MultilayerPerceptron(inputs=2, units=[5, 1])
        with pytest.raises(ValueError):
            multilayer_perceptron.train([], [], metrics='test')

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

    @pytest.mark.parametrize('inputs, units, x, valid', [
        (
            5,
            [3, 4],
            np.array([
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6]]),
            True,
        ),
        (
            3,
            [3, 3],
            np.array([1, 2, 3]),
            False,
        ),
        (
            3,
            [3, 3],
            np.array([
                [1, 2, 3, 4],
                [2, 3, 4, 5]]),
            False,
        ),
    ])
    def test__validate_x(self, inputs, units, x, valid):
        multilayer_perceptron = MultilayerPerceptron(inputs=inputs, units=units)
        if valid:
            multilayer_perceptron._validate_x(x)
        else:
            with pytest.raises(ValueError):
                multilayer_perceptron._validate_x(x)

    @pytest.mark.parametrize('inputs, units, y, valid', [
        (
            5,
            [3, 4],
            np.array([
                [1, 2, 3, 4],
                [2, 3, 4, 5]]),
            True,
        ),
        (
            3,
            [3, 3],
            np.array([1, 2, 3]),
            False,
        ),
        (
            3,
            [3, 3],
            np.array([
                [1, 2, 3, 4],
                [2, 3, 4, 5]]),
            False,
        ),
    ])
    def test__validate_y(self, inputs, units, y, valid):
        multilayer_perceptron = MultilayerPerceptron(inputs=inputs, units=units)
        if valid:
            multilayer_perceptron._validate_y(y)
        else:
            with pytest.raises(ValueError):
                multilayer_perceptron._validate_y(y)


@pytest.mark.parametrize('array, xpcd_output', [
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
def test__bias_extend(array, xpcd_output):
    output = _bias_extend(array)
    assert np.allclose(xpcd_output, output)


@pytest.mark.parametrize('array, xpcd_output', [
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
def test__bias_reduce(array, xpcd_output):
    output = _bias_reduce(array)
    assert np.allclose(xpcd_output, output)
