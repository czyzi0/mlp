# mlp
Multilayer perceptron implemented using [NumPy](http://www.numpy.org/).

## Setup
These instructions will help you get a copy of the project up and running. First clone repository then:

- if you want to just use package, then install it with:
  ```
  python setup.py install
  ```
- if you want to develop package, install dependencies from `environment.yml` and install package in developer mode. You can do that using [Conda](https://conda.io/docs/):
  ```
  conda env create -f environment.yml
  conda activate mlp
  python setup.py develop
  conda deactivate
  ```

## Examples

### Iris
First example uses [Iris](https://archive.ics.uci.edu/ml/datasets/iris) data set. Here is manual for this example:
```
usage: iris.py [-h] [-o OUTPUT]

Train and evaluate model on Iris data.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        path to save model at
```

To print this manual:
```
python -m mlp.examples.iris -h
```

### MNIST
Second example uses [MNIST](http://yann.lecun.com/exdb/mnist/) data set. Here is manual for this example:
```
usage: mnist.py [-h] [-o OUTPUT]

Train and evaluate model on MNIST data.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        path to save model at
```

To print this manual:
```
python -m mlp.examples.mnist -h
```

## Development

### Style guide
- Docstring are written according to [Google Style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) with exception for type hints (for those [PEP 484](https://www.python.org/dev/peps/pep-0484/) is followed).
- Commit messages are written according to [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/).

### Code analysis
Code analysis is performed with [Pylint](https://www.pylint.org/) and types are checked with [mypy](http://mypy-lang.org/). To run it type:
```
pylint mlp
mypy mlp --ignore-missing-imports
```

### Tests
Test are ran with [pytest](https://docs.pytest.org/en/latest/). To run them type:
```
pytest --cov mlp
```
