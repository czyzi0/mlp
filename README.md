# mlp
Multilayer perceptron implemented using NumPy with usage examples.

## Setup
These instructions will help you get a copy of the project up and running.
First step is to clone repository. After that you need to install library. You can do that with:
```bash
python3.6 setup.py install
```
You can also use _developer mode_ with:
```bash
python3.6 setup.py develop
```

## Development

### Guide
[This](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) convention is followed for documenting with exception for type hints (for documenting those [PEP 484](https://www.python.org/dev/peps/pep-0484/) is followed).

Commit messages are structured in accordance with:
- Write the summary line and description in the imperative mode. Start the line with "Fix", "Add", "Change" instead of "Fixed", "Added", "Changed".
- Always leave the second line blank.
- Don't end the summary with a period.

### Tests
Test are ran with __pytest__. First install it with:
```bash
pip install pytest
pip install pytest-cov
```
Then run tests with:
```bash
pytest --cov=mlp/
```

### Code analysis
Code analysis is performed with __Pylint__. First install it with:
```bash
pip install pylint
```
Then run code analysis with:
```bash
pylint mlp/
```
