# mlp
Multilayer perceptron implemented using NumPy with usage examples.

## Setup
These instructions will help you get a copy of the project up and running. First clone repository then:

- if you want to just use package, type:
  ```bash
  python setup.py install
  ```
- if you want to run examples/develop package, type:
  ```bash
  conda env create -f environment.yml
  conda activate mlp
  python setup.py develop
  conda deactivate
  ```

## Development

### Guide
[This](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) convention is followed for documenting with exception for type hints (for documenting those [PEP 484](https://www.python.org/dev/peps/pep-0484/) is followed).

Commit messages are structured in accordance with:
- Write the summary line and description in the imperative mode. Start the line with "Fix", "Add", "Change" instead of "Fixed", "Added", "Changed".
- Always leave the second line blank.
- Don't end the summary with a period.

### Tests
Test are ran with __pytest__. To run them type:
```bash
pytest --cov=mlp/
```

### Code analysis
Code analysis is performed with __Pylint__. To run it type:
```bash
pylint mlp/
```
