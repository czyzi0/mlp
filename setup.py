import pathlib

from setuptools import setup, find_packages


with open(pathlib.Path(__file__).parent / 'README.md', 'r') as long_desc_file:
    LONG_DESCRIPTION = long_desc_file.read()

setup(
    name='mlp',
    version='0.0.0',
    description='Multilayer perceptron implemented using NumPy with usage examples',
    long_description=LONG_DESCRIPTION,
    author='czyzi0',
    author_email='czyznikiewicz.mateusz@gmail.com',
    url='https://github.com/czyzi0/mlp',
    license='MIT',
    keywords='machine-learning multilayer-perceptron numpy',
    packages=find_packages(),
    python_requires='>=3.6.0',
    install_requires=['numpy>=1.14.0']
)
