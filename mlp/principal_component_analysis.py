"""Module with principal component analysis definition.

"""

import json

import numpy as np


class PrincipalComponentAnalysis:
    """Principal Component Analysis.

    Attributes:
        _model: 

    """

    def __init__(self):
        self._model = np.identity(1)

    def train(self, train_vectors_in, dims):
        train_vecs_in = np.array(train_vectors_in, ndmin=2).T

        covariance_matrix = np.cov(train_vecs_in)

        eigen_values, eigen_vecs = np.linalg.eig(covariance_matrix)
        eigen_pairs = [
            (np.abs(eigen_values[i]), np.atleast_2d(eigen_vecs[:, i]).T)
            for i in range(len(eigen_values))
        ]
        eigen_pairs.sort(key=lambda x: x[0], reverse=True)

        self._model = np.hstack([eigen_pairs[i][1] for i in range(dims)])

    def transform(self, vectors_in):
        vecs_pred = np.matmul(self._model.T, np.array(vectors_in, ndmin=2).T).T
        if isinstance(vectors_in, type([])):
            return vecs_pred.tolist()
        return vecs_pred

    def save(self, file_path):
        model = self._model.tolist()
        with open(file_path, 'w') as file_:
            json.dump(model, file_, indent=4, sort_keys=True)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as file_:
            model = json.loads(file_.read())
            pca = PrincipalComponentAnalysis()
            # pylint: disable=protected-access
            pca._model = np.array(model, ndmin=2)
            return pca
