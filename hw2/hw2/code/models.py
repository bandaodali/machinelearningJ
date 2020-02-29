""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np

class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures, lmbda):
        self.num_input_features = nfeatures
        self.lmbda = lmbda

    def fit(self, *, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class Pegasos(Model):

    def __init__(self, *, nfeatures, lmbda):
        """
        Args:
            nfeatures: size of feature space
            lmbda: regularizer term (lambda)

        Sets:
            W: weight vector
            t: current iteration
        """
        super().__init__(nfeatures=nfeatures, lmbda=lmbda)
        self.W = np.zeros((nfeatures, 1))
        self.t = 1

    def fit(self, *, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        raise Exception("You must implement this method!")



class KernelPegasos(Model):

    def __init__(self, *, nfeatures, nexamples, lmbda, kernel_degree):
        """
        Args:
            nfeatures: size of feature space (only needed for _fix_test_feats)
            nexamples: size of example space
            lmbda: regularizer term (lambda)
            kernel_degree: polynomial degree for kernel function

        Sets:
            b: beta vector (related to alpha in dual formulation)
            t: current iteration
            kernel_degree: polynomial degree for kernel function
            support_vectors: array of support vectors
            examples_corresp_to_svs: training examples that correspond with support vectors
        """
        super().__init__(nfeatures=nfeatures, lmbda=lmbda)
        self.b = np.zeros(nexamples)
        self.t = 1
        self.kernel_degree = kernel_degree
        self.support_vectors = None
        self.examples_corresp_to_svs = None


    def fit(self, *, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def compute_kernel_matrix(self, X, X_prime=None):
        """ Compute kernel matrix. Index into kernel matrix to evaulate kernel function.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            X_prime: A  compressed sparse row matrix of floats.

        Returns:
            A compressed sparse row matrix of floats with each element representing
            one kernel function evaluation.
        """
        X_prime = X if X_prime is None else X_prime
        # TODO: Implement this!
        raise Exception("You must implement this method!")
