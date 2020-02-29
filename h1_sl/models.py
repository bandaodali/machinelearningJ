""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures

    def fit(self, *, X, y, lr):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
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


class MCModel(Model):
    """ A multiclass model abstraction.
    It wants to know, up front:
        - How many features in the data
        - How many classes in the data
    """

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures)
        self.num_classes = nclasses



class Perceptron(Model):

    def __init__(self, *, nfeatures):
        super().__init__(nfeatures=nfeatures)
        self.W = np.zeros((nfeatures, 1))

    def fit(self, *, X, y, lr):
        X = X.toarray()
        N = X.shape[0]
        for i in range(N):
            y_value = X[i].dot(self.W)
            if y_value[0] > 0 and y[i] == 0:
                self.W -= lr * X[i].reshape(self.W.shape)
            if y_value[0] <= 0 and y[i] == 1:
                self.W += lr * X[i].reshape(self.W.shape)

    def predict(self, X):
        X = self._fix_test_feats(X)
        X = X.toarray()
        N = X.shape[0]
        y = np.zeros(N)
        for i in range(N):
            y[i] = 1 if X[i].dot(self.W)[0] > 0 else 0
        return y


class WeightedPerceptron(Perceptron):

    def __init__(self, *, nfeatures, alpha_pos, alpha_neg):
        super().__init__(nfeatures=nfeatures)
        self.W = np.zeros((nfeatures, 1))
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def fit(self, *, X, y, lr):
        X = X.toarray()
        N = X.shape[0]
        for i in range(N):
            y_value = X[i].dot(self.W)
            if y_value[0] > 0 and y[i] == 0:
                self.W -= lr * self.alpha_pos * X[i].reshape(self.W.shape)
            if y_value[0] <= 0 and y[i] == 1:
                self.W += lr * self.alpha_neg * X[i].reshape(self.W.shape)



class MCPerceptron(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        X = X.toarray()
        N = X.shape[0]
        for i in range(N):
            y_value = X[i].dot(self.W.T)
            y_max = np.argmax(y_value)
            if y_max != y[i]:
                self.W[y[i]] += lr * X[i].T
                self.W[y_max] -= lr * X[i].T

    def predict(self, X):
        X = self._fix_test_feats(X)
        X = X.toarray()
        N = X.shape[0]
        y = np.zeros(N)
        for i in range(N):
            y[i] = np.argmax(X[i].dot(self.W.T))
        return y

    def score(self, x_i):
        # TODO: Implement this!
        raise Exception("You must implement this method!")


class Logistic(Model):

    def __init__(self, *, nfeatures):
        super().__init__(nfeatures=nfeatures)
        self.W = np.zeros((nfeatures, 1))

    def sigmoid(self, inx):
            if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
                return 1.0 / (1 + np.exp(-inx))
            else:
                return np.exp(inx) / (1 + np.exp(inx))

    def fit(self, *, X, y, lr):
        X = X.toarray()
        N = X.shape[0]
        for j in range(self.W.shape[0]):
            dW = 0
            for i in range(N):
                dW += y[i] * self.sigmoid(-1 * X[i].dot(self.W)[0]) * X[i][j] + (1 - y[i]) * self.sigmoid(X[i] .dot(self.W)[0]) * (-1 * X[i][j])
            #print(j, dW)
            self.W[j][0] += lr * dW

    def predict(self, X):
        X = self._fix_test_feats(X)
        X = X.toarray()
        N = X.shape[0]
        y = np.zeros(N)
        for i in range(N):
            y[i] = 1 if self.sigmoid(X[i].dot(self.W)[0]) > 0.5 else 0
        return y
