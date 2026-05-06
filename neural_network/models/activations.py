import numpy as np


class Sigmoid:
    """Sigmoid activation function."""

    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, dvalues):
        return dvalues * self.output * (1 - self.output)


class ReLU:
    """ReLU activation function."""

    def forward(self, X):
        self.inputs = X
        return np.maximum(0, X)

    def backward(self, dvalues):
        dinputs = dvalues.copy()
        dinputs[self.inputs <= 0] = 0
        return dinputs


class Softmax:
    """Softmax activation for multi-class classification."""

    def forward(self, X):
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        return dvalues