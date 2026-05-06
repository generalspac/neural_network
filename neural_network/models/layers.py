import numpy as np


class Dense:
    """Fully connected layer."""

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, X):
        self.inputs = X
        self.output = np.dot(X, self.weights) + self.biases
        return self.output

    def backward(self, dvalues, learning_rate):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        dinputs = np.dot(dvalues, self.weights.T)

        # SGD update
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

        return dinputs