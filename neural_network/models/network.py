import numpy as np


class NeuralNetwork:
    """Simple neural network class."""

    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, gradient):
        for layer in reversed(self.layers):
            if hasattr(layer, 'weights'):
                gradient = layer.backward(gradient, self.learning_rate)
            else:
                gradient = layer.backward(gradient)

    def train(self, X, y, loss_function, epochs=1000):

        for epoch in range(epochs):

            # Forward propagation
            predictions = self.forward(X)

            # Compute loss
            loss = loss_function.forward(predictions, y)

            # Backward propagation
            gradient = loss_function.backward(predictions, y)
            self.backward(gradient)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {loss:.4f}")

    def predict(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)