import  numpy as np
class BinaryCrossEntropy:
    """Loss for binary classification."""

    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

    def backward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / len(y_true)


class CategoricalCrossEntropy:
    """Loss for multi-class classification."""

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidences = y_pred[range(samples), y_true]
        return np.mean(-np.log(correct_confidences))

    def backward(self, y_pred, y_true):
        samples = len(y_pred)

        dinputs = y_pred.copy()
        dinputs[range(samples), y_true] -= 1
        dinputs = dinputs / samples

        return dinputs