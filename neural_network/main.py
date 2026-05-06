import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.layers import Dense
from models.activations import ReLU, Softmax
from models.losses import CategoricalCrossEntropy
from models.network import NeuralNetwork

'''Trying the model on a dataset of digits'''
# Load dataset
digits = load_digits()

X = digits.data
y = digits.target

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = NeuralNetwork(learning_rate=0.01)

# Hidden layer
model.add(Dense(64, 32))
model.add(ReLU())

# Output layer
model.add(Dense(32, 10))
model.add(Softmax())

# Multi-class loss
loss_function = CategoricalCrossEntropy()

# Train only on training data
model.train(
    X_train,
    y_train,
    loss_function,
    epochs=10000
)

# Evaluate on unseen test data
predictions = model.predict(X_test)

accuracy = np.mean(predictions == y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")