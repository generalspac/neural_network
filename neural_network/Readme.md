# Neural Network From Scratch

A simple and flexible neural network implementation built completely from scratch using only Python and NumPy.

This project was created to better understand:
- Forward propagation
- Backward propagation
- Gradient descent (SGD)
- Activation functions
- Loss functions
- Multi-class and binary classification

The code is modular and easy to adapt to different datasets.

-The model is flexible, you only need to change
(depending the problem you're working on):

    Input size
    Output size
    Activation function
    Loss function

---

# Project Structure

```bash
neural-network-from-scratch/
│
├── models/
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   ├── network.py
│   └── optimizers.py
│
├── main.py
├── requirements.txt
└── README.md ```
    
# this is how it works (reminder)
Input Data
    ↓
Forward Propagation
    ↓
Prediction
    ↓
Loss Calculation
    ↓
Backward Propagation
    ↓
Weights Update (SGD)
    ↓
Repeat

