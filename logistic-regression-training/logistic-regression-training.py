import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # 1. Get the number of samples (N) and features (D)
    N, D = X.shape
    
    # 2. Initialize parameters: weights as zeros (D,) and bias as 0.0
    w = np.zeros(D)
    b = 0.0
    
    # 3. Gradient Descent Loop
    for _ in range(steps):
        # Forward pass: compute predictions
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # Backward pass: compute gradients
        dz = p - y
        dw = np.dot(X.T, dz) / N
        db = np.mean(dz)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
        
    return w, float(b)