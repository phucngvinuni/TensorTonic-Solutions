import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # 1. First linear transformation: Project from d_model to d_ff
    hidden = np.dot(x, W1) + b1
    
    # 2. ReLU activation: max(0, x)
    relu_out = np.maximum(0, hidden)
    
    # 3. Second linear transformation: Project back from d_ff to d_model
    output = np.dot(relu_out, W2) + b2
    
    return output