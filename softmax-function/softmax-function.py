import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # 1. Find the max value along the last axis for numerical stability.
    # keepdims=True ensures the shape allows for proper broadcasting (e.g., (N, 1) for 2D arrays).
    x_max = np.max(x, axis=-1, keepdims=True)
    
    # 2. Subtract the max from x and exponentiate.
    e_x = np.exp(x - x_max)
    
    # 3. Sum the exponentials along the last axis.
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
    
    # 4. Divide to get the probability distribution.
    return e_x / sum_e_x