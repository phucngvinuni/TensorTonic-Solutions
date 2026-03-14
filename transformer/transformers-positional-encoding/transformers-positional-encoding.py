import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    
    Args:
        seq_length (int): The length of the sequence.
        d_model (int): The dimensionality of the embeddings.
        
    Returns:
        np.ndarray: A positional encoding matrix of shape (seq_length, d_model).
    """
    # Initialize the positional encoding matrix with zeros
    pe = np.zeros((seq_length, d_model))
    
    # Create a column vector for positions: shape (seq_length, 1)
    position = np.arange(seq_length).reshape(-1, 1)
    
    # Compute the division term using the exponential of the log trick for stability
    # This computes 10000^(2i / d_model) efficiently.
    # shape: (d_model // 2,)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Calculate the angles by multiplying position and div_term (broadcasting handles the shapes)
    angles = position * div_term
    
    # Apply sine to even indices (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(angles)
    
    # Apply cosine to odd indices (1, 3, 5, ...)
    pe[:, 1::2] = np.cos(angles)
    
    return pe
