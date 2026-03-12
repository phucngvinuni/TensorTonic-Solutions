import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    # 1. Linear projections for Q, K, V
    # Shape transitions from (batch, seq_len, d_model) -> (batch, seq_len, d_model)
    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)
    
    # 2. Reshape to separate the heads: (batch, seq_len, num_heads, d_k)
    # Then transpose to: (batch, num_heads, seq_len, d_k) for parallel attention
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # 3. Scaled Dot-Product Attention
    # Matmul Q_heads and K_heads^T: (batch, num_heads, seq_len, d_k) @ (batch, num_heads, d_k, seq_len)
    # Results in scores of shape (batch, num_heads, seq_len, seq_len)
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)
    
    # Multiply by V_heads: (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, d_k)
    # Results in output of shape (batch, num_heads, seq_len, d_k)
    attention_output = np.matmul(attention_weights, V_heads)
    
    # 4. Concatenate the heads
    # Transpose back to (batch, seq_len, num_heads, d_k)
    # Reshape back to (batch, seq_len, d_model)
    concat_attention = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # 5. Final Output Projection
    # Multiply by W_o: (batch, seq_len, d_model) @ (d_model, d_model)
    output = np.dot(concat_attention, W_o)
    
    return output