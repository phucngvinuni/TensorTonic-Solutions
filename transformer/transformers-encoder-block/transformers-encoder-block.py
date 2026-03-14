import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    
    x_hat = (x - mean) / np.sqrt(var + eps)
    
    return gamma * x_hat + beta


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """

    B, N, d_model = Q.shape
    d_k = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Split into heads
    Q_heads = Q_proj.reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scores = Q_heads @ K_heads.transpose(0,1,3,2) / np.sqrt(d_k)
    attn = softmax(scores, axis=-1)

    context = attn @ V_heads

    # Concatenate heads
    context = context.transpose(0,2,1,3).reshape(B, N, d_model)

    # Final projection
    output = context @ W_o

    return output


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """

    hidden = x @ W1 + b1
    hidden = np.maximum(0, hidden)   # ReLU
    output = hidden @ W2 + b2

    return output


def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """

    # Multi-head attention
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)

    # Residual + LayerNorm
    x1 = layer_norm(x + attn_out, gamma1, beta1)

    # Feed-forward
    ff_out = feed_forward(x1, W1, b1, W2, b2)

    # Residual + LayerNorm
    out = layer_norm(x1 + ff_out, gamma2, beta2)

    return out