import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    Expects Q, K, V to have shapes like (..., seq_len, d_k)
    """
    # 1. Get the depth of the keys (d_k) for scaling
    d_k = Q.size(-1)
    
    # 2. Compute the dot product between Q and transposed K
    # We transpose the last two dimensions (-2 and -1) to align seq_len and d_k
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 3. Scale the scores
    scaled_scores = scores / math.sqrt(d_k)
    
    # 4. Apply softmax along the last dimension (key sequence length) to get probabilities
    attn_weights = F.softmax(scaled_scores, dim=-1)
    
    # 5. Multiply the attention weights by the Values (V)
    output = torch.matmul(attn_weights, V)
    
    return output