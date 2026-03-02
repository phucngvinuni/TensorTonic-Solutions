def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    
    Y = XW + b
    
    X : n x d_in
    W : d_in x d_out
    b : length d_out
    return : n x d_out
    """

    n = len(X)         
    d_in = len(X[0])    
    d_out = len(W[0])   


    Y = [[0.0 for _ in range(d_out)] for _ in range(n)]

   
    for i in range(n):           
        for j in range(d_out):    
            s = 0.0
            for k in range(d_in):
                s += X[i][k] * W[k][j]
            Y[i][j] = s + b[j]   

    return Y