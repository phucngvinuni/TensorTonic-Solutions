def conv2d(image: list[list[float]], kernel: list[list[float]], stride: int, padding: int) -> list[list[float]]:
    """
    Computes the 2D convolution of an image with a given kernel, stride, and padding.
    """
    H = len(image)
    W = len(image[0])
    k_h = len(kernel)
    k_w = len(kernel[0])
    
    # 1. Calculate output dimensions
    H_out = (H + 2 * padding - k_h) // stride + 1
    W_out = (W + 2 * padding - k_w) // stride + 1
    
    # 2. Create the padded image
    padded_H = H + 2 * padding
    padded_W = W + 2 * padding
    padded_image = [[0.0] * padded_W for _ in range(padded_H)]
    
    for i in range(H):
        for j in range(W):
            padded_image[i + padding][j + padding] = float(image[i][j])
            
    # 3. Perform the convolution
    output = [[0.0] * W_out for _ in range(H_out)]
    
    for i in range(H_out):
        for j in range(W_out):
            # Compute the dot product for the current patch
            patch_sum = 0.0
            for m in range(k_h):
                for n in range(k_w):
                    # Find the corresponding pixel in the padded image
                    img_val = padded_image[i * stride + m][j * stride + n]
                    # Multiply by the kernel weight and accumulate
                    patch_sum += img_val * kernel[m][n]
            
            output[i][j] = patch_sum
            
    return output

# --- Example Usage ---
if __name__ == "__main__":
    image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    kernel = [[1, 0], [0, 1]]
    stride = 1
    padding = 0
    
    result = conv2d(image, kernel, stride, padding)
    print("Output:")
    for row in result:
        print(row)