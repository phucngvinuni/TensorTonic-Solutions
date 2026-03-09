import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    
    Returns a list of [x1, y1, x2, y2] bounding boxes in row-major order.
    """
    anchors = []
    stride = image_size / feature_size
    
    # Iterate over the grid in row-major order
    for i in range(feature_size):      # Rows
        for j in range(feature_size):  # Columns
            # Calculate the center of the current grid cell in image coordinates
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            # Generate an anchor for each scale and aspect ratio combination
            for s in scales:
                for r in aspect_ratios:
                    # Calculate width and height based on scale and aspect ratio
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    
                    # Calculate corner coordinates [x1, y1, x2, y2]
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0
                    
                    anchors.append([x1, y1, x2, y2])
                    
    return anchors