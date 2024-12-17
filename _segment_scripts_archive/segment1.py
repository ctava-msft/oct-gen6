import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    """
    Preprocesses the OCT image:
    - Normalizes intensity
    - Applies Gaussian blur for noise reduction
    """
    # Normalize the image to range [0, 255]
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = np.uint8(image)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

def edge_detection(image):
    """
    Detects edges in the OCT image using the Sobel operator.
    """
    # Sobel edge detection (horizontal and vertical gradients)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    # Combine gradients
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(gradient_magnitude)

def segment_layers(image, edge_image):
    """
    Segments layers based on edge detection and intensity thresholds.
    """
    # Apply binary thresholding on the edge image
    _, binary_edges = cv2.threshold(edge_image, 50, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    cleaned_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel)

    # Use the cleaned edge image to segment layers
    segmented_layers = cv2.bitwise_and(image, image, mask=cleaned_edges)

    return segmented_layers

def visualize_results(original, edge_image, segmented_layers):
    """
    Visualizes the original image, edge detection result, and segmented layers.
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Edge Detection")
    plt.imshow(edge_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Segmented Layers")
    plt.imshow(segmented_layers, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    # Load the OCT image
    image_path = "./images/oct-id-105.jpg"
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(original_image)

    # Step 2: Detect edges
    edge_image = edge_detection(preprocessed_image)

    # Step 3: Segment layers
    segmented_layers = segment_layers(preprocessed_image, edge_image)

    # Step 4: Visualize results
    visualize_results(original_image, edge_image, segmented_layers)

if __name__ == "__main__":
    main()
