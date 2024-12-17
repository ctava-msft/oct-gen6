import cv2
import json
import numpy as np
from scipy.ndimage import gaussian_filter

def process_and_draw_layers(image_path):
    # Load the OCT image in grayscale
    oct_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if oct_image is None:
        raise ValueError("The provided file is not a valid image.")
    
    # Apply Gaussian smoothing
    smoothed_image = gaussian_filter(oct_image, sigma=2)

    # Use Canny edge detection to identify layers
    edges = cv2.Canny(smoothed_image, threshold1=30, threshold2=100)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define the purple color for the retinal nerve fiber layer
    purple = (255, 0, 255)

    # Function to draw a single contour with smoothing
    def draw_layer(output_image, contour, color, thickness=2):
        # Fit a smooth curve using polynomial approximation
        approx = cv2.approxPolyDP(contour, epsilon=5, closed=False)
        for i in range(len(approx) - 1):
            start_point = tuple(approx[i][0])
            end_point = tuple(approx[i + 1][0])
            cv2.line(output_image, start_point, end_point, color, thickness)

    # Create an output image for drawing
    output_image = cv2.cvtColor(oct_image, cv2.COLOR_GRAY2BGR)

    # Filter and draw the retinal nerve fiber layer (assumed to be the largest contour near the top)
    largest_contour = max(contours, key=cv2.contourArea)
    draw_layer(output_image, largest_contour, purple, thickness=3)

    # Save and return the processed image
    output_path = image_path.replace("images", "output_images").replace(".jpg", "_processed.jpg")
    cv2.imwrite(output_path, output_image)
    return output_path

# Path to the uploaded OCT image
image_path = './images/oct-id-105.jpg'
processed_image_path = process_and_draw_layers(image_path)
print(f"Processed image saved at: {processed_image_path}")
