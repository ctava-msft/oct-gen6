import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def process_and_draw_rnfl(image_path):
    # Load the OCT image in grayscale
    oct_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if oct_image is None:
        raise ValueError("The provided file is not a valid image.")

    # Apply Gaussian smoothing
    smoothed_image = gaussian_filter(oct_image, sigma=2)

    # Use Canny edge detection to identify layers
    edges = cv2.Canny(smoothed_image, threshold1=30, threshold2=100)

    # Focus on the top part of the image (region of interest)
    height, width = edges.shape
    roi_top = edges[: height // 3, :]  # Top third of the image

    # Find contours from the ROI
    contours, _ = cv2.findContours(roi_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an output image for drawing
    output_image = cv2.cvtColor(oct_image, cv2.COLOR_GRAY2BGR)

    # Define the purple color for the RNFL
    purple = (255, 0, 255)

    # Filter and identify the RNFL contour
    rnfl_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > width * 0.8 and h < height * 0.1:  # Wide and flat contour
            rnfl_contour = contour
            break

    # Draw the RNFL contour if found
    if rnfl_contour is not None:
        cv2.drawContours(output_image, [rnfl_contour], -1, purple, thickness=2)
    else:
        print("RNFL contour not found!")

    # Save and return the processed image
    output_path = image_path.replace(".png", "_processed.png")
    cv2.imwrite(output_path, output_image)
    return output_path

# Path to the uploaded OCT image
image_path = './images/oct-id-105.jpg'
processed_image_path = process_and_draw_rnfl(image_path)
print(f"Processed image saved at: {processed_image_path}")
