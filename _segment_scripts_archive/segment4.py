import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import fftpack
from scipy.ndimage import gaussian_filter

def process_and_draw_layers(image_path):
    # Load the OCT image in grayscale
    oct_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if oct_image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Apply Fourier Transform to the image
    oct_fft = fftpack.fft2(oct_image)
    oct_fft_shifted = fftpack.fftshift(oct_fft)

    # Get image dimensions and center coordinates
    rows, cols = oct_image.shape
    crow, ccol = rows // 2, cols // 2

    # Create a wider horizontal band-pass filter mask
    mask = np.zeros((rows, cols), np.uint8)
    band_width = 30  # Adjust band width to include more horizontal frequencies
    mask[crow - band_width:crow + band_width, :] = 1

    # Apply the mask to the shifted FFT
    filtered_fft = oct_fft_shifted * mask

    # Inverse FFT to get the filtered image
    img_back = fftpack.ifft2(fftpack.ifftshift(filtered_fft))
    img_back = np.abs(img_back)

    # Normalize and enhance contrast of the image
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = img_back.astype(np.uint8)
    img_back = cv2.equalizeHist(img_back)

    # Apply Gaussian smoothing
    smoothed_image = gaussian_filter(img_back, sigma=1)

    # Use Canny edge detection
    edges = cv2.Canny(smoothed_image, threshold1=10, threshold2=50)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define colors for layers
    layer_colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 255, 255) # White
    ]

    # Function to draw layer lines
    def draw_layer_lines(image, contours, min_contour_length, colors):
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for idx, contour in enumerate(contours):
            if len(contour) < min_contour_length:
                continue
            approx = cv2.approxPolyDP(contour, 3, True)
            for i in range(len(approx) - 1):
                cv2.line(output_image,
                         tuple(approx[i][0]),
                         tuple(approx[i + 1][0]),
                         colors[idx % len(colors)],
                         2)
        return output_image

    min_contour_length = 100  # Increase minimum contour length if needed
    filtered_image_with_lines = draw_layer_lines(oct_image, contours,
                                                 min_contour_length, layer_colors)

    # Plot the results
    plt.figure(figsize=(15, 8))
    plt.title('OCT Image with Color-Coded Layers')
    # Convert image from BGR to RGB for displaying
    img_rgb = cv2.cvtColor(filtered_image_with_lines, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

    # Save the result
    random_suffix = random.randint(1000, 9999)
    filename = "output.png"
    filename = filename.replace(".png", f"_{random_suffix}.png")
    cv2.imwrite(f"./{filename}", filtered_image_with_lines)

if __name__ == "__main__":
    image_path = './images/oct-id-105.jpg'  # Replace with your actual image path
    process_and_draw_layers(image_path)