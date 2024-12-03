import cv2
import json
import numpy as np
import random
from scipy import fftpack
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def process_and_draw_layers(image_path):
    # Load the OCT image in grayscale
    oct_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if oct_image is None:
        raise ValueError("The provided file is not a valid image.")
    # Apply Fourier Transform to the image
    oct_fft = fftpack.fft2(oct_image)
    oct_fft_shifted = fftpack.fftshift(oct_fft)  # Shift zero frequency to the center

    # Create a Gaussian filter in the frequency domain to enhance high frequencies (edges)
    rows, cols = oct_image.shape
    crow, ccol = rows // 2, cols // 2  # Center
    sigma = 50  # Gaussian standard deviation
    x, y = np.ogrid[:rows, :cols]
    mask = np.exp(-((x - crow)**2 + (y - ccol)**2) / (2.0 * sigma**2))

    # Apply the filter to the FFT of the image
    filtered_fft = oct_fft_shifted * mask

    # Transform back to the spatial domain
    filtered_oct_image = np.abs(fftpack.ifft2(fftpack.ifftshift(filtered_fft)))

    # Apply Gaussian smoothing
    smoothed_image = gaussian_filter(filtered_oct_image, sigma=1).astype(np.uint8)

    # Use Canny edge detection to identify layers
    edges = cv2.Canny(smoothed_image, threshold1=30, threshold2=100)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    areas = []
    scaled_contours = []
    scale_factor = 10.2  # Adjust scale factor as needed
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)

        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        contour_shifted = contour - [cx, cy]
        contour_scaled = contour_shifted * scale_factor + [cx, cy]
        contour_scaled = contour_scaled.astype(np.int32)
        scaled_contours.append(contour_scaled)

    # Define layer colors based on the legend (approximation)
    layer_colors = [
        (255, 0, 0),    # Blue for RNFL
        (255, 165, 0),  # Orange for GCL
        (0, 255, 0),    # Green for IPL
        (255, 0, 255),  # Purple for INL
        (255, 255, 0),  # Yellow for OPL
        (255, 69, 0),   # Red-Orange for ONL/ELM
        (255, 20, 147), # Pink for EZ
        (255, 0, 0),    # Red for POS
        (0, 255, 255),  # Cyan for RPE/BM
    ]

    # Draw detected layers with color-coded lines
    def draw_layer_lines(image, contours, min_contour_length, colors):
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for idx, contour in enumerate(contours):
            if len(contour) > min_contour_length:
                # Fit a smooth curve to the contour using polynomial approximation
                approx = cv2.approxPolyDP(contour, epsilon=2, closed=False)
                line_coordinates = []
                for i in range(len(approx) - 1):
                    start_point = tuple(int(coord) for coord in approx[i][0])
                    end_point = tuple(int(coord) for coord in approx[i + 1][0])
                    cv2.line(output_image, start_point, end_point, colors[idx % len(colors)], 2)
                    line_coordinates.append({'start': start_point, 'end': end_point})
                filename = f'./output_contours/lines_contour_{idx}.json'
                print(filename)
                with open(filename, 'w') as f:
                    json.dump(line_coordinates, f)
        return output_image

    min_contour_length = 250  # Minimum length of contours to process
    filtered_image_with_lines = draw_layer_lines(oct_image, scaled_contours, min_contour_length, layer_colors)

    # Plot the results
    #plt.figure(figsize=(15, 8))
    #plt.title('OCT Image with Color-Coded Layers')
    #img = cv2.cvtColor(filtered_image_with_lines, cv2.COLOR_GRAY2RGB)
    #plt.imshow(img)
    random_suffix = random.randint(1000, 9999)
    image_path = image_path.replace(".bmp", f"_{random_suffix}_{exp_number}.bmp")
    image_path = image_path.replace(".png", f"_{random_suffix}_{exp_number}.png")
    image_path = image_path.replace(".jpg", f"_{random_suffix}_{exp_number}.jpg")
    image_path = image_path.replace("images", "output_images")
    print(image_path)
    cv2.imwrite(f"./{image_path}", filtered_image_with_lines)
    #plt.axis('off')
    #plt.show()

# experiement number
exp_number = 5
# Path to the uploaded OCT image
image_path = './images/oct-id-105.jpg'
#image_path = './images/oct-500-3-10301-1.bmp'
process_and_draw_layers(image_path)
