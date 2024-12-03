import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def process_and_draw_layers(image_path):
    # Read the OCT image in grayscale
    oct_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute the 2D Fourier Transform of the image
    oct_fft = np.fft.fft2(oct_image)
    oct_fft_shifted = np.fft.fftshift(oct_fft)
    
    # Create a mask to keep only horizontal frequencies
    rows, cols = oct_image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a horizontal band mask
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - 5:crow + 5, :] = 1  # Adjust the band width as needed
    
    # Apply the mask to the shifted FFT
    fshift = oct_fft_shifted * mask
    
    # Inverse FFT to get the filtered image
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize the image
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = img_back.astype(np.uint8)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(img_back, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
                cv2.line(output_image, tuple(approx[i][0]), tuple(approx[i + 1][0]), colors[idx % len(colors)], 2)
        return output_image
    
    min_contour_length = 50  # Minimum length of contours to process
    filtered_image_with_lines = draw_layer_lines(oct_image, contours, min_contour_length, layer_colors)
    
    # Plot the results
    plt.figure(figsize=(15, 8))
    plt.title('OCT Image with Color-Coded Layers')
    # Convert image from BGR to RGB
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