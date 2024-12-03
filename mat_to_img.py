import sys
import scipy.io
from PIL import Image
import numpy as np

def inspect_image_layer(mat_path):
    # Load the .mat file
    mat = scipy.io.loadmat(mat_path)
    
    # Extract the imageLayer data
    image_layer = mat['imageLayer']['retinalLayers'][0, 0]
    
    # Print the structure of image_layer
    print("Structure of image_layer:", image_layer.dtype)

def mat_to_img(mat_path, save_path):
    # Load the .mat file
    mat = scipy.io.loadmat(mat_path)

    # Print the keys of the .mat file
    print("Keys in the .mat file:", mat.keys())
    
    # Extract the image data (assuming the image data is stored in a variable named 'image')
    #image_layer = mat['imageLayer']
    image_layer = mat['imageLayer']['retinalLayers'][0, 0]

    # Print the structure of imageLayer
    print("Structure of imageLayer:", image_layer.dtype)
    
    # Convert the image data to uint8 format
    image_layer = (image_layer * 255).astype(np.uint8)
    
    # Create an image from the data
    image = Image.fromarray(image_layer)
    
    # Save the image as a .jpg file
    image.save(save_path)

def main(mat_path, save_path):
    #mat_to_img(mat_path, save_path)
    inspect_image_layer(mat_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mat_to_img.py <mat_path> <save_path>")
        sys.exit(1)
    
    mat_path = sys.argv[1]
    save_path = sys.argv[2]
    main(mat_path, save_path)