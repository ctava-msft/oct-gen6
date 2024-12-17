import unittest
import numpy as np
import cv2
import os
from segment5 import segment_rpe_layer

class TestSegmentRPE(unittest.TestCase):

    # def test_segment_rpe_layer_with_synthetic_data(self):
    #     # Create a synthetic B-scan image
    #     bscan = np.random.rand(200, 512)
    #     params = {
    #         'MEDLINE_SIGMA1': 1.0,
    #         'MEDLINE_SIGMA2': 2.0,
    #         'MEDLINE_LINESWEETER': 5,
    #         'MEDLINE_MINDIST': 10,
    #         'RPE_SEGMENT_MEDFILT1': (5, 7),
    #         'RPE_SEGMENT_MEDFILT2': (5, 7),
    #         'RPE_SEGMENT_LINESWEETER1': 5,
    #         'RPE_SEGMENT_LINESWEETER2': 5,
    #         'RPE_SEGMENT_POLYDIST': 10,
    #         'RPE_SEGMENT_POLYNUMBER': 5,
    #         'REMOVEBIAS_FRACTION': 0.75,
    #         'REMOVEBIAS_REGIONWIDTH': 10,
    #         'SPLITNORMALIZE_CUTOFF': 2.0
    #     }
    #     medline = np.ones(512) * 100

    #     # Call the function to be tested
    #     rpe_auto = segment_rpe_layer(bscan, params, medline)

    #     # Print output length and expected length
    #     print(f"{len(rpe_auto)} {bscan.shape[1]}")

    #     # Assert the output shape matches input width
    #     self.assertEqual(len(rpe_auto), bscan.shape[1])

    def test_segment_rpe_layer_with_image(self):
        # Define the path to the image
        image_path = os.path.join(os.path.dirname(__file__), 'images', 'oct-id-105.jpg')

        # Check if the image file exists
        self.assertTrue(os.path.exists(image_path), f"Image file not found at {image_path}")

        # Load the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Ensure the image was loaded correctly
        self.assertIsNotNone(image, "Failed to load the image.")

        # Normalize the image to [0, 1] if it's not already
        # if image.dtype == np.uint8:
        #     image = image.astype(np.float32) / 255.0
        # elif image.dtype == np.uint16:
        #     image = image.astype(np.float32) / 65535.0
        # else:
        #     image = image.astype(np.float32)

        # Define parameters for segmentation
        params = {
            'MEDLINE_SIGMA1': 1.0,
            'MEDLINE_SIGMA2': 2.0,
            'MEDLINE_LINESWEETER': 5,
            'MEDLINE_MINDIST': 10,
            'RPE_SEGMENT_MEDFILT1': (5, 7),
            'RPE_SEGMENT_MEDFILT2': (5, 7),
            'RPE_SEGMENT_LINESWEETER1': 5,
            'RPE_SEGMENT_LINESWEETER2': 5,
            'RPE_SEGMENT_POLYDIST': 10,
            'RPE_SEGMENT_POLYNUMBER': 5,
            'REMOVEBIAS_FRACTION': 0.75,
            'REMOVEBIAS_REGIONWIDTH': 10,
            'SPLITNORMALIZE_CUTOFF': 2.0
        }

        # Create a medline array based on image dimensions
        #medline = np.full(image.shape[1], image.shape[0] // 2)

        # Call the segmentation function
        rpe_auto = segment_rpe_layer(image, params)
        print("./tests/rpe_auto.png")
        cv2.imwrite(f"./tests/rpe_auto.png", rpe_auto)
        #print(rpe_auto)
"""
        if not isinstance(rpe_auto, np.ndarray):
            rpe_auto = np.array(rpe_auto)

        print(f"Type of rpe_auto: {type(rpe_auto["min"])}")
        print(f"Contents of rpe_auto: {rpe_auto["min"]}")

        # Convert to uint8 if necessary
        if rpe_auto.dtype != np.uint8:
            rpe_auto = rpe_auto.astype(np.uint8)
        cv2.imwrite(f"./rpe_auto.png", rpe_auto)
        # Assertions to verify the output
        # Check that the output array has the correct length
        self.assertEqual(len(rpe_auto), image.shape[1], f"Output length {len(rpe_auto)}does not match image width.{image.shape[1]}")
"""

if __name__ == '__main__':
    unittest.main()