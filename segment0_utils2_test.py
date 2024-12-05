import cv2
import unittest
import numpy as np
import os
from segment0 import extrema_finder
from segment0 import extend_blood_vessels
from segment0 import find_blood_vessels
from segment0 import find_low_pos_near_lmax
from segment0 import find_medline
from segment0 import line_sweeter
from segment0 import remove_bias
from segment0 import split_normalize

def convert_int64(d):
    return {
        'max': [[int(x) for x in sublist] for sublist in d['max']],
        'min': [[int(x) for x in sublist] for sublist in d['min']]
    }

class UtilsTest(unittest.TestCase):

    def test_split_normalize_with_real_image(self):

        # Define the path to the image
        image_path = os.path.join(os.path.dirname(__file__), 'images', 'oct-id-105.jpg')

        # Check if the image file exists
        self.assertTrue(os.path.exists(image_path), f"Image file not found at {image_path}")

        # Load the image using OpenCV
        octimg = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Normalize the image to [0, 1]
        octimg = octimg.astype(np.float64) / 255.0

        params = {
            'SPLITNORMALIZE_CUTOFF': 2.0
        }
        params = {
            'MEDLINE_SIGMA1': 1.0,
            'MEDLINE_SIGMA2': 2.0,
            'MEDLINE_LINESWEETER': 5,
            'MEDLINE_MINDIST': 10
        }
        #medline = np.full(512, 100)

        # Call the function to be tested
        modes = 'ipsimple opsimple soft'
        noctimg, medline_output = split_normalize(octimg, params, mode=modes, medline=None)

        print("medline")
        print(medline_output)
        # Check that the output image has the same shape as input
        self.assertEqual(noctimg.shape, octimg.shape)

        # Check that medline output matches the input medline
        #np.testing.assert_array_equal(medline_output, medline)

        # Check that the output image values are within [0, 1]
        # self.assertTrue(np.all(noctimg >= 0))
        # self.assertTrue(np.all(noctimg <= 1))
        cv2.imwrite(f"./tests/test_split_normalize.png", noctimg)
        
if __name__ == '__main__':
    unittest.main()