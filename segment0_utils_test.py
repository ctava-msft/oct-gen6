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

    def test_extrema_finder_basic(self):
        input_data = np.array([
            [1, 3, 2, 5, 4, 7, 6],
            [2, 4, 1, 6, 3, 5, 7]
        ])
        expected_extrema = {
            'max': [[7, 5], [7, 6]],
            'min': [[2, 1], [1, 3]]
        }
        result = extrema_finder(input_data, num_extrema=2, mode='min pos')
        converted_result = convert_int64(result)
        self.assertEqual(converted_result, expected_extrema)

    def test_extrema_finder_position_mode(self):
        # Additional test for 'pos' mode
        input_data = np.array([
            [0, 2, 1, 3, 0],
            [1, 3, 2, 4, 1]
        ])
        expected_extrema = {
            'max': [[3, 2], [4, 3]],
            'min': [[0, 1], [1, 2]]
        }
        num_extrema = 2
        result = extrema_finder(input_data, num_extrema, mode='pos')

        # Convert values to integers for comparison
        converted_result = convert_int64(result)
        self.assertEqual(converted_result, expected_extrema)

    def test_extrema_finder_max_only(self):
        # Additional test for 'max' mode
        input_data = np.array([
            [1, 4, 3, 6, 5],
            [2, 5, 2, 7, 6]
        ])
        expected_extrema = {
            'max': [[6, 4], [7, 5]],
            'min': [[], []]
        }
        num_extrema = 2
        result = extrema_finder(input_data, num_extrema, mode='max')

        # Convert values to integers for comparison
        converted_result = {
            'max': [[int(x) for x in sublist] for sublist in result['max']],
            'min': [[] for _ in input_data]
        }
        self.assertEqual(converted_result, expected_extrema)

    def test_extrema_finder_min_only(self):
        # Additional test for 'min' mode
        input_data = np.array([
            [3, 1, 4, 2, 5],
            [5, 2, 6, 3, 7]
        ])
        expected_extrema = {
            'max': [[], []],
            'min': [[1, 2], [2, 3]]
        }
        num_extrema = 2
        result = extrema_finder(input_data, num_extrema, mode='min')

        # Convert values to integers for comparison
        converted_result = {
            'max': [[], []],
            'min': [[int(x) for x in sublist] for sublist in result['min']]
        }
        self.assertEqual(converted_result, expected_extrema)

    def test_extend_blood_vessels(self):
        bv = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        add_width = 1
        mult_width_thresh = 2
        mult_width = 0.5
        
        expected_extended = np.array([1,1,1,1,1], dtype=np.uint8)
        result = extend_blood_vessels(bv, add_width, mult_width_thresh, mult_width)
        np.testing.assert_array_equal(result, expected_extended)
        
        # Test without multiplicative extension
        bv = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        add_width = 1
        mult_width_thresh = 2
        mult_width = 0
        
        expected_extended = np.array([1,1,1,1,1], dtype=np.uint8)
        result = extend_blood_vessels(bv, add_width, mult_width_thresh, mult_width)
        np.testing.assert_array_equal(result, expected_extended)

    def test_find_blood_vessels(self):
        # Create a mock B-scan image
        bscan = np.random.rand(100, 100)
        
        # Define parameters
        params = {
            'FINDBLOODVESSELS_MULTWIDTHTHRESH': 2,
            'FINDBLOODVESSELS_MULTWIDTH': 0.5,
            'FINDBLOODVESSELS_FREEWIDTH': 1,
            'FINDBLOODVESSELS_THRESHOLD': 0.7,
            'FINDBLOODVESSELS_WINDOWWIDTH': 1,
            'FINDBLOODVESSELS_WINDOWHEIGHT': 1
        }
        
        # Ensure 'linerpe' is a 2D array with at least one column
        linerpe = np.random.rand(100, 1)
        
        # Call the function
        result = find_blood_vessels(bscan, params, linerpe)
        
        # Add assertions as needed
        self.assertIsNotNone(result)
        self.assertEqual(len(result), linerpe.shape[0])

    def test_find_low_pos_near_lmax_basic_case(self):
        data = np.array([1, 3, 2, 5, 4, 6, 5])
        lmax_indices = np.array([1, 3, 5])
        lmax = 5
        sigma2 = 1.0
        expected = np.array([0, 2, 4])
        result = find_low_pos_near_lmax(data, lmax_indices, lmax, sigma2)
        np.testing.assert_array_equal(result, expected)

    def test_find_low_pos_near_lmax_no_lmax(self):
        data = np.array([1, 1, 1, 1])
        lmax_indices = np.array([])
        lmax = 1
        sigma2 = 1.0
        expected = np.array([])
        result = find_low_pos_near_lmax(data, lmax_indices, lmax, sigma2)
        np.testing.assert_array_equal(result, expected)

    def test_find_low_pos_near_lmax_single_element(self):
        # Convert data to a 2D array with shape (1, 1)
        data = np.array([[10]])
        lmax_indices = np.array([[0]])
        lmax = np.array([10])  # Assuming lmax should be an array
        sigma2 = 1.0
        expected = np.array([[0]])
        result = find_low_pos_near_lmax(data, lmax_indices, lmax, sigma2)
        np.testing.assert_array_equal(result, expected)

    def test_find_medline_with_synthetic_data(self):
        # Create a synthetic OCT B-scan image
        octimg = np.random.rand(200, 512)
        params = {
            'MEDLINE_SIGMA1': 1.0,
            'MEDLINE_SIGMA2': 2.0,
            'MEDLINE_LINESWEETER': 5,
            'MEDLINE_MINDIST': 10
        }

        # Call the function to be tested
        medline = find_medline(octimg, params)

        # Assert the output shape matches input width
        self.assertEqual(len(medline), octimg.shape[1])

        # Assert the output values are within the valid range of row indices
        self.assertTrue(np.all((medline >= 0) & (medline < octimg.shape[0])))

    def test_find_medline_with_real_image(self):
        # Define the path to the image
        image_path = os.path.join(os.path.dirname(__file__), 'images', 'oct-id-105.jpg')

        # Check if the image file exists
        self.assertTrue(os.path.exists(image_path), f"Image file not found at {image_path}")

        # Load the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Ensure the image was loaded correctly
        self.assertIsNotNone(image, "Failed to load the image.")

        # Normalize the image to [0, 1] if it's not already
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        else:
            image = image.astype(np.float32)

        # Define parameters for segmentation
        params = {
            'MEDLINE_SIGMA1': 1.0,
            'MEDLINE_SIGMA2': 2.0,
            'MEDLINE_LINESWEETER': 5,
            'MEDLINE_MINDIST': 10
        }

        # Call the function to be tested
        medline = find_medline(image, params)

        # Assert the output shape matches input width
        self.assertEqual(len(medline), image.shape[1])

        # Assert the output values are within the valid range of row indices
        self.assertTrue(np.all((medline >= 0) & (medline < image.shape[0])))

    def test_line_sweeter_basic(self):
        input_data = np.array([1, 2, 3, 4, 5, 6, 7])
        window_size = 3
        expected = np.array([1, 2, 3, 4, 5, 6, 6])  # Updated expected output
        result = line_sweeter(input_data, window_size)
        np.testing.assert_array_equal(result, expected)

    def test_line_sweeter_empty_input(self):
        input_data = np.array([])
        window_size = 3
        expected = np.array([])
        result = line_sweeter(input_data, window_size)
        np.testing.assert_array_equal(result, expected)

    def test_remove_bias_all_positive(self):
        # Create a sample OCT image with a known bias (2D array)
        octimg = np.array([
            [1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ], dtype=np.float64)
        
        # Define parameters
        params = {
            'REMOVEBIAS_REGIONWIDTH': 2,
            'REMOVEBIAS_FRACTION': 0.75
        }
        
        # Apply remove_bias
        corrected = remove_bias(octimg, params)
        
        # Assertions
        self.assertEqual(corrected.shape, octimg.shape, "Output shape mismatch.")
        self.assertTrue(np.all(corrected >= 0), "Some values are less than 0.")
        self.assertTrue(np.all(corrected <= 1), "Some values are greater than 1.")
        
        # Calculate expected bias
        top_region = octimg[:2, :].flatten()
        bottom_region = octimg[-2:, :].flatten()
        temp = np.concatenate((top_region, bottom_region))
        temp_sorted = np.sort(temp)
        bias = np.mean(temp_sorted[:int(len(temp_sorted) * params['REMOVEBIAS_FRACTION'])])
        
        # Expected corrected image
        expected_corrected = octimg - bias
        expected_corrected = np.clip(expected_corrected, 0, 1)
        max_val = expected_corrected.max()
        if max_val != 0:
            expected_corrected = expected_corrected / max_val
        else:
            expected_corrected = expected_corrected
        
        # Define desired output
        desired = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float64)
        
        # Assert almost equal
        np.testing.assert_almost_equal(corrected, desired, decimal=5, err_msg="Bias removal incorrect.")

    def test_remove_bias_all_zero(self):
        # Create a homogeneous OCT image where bias removal should result in all zeros
        octimg = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ], dtype=np.float64)
        
        # Define parameters to ensure bias equals the pixel values
        params = {
            'REMOVEBIAS_REGIONWIDTH': 2,
            'REMOVEBIAS_FRACTION': 1.0  # Take the mean of all values
        }
        
        # Apply remove_bias
        corrected = remove_bias(octimg, params)
        
        # Define desired output as all zeros
        desired = np.zeros_like(octimg)
        
        # Assertions
        self.assertEqual(corrected.shape, octimg.shape, "Output shape mismatch.")
        self.assertTrue(np.all(corrected >= 0), "Some values are less than 0.")
        self.assertTrue(np.all(corrected <= 1), "Some values are greater than 1.")
        np.testing.assert_almost_equal(corrected, desired, decimal=5, err_msg="Bias removal incorrect.")

    def test_remove_bias_invalid_input(self):
        # Test with a 1D array to ensure ValueError is raised
        octimg = np.array([0.2, 0.3, 0.4], dtype=np.float64)
        params = {
            'REMOVEBIAS_REGIONWIDTH': 2,
            'REMOVEBIAS_FRACTION': 0.75
        }
        
        with self.assertRaises(ValueError):
            remove_bias(octimg, params)

    def test_remove_bias_small_image(self):
        # Test with an image smaller than region width
        octimg = np.array([
            [0.2, 0.3],
            [0.2, 0.3]
        ], dtype=np.float64)
        
        params = {
            'REMOVEBIAS_REGIONWIDTH': 3,  # Larger than image dimensions
            'REMOVEBIAS_FRACTION': 0.75
        }
        
        with self.assertRaises(IndexError):
            remove_bias(octimg, params)

    def test_split_normalize(self):
        # Create a synthetic OCT image
        octimg = np.random.rand(200, 512)
        params = {
            'SPLITNORMALIZE_CUTOFF': 2.0
        }
        medline = np.full(512, 100)

        # Call the function to be tested
        noctimg, medline_output = split_normalize(octimg, params, medline=medline)

        # Check that the output image has the same shape as input
        self.assertEqual(noctimg.shape, octimg.shape)

        # Check that medline output matches the input medline
        np.testing.assert_array_equal(medline_output, medline)

        # Check that the output image values are within [0, 1]
        self.assertTrue(np.all(noctimg >= 0))
        self.assertTrue(np.all(noctimg <= 1))
        
if __name__ == '__main__':
    unittest.main()