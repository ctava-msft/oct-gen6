import unittest
import os
import cv2
import numpy as np
from segment2 import process_and_draw_layers

class TestProcessAndDrawLayers(unittest.TestCase):

    def setUp(self):
        # Create a synthetic image for testing
        self.test_image_path = './tests/test_oct_image.jpg'
        self.invalid_image_path = './tests/invalid_image_path.jpg'
        self.non_image_file_path = './tests/test_non_image.txt'
        
        # Create a synthetic OCT image
        synthetic_image = np.random.randint(0, 256, (200, 512), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, synthetic_image)
                
        # Create a non-image file
        with open(self.non_image_file_path, 'w') as f:
            f.write("This is a test file and not an image.")

    def tearDown(self):
        # Remove the test files
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        if os.path.exists(self.non_image_file_path):
            os.remove(self.non_image_file_path)

    def test_process_and_draw_layers_with_valid_image(self):
        try:
            process_and_draw_layers(self.test_image_path)
        except Exception as e:
            self.fail(f"process_and_draw_layers raised an exception with a valid image: {e}")

    def test_process_and_draw_layers_with_invalid_image_path(self):
        with self.assertRaises(FileNotFoundError):
            process_and_draw_layers(self.invalid_image_path)

    def test_process_and_draw_layers_with_non_image_file(self):
        with self.assertRaises(ValueError):
            self.assertIn(" The provided file is not a valid image.", str(ValueError))

    def test_process_and_draw_layers_with_empty_image(self):
        empty_image_path = './tests/empty_image.jpg'
        cv2.imwrite(empty_image_path, np.zeros((200, 512), dtype=np.uint8))
        try:
            process_and_draw_layers(empty_image_path)
        except Exception as e:
            self.fail(f"process_and_draw_layers raised an exception with an empty image: {e}")
        finally:
            if os.path.exists(empty_image_path):
                os.remove(empty_image_path)

    def test_process_and_draw_layers_with_small_image(self):
        small_image_path = './tests/small_image.jpg'
        cv2.imwrite(small_image_path, np.random.randint(0, 256, (10, 10), dtype=np.uint8))
        try:
            process_and_draw_layers(small_image_path)
        except ValueError as e:
            self.assertIn("axes exceeds dimensionality of input", str(e))
        finally:
            if os.path.exists(small_image_path):
                os.remove(small_image_path)

    def test_process_and_draw_layers_with_large_image(self):
        large_image_path = './tests/large_image.jpg'
        cv2.imwrite(large_image_path, np.random.randint(0, 256, (1000, 1000), dtype=np.uint8))
        try:
            process_and_draw_layers(large_image_path)
        except Exception as e:
            self.fail(f"process_and_draw_layers raised an exception with a large image: {e}")
        finally:
            if os.path.exists(large_image_path):
                os.remove(large_image_path)

    def test_process_and_draw_layers_with_noisy_image(self):
        noisy_image_path = './tests/noisy_image.jpg'
        noisy_image = np.random.randint(0, 256, (200, 512), dtype=np.uint8)
        cv2.imwrite(noisy_image_path, noisy_image)
        try:
            process_and_draw_layers(noisy_image_path)
        except Exception as e:
            self.fail(f"process_and_draw_layers raised an exception with a noisy image: {e}")
        finally:
            if os.path.exists(noisy_image_path):
                os.remove(noisy_image_path)

if __name__ == '__main__':
    unittest.main()