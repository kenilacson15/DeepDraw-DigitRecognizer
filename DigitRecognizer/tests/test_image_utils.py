import unittest
import numpy as np
import sys
import os
from PIL import Image, ImageDraw

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils.image_utils import preprocess_image

class TestImageUtils(unittest.TestCase):
    def test_preprocess_image(self):
        # Create a test image
        img = Image.new('L', (280, 280), 0)
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 180, 180], fill=255)
        
        # Preprocess the image
        processed = preprocess_image(img)
        
        # Check the shape
        self.assertEqual(processed.shape, (1, 28, 28, 1))
        
        # Check that the image contains non-zero values (the shape we drew)
        self.assertTrue(np.max(processed) > 0)
        
        # Check that values are normalized between 0 and 1
        self.assertTrue(np.max(processed) <= 1.0)
        self.assertTrue(np.min(processed) >= 0.0)

if __name__ == '__main__':
    unittest.main()