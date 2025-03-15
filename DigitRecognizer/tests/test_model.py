import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.model.build_model import build_model

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def test_model_structure(self):
        self.assertEqual(self.model.input_shape, (None, 28, 28, 1))
        self.assertEqual(self.model.output_shape, (None, 10))
        self.assertIsNotNone(self.model.optimizer)
        self.assertIsNotNone(self.model.loss)

    def test_model_prediction(self):
        test_image = np.random.random((1, 28, 28, 1))
        prediction = self.model.predict(test_image, verbose=0)
        self.assertEqual(prediction.shape, (1, 10))
        self.assertAlmostEqual(np.sum(prediction), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
