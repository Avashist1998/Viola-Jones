import unittest
import numpy as np
from utils.ada_boost import AdaBoost


class TestDecision(unittest.TestCase):
    """Test detect AdaBoost Classifier."""
    
    def setUp(self):
        """Setup the dataset that will used by the test cases."""
        test_dataset_path = "tests/data/test_dataset.csv"
        tmp_matrix = np.genfromtxt(test_dataset_path, delimiter=",")[1:, :]
        self.X, self.y, self.dist = tmp_matrix[:, :-2], tmp_matrix[:, -2], tmp_matrix[:, -1]

    def test_model_normal_data_fit(self):
        """Test the model training on simple dataset."""

        test_est = AdaBoost(2)
        test_est.fit(self.X, self.y, self.dist)

        self.assertEqual(test_est.num_of_est, 2)
        self.assertEqual(test_est.weak_est[0].feature_index, 2)
        self.assertAlmostEqual(test_est.weak_est[0].theta, -3.226097158)
        self.assertAlmostEqual(test_est.weak_est[0].weighted_error,  0.0054768901)
    
        self.assertEqual(test_est.weak_est[1].feature_index, 8)
        self.assertAlmostEqual(test_est.weak_est[1].theta, 3.257529189)
        self.assertAlmostEqual(test_est.weak_est[1].weighted_error,  0.0029391633)

    def test_model_flipped_data_fit(self):
        """Test the model training on flipped polarity dataset."""

        test_est = AdaBoost(2)
        test_est.fit(self.X, self.y*-1, self.dist)

        self.assertEqual(test_est.num_of_est, 2)
        self.assertEqual(test_est.weak_est[0].feature_index, 2)
        self.assertAlmostEqual(test_est.weak_est[0].theta, -3.226097158243836)
        self.assertAlmostEqual(test_est.weak_est[0].weighted_error,  0.0054768901)
    
        self.assertEqual(test_est.weak_est[1].feature_index, 8)
        self.assertAlmostEqual(test_est.weak_est[1].theta, 3.257529189)
        self.assertAlmostEqual(test_est.weak_est[1].weighted_error,  0.00293916338)

    def test_model_no_dist_data_fit(self):
        """Test the model training on no distribution dataset."""

        test_est = AdaBoost()
        test_est.fit(self.X, self.y)

        self.assertEqual(test_est.num_of_est, 5)
        self.assertEqual(test_est.weak_est[0].feature_index, 9)
        self.assertAlmostEqual(test_est.weak_est[0].theta, -0.7451168348)
        self.assertAlmostEqual(test_est.weak_est[0].weighted_error,  0.075757575)
    
        self.assertEqual(test_est.weak_est[1].feature_index, 6)
        self.assertAlmostEqual(test_est.weak_est[1].theta, -0.2656794836)
        self.assertAlmostEqual(test_est.weak_est[1].weighted_error,  0.0464480874316)
