import unittest
import numpy as np
from utils.decision_stamp import DecisionStamp


class TestDecision(unittest.TestCase):
    """Test detect AdaBoost Classifier."""
    
    def setUp(self):
        """Setup the dataset that will used by the test cases."""
        test_dataset_path = "tests/data/test_dataset.csv"
        tmp_matrix = np.genfromtxt(test_dataset_path, delimiter=",")[1:, :]
        self.X, self.y, self.dist = tmp_matrix[:, :-2], tmp_matrix[:, -2], tmp_matrix[:, -1]

    def test_initial_distribution(self):
        """Test the initial distribution calculation function."""

        mock_labels_balance = np.array([1]*25 + [-1]*25)
        cal_dist = DecisionStamp.init_distribution(mock_labels_balance)
        self.assertEqual(len(cal_dist), 50)
        self.assertEqual(cal_dist[0], 1/(2*25))
        self.assertEqual(cal_dist[49], 1/(2*25))
        self.assertEqual(cal_dist[0], cal_dist[49])

        mock_labels_balance = np.array([1]*40 + [-1]*10)
        cal_dist = DecisionStamp.init_distribution(mock_labels_balance)
        self.assertEqual(len(cal_dist), 50)
        self.assertEqual(cal_dist[0], 1/(2*40))
        self.assertEqual(cal_dist[49], 1/(2*10))
        self.assertNotEqual(cal_dist[0], cal_dist[49])

        mock_labels_balance = np.array([1]*10 + [-1]*40)
        cal_dist = DecisionStamp.init_distribution(mock_labels_balance)
        self.assertEqual(len(cal_dist), 50)
        self.assertEqual(cal_dist[0], 1/(2*10))
        self.assertEqual(cal_dist[49], 1/(2*40))
        self.assertNotEqual(cal_dist[0], cal_dist[49])


    def test_model_normal_data_fit(self):
        """Test the model training on simple dataset."""

        test_est = DecisionStamp()
        test_est.fit(self.X, self.y, self.dist)

        self.assertEqual(test_est.polarity, -1)
        self.assertEqual(test_est.feature_index, 2)
        self.assertAlmostEqual(test_est.theta, -3.226097158)
        self.assertAlmostEqual(test_est.weighted_error,  0.0054768901)


    def test_model_flipped_data_fit(self):
        """Test the model training on flipped polarity dataset."""

        test_est = DecisionStamp()
        test_est.fit(self.X, self.y*-1, self.dist)

        self.assertEqual(test_est.polarity, 1)
        self.assertEqual(test_est.feature_index, 2)
        self.assertAlmostEqual(test_est.theta, -3.226097158)
        self.assertAlmostEqual(test_est.weighted_error,  0.0054768901)

    def test_model_no_dist_data_fit(self):
        """Test the model training on no distribution dataset."""

        test_est = DecisionStamp()
        test_est.fit(self.X, self.y)

        self.assertEqual(test_est.polarity, -1)
        self.assertEqual(test_est.feature_index, 9)
        self.assertAlmostEqual(test_est.theta, -0.7451168348)
        self.assertAlmostEqual(test_est.weighted_error,  0.07575757575)
        
        classes, counts = np.unique(test_est.dist, return_counts=True)
        self.assertAlmostEqual(classes[0], 1/(2*counts[0]))
        self.assertAlmostEqual(classes[1], 1/(2*counts[1]))