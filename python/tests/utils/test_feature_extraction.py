import unittest
import numpy as np
from utils.feature_extraction import (get_sum_from_i_image, 
                                      get_integral_image, get_feature_values, 
                                      get_v_edge_feature, get_h_edge_feature,
                                      get_v_band_feature, get_h_band_feature,
                                      get_slant_edge_feature,
                                      extract_image_features)


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction from image."""

    def setUp(self):
        """Setup the dataset that will used by the test cases."""

        self.test_image = np.array([[1, 2, 2, 4, 1],
                                    [3, 4, 1, 5, 2],
                                    [2, 3, 3, 2, 4],
                                    [4, 1, 5, 4, 6],
                                    [6, 3, 2, 1, 3],
                                    [255, 1, 2, 3, 4]], dtype=np.uint8)
        self.expected_i_image = np.array([[1,  3,  5,  9, 10],
                                          [4, 10, 13, 22, 25],
                                          [6, 15, 21, 32, 39],
                                          [10, 20, 31, 46, 59],
                                          [16, 29, 42, 58, 74],
                                          [271, 285, 300, 319, 339]])

        self.test_i_image = np.array([[46, 254, 414, 464, 662, 829, 882, 1131, 1210, 1279],
                                      [144,  479,  781,  906, 1339,
                                          1564, 1766, 2214, 2371, 2482],
                                      [339,  726, 1096, 1304, 1952,
                                          2241, 2644, 3324, 3526, 3876],
                                      [477, 1107, 1642, 1946, 2614,
                                          2968, 3412, 4236, 4655, 5060.],
                                      [680, 1384, 1940, 2406, 3146,
                                          3534, 4148, 5106, 5626, 6245],
                                      [779, 1591, 2324, 2875, 3681,
                                          4201, 5006, 6080, 6642, 7503],
                                      [780, 1815, 2788, 3552, 4524,
                                          5169, 6213, 7376, 8071, 9096],
                                      [863, 1911, 2893, 3772, 4966,
                                          5758, 7033, 8259, 9044, 10086],
                                      [1034, 2327, 3338, 4301, 5649,
                                          6524, 8025, 9437, 10438, 11649],
                                      [1220, 2709, 3948, 5020, 6399, 7413, 8994, 10446, 11449, 12677]], dtype=np.int32)

        self.expected_feature_val = { 
                                        0: (0, 0, 1, 1, 'Vertical edge feature'),
                                        8988: (0, 8, 3, 16, 'Vertical edge feature'),
                                        17976: (12, 2, 1, 3, 'Horizontal edge feature'),
                                        26964: (3, 5, 7, 6, 'Horizontal edge feature'),
                                        35952: (4, 1, 1, 7, 'Vertical band feature'),
                                        44940: (0, 0, 6, 11, 'Vertical band feature'),
                                        53927: (14, 7, 12, 1, 'Horizontal band feature'),
                                        63927: (3, 1, 9, 6, 'Slant feature'),
                                        63959: (1, 1, 9, 9, 'Slant feature')
                                    }

    def test_integral_image(self):
        """Test the integral image creation."""

        i_image = get_integral_image(self.test_image)
        np.testing.assert_array_equal(i_image, self.expected_i_image)

    def test_get_sum_from_i_image(self):
        """Test the sum of area from integral image."""
        self.assertEqual(get_sum_from_i_image(0, 0, 2, 2, self.test_i_image), 479)
        self.assertEqual(get_sum_from_i_image(5, 0, 2, 2, self.test_i_image), 431)

    def test_v_edge_feature(self):
        """Test the v edge feature function"""
        self.assertEqual(get_v_edge_feature(
            0, 0, 2, 2, self.test_i_image), 52)
        self.assertEqual(get_v_edge_feature(7, 0, 2, 2, self.test_i_image), 275)
        self.assertEqual(get_v_edge_feature(0, 6, 2, 2, self.test_i_image), 382)
        self.assertEqual(get_v_edge_feature(7, 6, 2, 2, self.test_i_image), 214)


    def test_h_edge_feature(self):
        """Test the h edge feature function"""
        self.assertEqual(get_h_edge_feature(
            0, 0, 2, 2, self.test_i_image), -149)
        self.assertEqual(get_h_edge_feature(6, 0, 2, 2, self.test_i_image), -478)
        self.assertEqual(get_h_edge_feature(0, 7, 2, 2, self.test_i_image), -33)
        self.assertEqual(get_h_edge_feature(6, 7, 2, 2, self.test_i_image), -69)

    def test_v_band_feature(self):
        """Test the v band feature function"""
        self.assertEqual(get_v_band_feature(
            0, 0, 2, 2, self.test_i_image), 710)
        self.assertEqual(get_v_band_feature(6, 0, 2, 2, self.test_i_image), 403)
        self.assertEqual(get_v_band_feature(0, 3, 2, 2, self.test_i_image), 736)
        self.assertEqual(get_v_band_feature(6, 3, 2, 2, self.test_i_image), 349)


    def test_h_band_feature(self):
        """Test the h band feature function"""
        self.assertEqual(get_h_band_feature(
            0, 0, 2, 2, self.test_i_image), 335)
        self.assertEqual(get_h_band_feature(3, 0, 2, 2, self.test_i_image), 739)
        self.assertEqual(get_h_band_feature(0, 6, 2, 2, self.test_i_image), 643)
        self.assertEqual(get_h_band_feature(3, 6, 2, 2, self.test_i_image), 560)


    def test_slant_feature(self):
        """Test the slant edge feature function."""

        self.assertEqual(get_slant_edge_feature(
            0, 0, 2, 2, self.test_i_image), -164)
        self.assertEqual(get_slant_edge_feature(6, 0, 2, 2, self.test_i_image), -605)
        self.assertEqual(get_slant_edge_feature(0, 6, 2, 2, self.test_i_image), 320)
        self.assertEqual(get_slant_edge_feature(6, 6, 2, 2, self.test_i_image), 90)

    def test_get_feature_values(self):
        """Test the Get feature from index."""

        for key in self.expected_feature_val.keys():
            self.assertEqual(get_feature_values(
                19, 19, key), self.expected_feature_val[key])

    def test_extract_image_features(self):
        """Test the extract_image_feature methods."""

        features = extract_image_features(self.test_image)
        self.assertEqual(features[25], 3)
        self.assertEqual(features[64], 4)
        self.assertEqual(features[87], -2)
        self.assertEqual(features[164], 2)
        self.assertEqual(features[218], 9)
        self.assertEqual(features[318], 271)
        self.assertEqual(len(features), 453)

    def test_extract_all_features(self):
        """Test the extract_image_feature methods for all values for test image."""

        image = self.test_image
        row, col = self.test_image.shape
        features = extract_image_features(self.test_image)

        for i in range(len(features)):
            x, y, w, h, feature_type = get_feature_values(row, col, i)
            if feature_type == "Vertical edge feature":
                assert features[i] == np.sum(image[x:x+h, y:y+w], dtype=np.int32) - np.sum(image[x:x+h, y+w:y+2*w], dtype=np.int32)
            elif feature_type == "Horizontal edge feature":
                assert features[i] == np.sum(image[x:x+h, y:y+w], dtype=np.int32) - np.sum(image[x+h:x+2*h, y:y+w], dtype=np.int32)
            elif feature_type == "Vertical band feature":
                assert features[i] == np.sum(image[x:x+h, y:y+w], dtype=np.int32) +  np.sum(image[x:x+h, y+2*w:y+3*w]) - np.sum(image[x:x+h, y+w:y+2*w], dtype=np.int32)
            elif feature_type == "Horizontal band feature":
                assert features[i] == np.sum(image[x:x+h, y:y+w], dtype=np.int32) + np.sum(image[x+2*h:x+3*h, y:y+w], dtype=np.int32) - np.sum(image[x+h:x+2*h, y:y+w], dtype=np.int32)
            else:
                assert features[i] == np.sum(image[x:x+h, y:y+w], dtype=np.int32) + np.sum(image[x+h:x+2*h, y+w:y+2*w], dtype=np.int32) - np.sum(image[x:x+h, y+w:y+2*w], dtype=np.int32) - np.sum(image[x+h:x+2*h, y:y+w], dtype=np.int32)
