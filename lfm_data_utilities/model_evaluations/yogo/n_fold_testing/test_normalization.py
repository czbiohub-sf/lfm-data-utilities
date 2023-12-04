import unittest
import numpy as np

from n_fold_test import normalize_confusion_matrix


class NormalizationTests(unittest.TestCase):
    def test_normalization_1(self):
        confusion_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normalized = normalize_confusion_matrix(confusion_matrix)

        self.assertTrue(np.all(normalized.sum(axis=1) == 1))
        self.assertTrue(np.isfinite(normalized).all())

    def test_normalization_2(self):
        confusion_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        normalized = normalize_confusion_matrix(confusion_matrix)

        self.assertTrue(np.all(normalized.sum(axis=1) == 1))

    def test_normalization_3(self):
        confusion_matrix = np.array([[5, 0, 0], [0, 7, 0], [0, 0, 9]])
        normalized = normalize_confusion_matrix(confusion_matrix)

        identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(normalized, identity))
