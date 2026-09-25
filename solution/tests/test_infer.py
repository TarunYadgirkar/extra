import sys
import unittest
from unittest.mock import Mock
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer import predict_probabilities, postprocess


class BinnedPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.head = joblib.load(Path(__file__).resolve().parents[1] / 'weights/head.joblib')

    def test_matches_sklearn_at_bin_boundaries_and_missing_values(self):
        thresholds = self.head._bin_mapper.bin_thresholds_
        x = np.column_stack([np.resize(t, max(map(len, thresholds))) for t in thresholds])
        for values in (x, np.nextafter(x, -np.inf), np.nextafter(x, np.inf), x.astype(np.float32)):
            np.testing.assert_array_equal(
                predict_probabilities(self.head, values), self.head.predict_proba(values)[:, 1])
        for value in (np.nan, -np.inf, np.inf):
            values = np.full_like(x, value)
            np.testing.assert_array_equal(
                predict_probabilities(self.head, values), self.head.predict_proba(values)[:, 1])

    def test_preserves_other_estimator_prediction_path(self):
        head = Mock()
        head.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
        values = np.zeros((2, 162), np.float32)
        np.testing.assert_array_equal(predict_probabilities(head, values), [0.2, 0.9])
        head.predict_proba.assert_called_once_with(values)

    def test_does_not_modify_input_or_model(self):
        x = np.zeros((12, self.head.n_features_in_), np.float32)
        before = joblib.hash(self.head)
        predict_probabilities(self.head, x)
        np.testing.assert_array_equal(x, 0)
        self.assertEqual(joblib.hash(self.head), before)


class PostprocessTests(unittest.TestCase):
    def test_fills_large_enclosed_hole_but_preserves_border_background(self):
        g = np.zeros((400, 400), np.float32)
        g[10:390, 10:390] = 0.9
        g[40:360, 40:360] = 0.1
        out = postprocess(g, (1597, 1599), 0.5)
        self.assertEqual(out.shape, (1597, 1599))
        self.assertEqual(out.dtype, np.bool_)
        self.assertTrue(out[800, 800])
        self.assertFalse(out[0, 0])

    def test_uniform_maps_and_threshold(self):
        for value, expected in ((0.1, False), (0.8, True)):
            out = postprocess(np.full((8, 9), value, np.float32), (31, 34), 0.6)
            self.assertTrue(np.all(out == expected))


if __name__ == '__main__':
    unittest.main()
