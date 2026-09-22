"""stdlib tests run by default; RUN_IMAGE_TESTS=1 enables tiny CPU image tests."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate


class ScalarTests(unittest.TestCase):
    def test_document_macro_is_not_query_or_pixel_micro(self):
        rows = [
            {"document_id": "a", "metrics": evaluate.metrics_from_counts(1, 0, 0, 1)},
            {"document_id": "a", "metrics": evaluate.metrics_from_counts(9, 0, 0, 1)},
            {"document_id": "b", "metrics": evaluate.metrics_from_counts(0, 0, 100, 1)},
            {"document_id": "c", "metrics": evaluate.metrics_from_counts(0, 2, 0, 8)},
        ]
        summary = evaluate.aggregate(rows)
        self.assertEqual(summary["document_macro_iou"], 0.5)
        self.assertAlmostEqual(summary["mean_query_iou"], 2 / 3)
        self.assertEqual(summary["global_precision"], 10 / 12)
        self.assertEqual(summary["global_recall"], 10 / 110)
        self.assertEqual(summary["empty_target_false_positive_rate"], 0.2)
        self.assertNotIn("c", summary["per_document_macro_iou"])

    def test_empty_and_undefined_metrics_are_null(self):
        empty = evaluate.metrics_from_counts(0, 0, 0, 8, 0, 5)
        self.assertIsNone(empty["iou"])
        self.assertIsNone(empty["precision"])
        self.assertIsNone(empty["recall"])
        self.assertEqual(empty["empty_target_false_positive_rate"], 0)
        summary = evaluate.aggregate([{"document_id": "empty", "metrics": empty}])
        self.assertIsNone(summary["document_macro_iou"])
        self.assertEqual(summary["blank_false_positive_rate"], 0)
        self.assertNotIn("NaN", json.dumps(summary, allow_nan=False))

    def test_blank_subset_is_distinct_from_all_known_negatives(self):
        result = evaluate.metrics_from_counts(2, 3, 1, 9, 1, 4)
        self.assertEqual(result["blank_false_positive_rate"], 0.25)
        self.assertEqual(result["iou"], 2 / 6)
        self.assertIsNone(result["empty_target_false_positive_rate"])
        for counts in [(0, 0, 0, 0), (1, -1, 0, 0), (True, 0, 0, 1)]:
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.metrics_from_counts(*counts)
        with self.assertRaises(evaluate.EvaluationError):
            evaluate.metrics_from_counts(1, 0, 0, 2, 1, 1)

    def test_manifest_rejects_duplicate_ids_invalid_boxes_and_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            row = {"id": "a", "image": "a.png", "width": 4, "height": 4,
                   "document_id": "d", "kind": "toy", "query_box": [0, 0, 1, 1],
                   "labels": {"positive_boxes": [[2, 2, 3, 3]]}}
            path.write_text(json.dumps([row, row]))
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.load_manifest(path)
            row["query_box"] = [0, 0, 5, 1]
            path.write_text(json.dumps([row]))
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.load_manifest(path)
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.data_path(folder, "../outside.png")

    def test_label_free_manifest_is_valid_for_inference_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            row = {"id": "private", "image": "source.png", "width": 4, "height": 4,
                   "document_id": "d", "kind": "toy", "query_box": [0, 0, 2, 2]}
            path.write_text(json.dumps([row]))
            self.assertEqual(evaluate.load_manifest(path, require_labels=False), [row])
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.load_manifest(path)
            row["labels"] = "unavailable-to-inference"
            path.write_text(json.dumps([row]))
            self.assertEqual(evaluate.load_manifest(path, require_labels=False), [row])


@unittest.skipUnless(os.environ.get("RUN_IMAGE_TESTS") == "1", "set RUN_IMAGE_TESTS=1 for tiny CPU image fixtures")
class ImageTests(unittest.TestCase):
    def setUp(self):
        # Do not import native libraries when this class is skipped.
        import numpy as np
        from PIL import Image
        self.np, self.Image = np, Image
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pred = self.root / "predictions"
        self.pred.mkdir()
        self.manifest = self.root / "val.json"
        self.row = {"id": "toy", "image": "source.png", "width": 4, "height": 4,
                    "document_id": "doc", "kind": "synthetic_test",
                    "query_box": [0, 0, 1, 1], "context_boxes": [[1, 0, 2, 1]],
                    "labels": {"positive_boxes": [[0, 0, 4, 1]],
                               "negative_boxes": [[0, 1, 4, 2]],
                               "blank_boxes": [[0, 1, 2, 2]]}}

    def tearDown(self):
        self.temp.cleanup()

    def score(self, pixels):
        self.manifest.write_text(json.dumps({"schema": "hatch-matching-challenge/v1", "examples": [self.row]}))
        self.Image.fromarray(pixels).save(self.pred / "toy.png")
        return evaluate.evaluate(self.manifest, self.root, self.pred)

    def test_query_context_and_unknown_excluded_but_known_negative_scored(self):
        a = self.np.full((4, 4), 255, dtype=self.np.uint8)
        result = self.score(a)["examples"][0]
        self.assertEqual((result["metrics"]["tp"], result["metrics"]["fp"], result["metrics"]["fn"]), (2, 4, 0))
        self.assertEqual(result["metrics"]["iou"], 1 / 3)
        self.assertEqual(result["metrics"]["blank_false_positive_rate"], 1)
        self.assertEqual(result["unscored_pixels"], 10)
        self.assertEqual(result["excluded_known_pixels"], 2)
        a[0, :2] = 0
        a[2:, :] = 0
        self.assertEqual(self.score(a)["examples"][0]["metrics"], result["metrics"])

    def test_positive_only_never_invents_background_negatives(self):
        self.row["labels"] = {"positive_boxes": [[2, 2, 4, 4]]}
        result = self.score(self.np.full((4, 4), 1, dtype=self.np.uint8))["examples"][0]
        self.assertTrue(result["positive_only"])
        self.assertEqual(result["metrics"]["reviewed_pixels"], 4)
        self.assertEqual(result["metrics"]["fp"], 0)
        self.assertEqual(result["metrics"]["iou"], 1)

    def test_explicit_known_domain_is_authoritative_and_blank_mask_works(self):
        blank = self.np.zeros((4, 4), dtype=self.np.uint8)
        blank[2:, 2:] = 255
        self.Image.fromarray(blank).save(self.root / "blank.png")
        self.row["labels"] = {"known_boxes": [[0, 0, 4, 4]], "blank_mask": "blank.png"}
        result = self.score(self.np.ones((4, 4), dtype=self.np.uint8))["examples"][0]["metrics"]
        self.assertIsNone(result["iou"])
        self.assertEqual(result["fp"], 14)
        self.assertEqual(result["blank_false_positive_pixels"], 4)
        self.assertEqual(result["empty_target_false_positive_rate"], 1)

    def test_invalid_prediction_encodings_and_shape_are_rejected(self):
        for a in [self.np.zeros((3, 4), dtype=self.np.uint8),
                  self.np.zeros((4, 4, 3), dtype=self.np.uint8),
                  self.np.full((4, 4), 127, dtype=self.np.uint8),
                  self.np.array([[1, 255, 0, 0]] * 4, dtype=self.np.uint8)]:
            with self.subTest(shape=a.shape, maximum=int(a.max())):
                with self.assertRaises(evaluate.EvaluationError):
                    self.score(a)

    def test_missing_prediction_and_contradictory_labels_are_rejected(self):
        a = self.np.zeros((4, 4), dtype=self.np.uint8)
        self.score(a)
        (self.pred / "toy.png").unlink()
        with self.assertRaises(evaluate.EvaluationError):
            evaluate.evaluate(self.manifest, self.root, self.pred)
        self.row["labels"]["negative_boxes"] = [[2, 0, 3, 1]]
        with self.assertRaises(evaluate.EvaluationError):
            self.score(a)
        self.row["labels"] = {"positive_boxes": [[2, 2, 4, 4]], "known_boxes": [[0, 0, 2, 2]]}
        with self.assertRaises(evaluate.EvaluationError):
            self.score(a)

    def test_no_known_pixels_after_query_exclusion_is_rejected(self):
        self.row["labels"] = {"positive_boxes": [[0, 0, 1, 1]]}
        with self.assertRaises(evaluate.EvaluationError):
            self.score(self.np.zeros((4, 4), dtype=self.np.uint8))



if __name__ == "__main__":
    unittest.main()
