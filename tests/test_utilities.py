"""Pure stdlib checks for input export and submission metadata."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate
import make_inputs
import validate_submission as submission


class InputExportTests(unittest.TestCase):
    def test_strips_non_input_fields_and_roundtrips_inference_loader(self):
        row = {"id": "q1", "image": "assets/source.png", "width": 4, "height": 4,
               "query_box": [0, 0, 1, 1], "context_boxes": [[1, 0, 2, 1]],
               "document_id": "private-doc", "kind": "real",
               "labels": {"positive_boxes": [[2, 2, 3, 3]]},
               "expected_counts": {"positive": 1}, "internal_note": "private"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"examples": [row]}))
            result = make_inputs.make_inputs(path)
            self.assertEqual(set(result), {"schema", "examples"})
            self.assertEqual(result["examples"], [{key: row[key] for key in make_inputs.INPUT_KEYS}])
            self.assertEqual(json.loads(path.read_text())["examples"][0], row)
            inputs = Path(directory) / "inputs.json"
            inputs.write_text(json.dumps(result))
            self.assertEqual(evaluate.load_manifest(inputs, require_labels=False), result["examples"])
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.load_manifest(inputs)

    def test_source_manifest_still_requires_valid_labels(self):
        row = {"id": "q", "image": "a.png", "width": 4, "height": 4,
               "query_box": [0, 0, 1, 1], "document_id": "d", "kind": "real"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps([row]))
            with self.assertRaises(evaluate.EvaluationError):
                make_inputs.make_inputs(path)
            row["labels"] = {"positive_boxes": [[2, 2, 5, 3]]}
            path.write_text(json.dumps([row]))
            with self.assertRaises(evaluate.EvaluationError):
                make_inputs.make_inputs(path)


class SubmissionTests(unittest.TestCase):
    def valid(self):
        return {"schema_version": "1.0", "repository": "https://github.com/example/submission",
                "commit": "a" * 40,
                "inference_command": "python infer.py --inputs inputs.json --data-root data --output-dir predictions",
                "environment": "requirements.txt", "weights": [],
                "validation_metrics": "results/validation.json", "hardware": "Declared hardware",
                "runtime_seconds": 0, "peak_memory_mb": None, "time_spent_hours": 1.5,
                "external_resources": [], "ai_tools": []}

    def test_completed_metadata_and_weight_pins(self):
        value = self.valid()
        self.assertIs(submission.validate_submission(value), value)
        value["weights"] = [{"url": "https://example.org/weights.bin", "sha256": "b" * 64, "bytes": 42}]
        value["inference_command"] = "python infer.py --inputs=inputs.json --data-root=data --output-dir=predictions"
        value["peak_memory_mb"] = 128
        submission.validate_submission(value)
        del value["weights"][0]["bytes"]
        submission.validate_submission(value)

    def test_missing_placeholders_bad_types_and_nonfinite_numbers(self):
        bad = [
            ("schema_version", 1.0), ("repository", "http://example.org/repo"),
            ("repository", "REPLACE_WITH_URL"), ("repository", "https://example.invalid/repo"),
            ("commit", "a" * 39), ("commit", "g" * 40), ("hardware", ""),
            ("environment", "../outside.txt"), ("environment", "/tmp/environment.txt"),
            ("validation_metrics", ""), ("runtime_seconds", True),
            ("runtime_seconds", float("nan")), ("runtime_seconds", -1),
            ("time_spent_hours", float("inf")), ("peak_memory_mb", -1),
            ("weights", {}), ("weights", [{"url": "https://example.org/w", "sha256": "x"}]),
            ("weights", [{"url": "https://example.org/w", "sha256": "b" * 64, "bytes": True}]),
            ("external_resources", "none"), ("ai_tools", [""]),
        ]
        for key, value in bad:
            with self.subTest(key=key, value=value):
                obj = self.valid(); obj[key] = value
                with self.assertRaises(submission.SubmissionError):
                    submission.validate_submission(obj)
        obj = self.valid(); del obj["environment"]
        with self.assertRaises(submission.SubmissionError):
            submission.validate_submission(obj)
        sample = Path(__file__).resolve().parents[1] / "submission.example.json"
        with self.assertRaises(submission.SubmissionError):
            submission.validate_submission(json.loads(sample.read_text()))

    def test_inference_flags_are_metadata_only_and_require_values(self):
        for command in ("", "python infer.py --inputs x --data-root d",
                        "python infer.py --inputs --data-root d --output-dir p",
                        "python infer.py --inputs=x --data-root=d --output-dir=",
                        "python infer.py --inputs=x --inputs=y --data-root=d --output-dir=p"):
            with self.subTest(command=command):
                obj = self.valid(); obj["inference_command"] = command
                with self.assertRaises(submission.SubmissionError):
                    submission.validate_submission(obj)
        obj = self.valid()
        # Nonexistent program/paths are valid metadata; validation never executes or reads them.
        obj["inference_command"] = "nonexistent-command --inputs absent --data-root absent --output-dir absent"
        submission.validate_submission(obj)


if __name__ == "__main__":
    unittest.main()
