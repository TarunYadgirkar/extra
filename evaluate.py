"""Score query-conditioned masks on reviewed pixels. No model code is required.

Image dependencies are deliberately lazy: counts and aggregation use stdlib only.
Run ``python evaluate.py --help`` for the prediction directory contract.
"""
import argparse
import json
import math
import re
from pathlib import Path


class EvaluationError(ValueError):
    """Invalid dataset or prediction; never silently repair either."""


def data_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise EvaluationError("Dataset paths must be nonempty relative paths")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise EvaluationError("Dataset path escapes --data-root: " + relative)
    return path


def validate_boxes(boxes, width, height, name):
    if not isinstance(boxes, list):
        raise EvaluationError(name + " must be a list")
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise EvaluationError(name + " needs integer [x0,y0,x1,y1] boxes")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise EvaluationError(name + " box is empty or outside the image")
    return boxes


def load_manifest(path, require_labels=True):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = obj.get("examples") if isinstance(obj, dict) else obj
    if not isinstance(rows, list) or not rows:
        raise EvaluationError("Manifest must be a nonempty list or an object with examples")
    ids = set()
    allowed_labels = {kind + suffix for kind in ("positive", "negative", "known", "blank")
                      for suffix in ("_mask", "_boxes")}
    for row in rows:
        if not isinstance(row, dict):
            raise EvaluationError("Each example must be an object")
        ident = row.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", ident):
            raise EvaluationError("Example id must be a safe, nonempty filename stem")
        if ident in ids:
            raise EvaluationError("Duplicate example id: " + ident)
        ids.add(ident)
        required_metadata = ("image", "document_id", "kind") if require_labels else ("image",)
        for key in required_metadata:
            if not isinstance(row.get(key), str) or not row[key]:
                raise EvaluationError(ident + ": missing " + key)
        width, height = row.get("width"), row.get("height")
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise EvaluationError(ident + ": width and height must be positive integers")
        validate_boxes([row.get("query_box")], width, height, ident + " query_box")
        validate_boxes(row.get("context_boxes", []), width, height, ident + " context_boxes")
        if not require_labels:
            # Inference needs only the input contract; private labels may be
            # absent, and any supplied labels must not influence prediction.
            continue
        labels = row.get("labels")
        if not isinstance(labels, dict) or not labels:
            raise EvaluationError(ident + ": reviewed labels are required")
        if set(labels) - allowed_labels:
            raise EvaluationError(ident + ": unrecognized label fields: " + str(sorted(set(labels) - allowed_labels)))
        for key, value in labels.items():
            if key.endswith("_boxes"):
                validate_boxes(value, width, height, ident + " " + key)
            elif not isinstance(value, str) or not value:
                raise EvaluationError(ident + ": mask path must be a nonempty string")
    return rows


def load_binary_png(path, width, height):
    import numpy as np
    from PIL import Image

    try:
        with Image.open(path) as im:
            if im.format != "PNG" or im.mode not in ("1", "L"):
                raise EvaluationError(str(path) + ": expected a grayscale binary PNG (mode 1 or L)")
            if im.size != (width, height):
                raise EvaluationError(str(path) + ": wrong shape; resizing is not allowed")
            raw = np.asarray(im)
            # Two supported encodings; mixed 1/255 and probability-like values fail.
            values = set(int(v) for v in np.unique(raw))
            if not (values <= {0, 1} or values <= {0, 255}):
                raise EvaluationError(str(path) + ": expected only 0/1 or 0/255 pixels")
            return raw != 0
    except (OSError, FileNotFoundError) as exc:
        raise EvaluationError("Cannot read mask " + str(path) + ": " + str(exc)) from exc


def reviewed_domains(row, data_root):
    """Explicit known domains are authoritative; otherwise use labeled domains only.

    Boxes and masks of the same kind are unioned. A positive-only row without an
    explicit known domain has no scored negatives. Blank is a reviewed negative
    subset, not a synonym for every pixel outside a positive polygon.
    """
    import numpy as np

    width, height = row["width"], row["height"]
    labels = row["labels"]
    domains = {}
    for kind in ("positive", "negative", "blank", "known"):
        domain = np.zeros((height, width), dtype=bool)
        if kind + "_mask" in labels:
            domain |= load_binary_png(data_path(data_root, labels[kind + "_mask"]), width, height)
        for x0, y0, x1, y1 in labels.get(kind + "_boxes", []):
            domain[y0:y1, x0:x1] = True
        domains[kind] = domain
    positive, negative, blank = (domains[k] for k in ("positive", "negative", "blank"))
    if (positive & (negative | blank)).any():
        raise EvaluationError(row["id"] + ": positive and negative/blank labels overlap")
    explicit_known = "known_mask" in labels or "known_boxes" in labels
    known = domains["known"] if explicit_known else (positive | negative | blank)
    if ((positive | negative | blank) & ~known).any():
        raise EvaluationError(row["id"] + ": label lies outside the explicit known domain")
    original_known = int(known.sum())
    for x0, y0, x1, y1 in [row["query_box"]] + row.get("context_boxes", []):
        known[y0:y1, x0:x1] = False
    positive &= known
    blank &= known
    if not known.any():
        raise EvaluationError(row["id"] + ": no scored pixels remain after input-query exclusions")
    metadata = {
        "original_known_pixels": original_known,
        "excluded_known_pixels": original_known - int(known.sum()),
        "unscored_pixels": width * height - int(known.sum()),
        "unknown_pixels_before_query_exclusion": width * height - original_known,
        "positive_only": bool(positive.any() and not (known & ~positive).any()),
    }
    return positive, known, blank, metadata


def metrics_from_counts(tp, fp, fn, tn, blank_fp=0, blank_pixels=0):
    """Pure scalar implementation. Empty-target IoU is undefined, never 1."""
    numbers = (tp, fp, fn, tn, blank_fp, blank_pixels)
    if any(type(v) is not int or v < 0 for v in numbers):
        raise EvaluationError("Counts must be nonnegative integers")
    if blank_fp > min(fp, blank_pixels) or blank_pixels > fp + tn:
        raise EvaluationError("Reviewed blank counts are not a negative-domain subset")
    known = tp + fp + fn + tn
    if not known:
        raise EvaluationError("Cannot score an empty known domain")
    positive = tp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "reviewed_pixels": known, "positive_pixels": positive,
        "iou": tp / (tp + fp + fn) if positive else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positive if positive else None,
        "positive_target_available": bool(positive),
        "empty_target_false_positive_rate": fp / known if not positive else None,
        "blank_false_positive_pixels": blank_fp, "reviewed_blank_pixels": blank_pixels,
        "blank_false_positive_rate": blank_fp / blank_pixels if blank_pixels else None,
    }


def aggregate(rows):
    """Document-macro averages nonempty query IoUs within each document first."""
    if not rows:
        raise EvaluationError("No evaluated examples")
    documents = {}
    empty = []
    sums = dict.fromkeys(("tp", "fp", "fn", "tn", "blank_false_positive_pixels", "reviewed_blank_pixels"), 0)
    query_ious = []
    for row in rows:
        m = row["metrics"]
        for key in sums:
            sums[key] += m[key]
        if m["iou"] is None:
            empty.append(m)
        else:
            query_ious.append(m["iou"])
            documents.setdefault(row["document_id"], []).append(m["iou"])
    mean = lambda values: math.fsum(values) / len(values) if values else None
    per_document = {key: mean(values) for key, values in sorted(documents.items())}
    tp, fp, fn = (sums[k] for k in ("tp", "fp", "fn"))
    blank_pixels = sums["reviewed_blank_pixels"]
    empty_known = sum(m["reviewed_pixels"] for m in empty)
    return {
        "examples": len(rows), "documents": len({r["document_id"] for r in rows}),
        "nonempty_target_examples": len(query_ious), "empty_target_examples": len(empty),
        "positive_only_examples": sum(bool(r.get("positive_only", False)) for r in rows),
        "document_macro_iou": mean(list(per_document.values())),
        "per_document_macro_iou": per_document, "mean_query_iou": mean(query_ious),
        "global_precision": tp / (tp + fp) if tp + fp else None,
        "global_recall": tp / (tp + fn) if tp + fn else None,
        "empty_target_false_positive_pixels": sum(m["fp"] for m in empty),
        "empty_target_reviewed_pixels": empty_known,
        "empty_target_false_positive_rate": sum(m["fp"] for m in empty) / empty_known if empty_known else None,
        "blank_false_positive_rate": sums["blank_false_positive_pixels"] / blank_pixels if blank_pixels else None,
        **sums,
    }


def evaluate(manifest, data_root, predictions):
    results = []
    for row in load_manifest(manifest):
        positive, known, blank, metadata = reviewed_domains(row, data_root)
        mask = load_binary_png(Path(predictions) / (row["id"] + ".png"), row["width"], row["height"])
        tp = int((mask & positive).sum())
        fp = int((mask & known & ~positive).sum())
        fn = int((~mask & positive).sum())
        tn = int((~mask & known & ~positive).sum())
        results.append({
            "id": row["id"], "document_id": row["document_id"], "kind": row["kind"],
            **metadata,
            "metrics": metrics_from_counts(tp, fp, fn, tn, int((mask & blank).sum()), int(blank.sum())),
        })
    return {
        "schema": "hatch-matching-challenge-metrics/v1",
        "protocol": "Known pixels only; supplied query/context boxes excluded; document-macro IoU over nonempty targets",
        "whole_sheet_accuracy_certified": False,
        "summary": aggregate(results), "examples": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate <id>.png binary predictions at native resolution.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.manifest, args.data_root, args.predictions)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (EvaluationError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, "Evaluation failed: " + str(exc) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
