"""Export only the fields available to an inference program (stdlib only)."""
import argparse
import json
from pathlib import Path
from evaluate import load_manifest

INPUT_KEYS = ("id", "image", "width", "height", "query_box", "context_boxes")


def make_inputs(manifest):
    # The source manifest keeps the scorer's default strict label validation.
    rows = load_manifest(manifest)
    return {"schema": "hatch-matching-inputs/v1", "examples": [
        {key: row[key] for key in INPUT_KEYS if key in row} for row in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Reviewed source manifest")
    parser.add_argument("--output", required=True, help="Label-free input JSON to write")
    args = parser.parse_args()
    result = make_inputs(args.manifest)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['examples'])} input examples to {args.output}")


if __name__ == "__main__":
    main()
