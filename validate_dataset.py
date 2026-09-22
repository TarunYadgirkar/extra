"""Verify released annotations against their pinned source pixel counts."""
import argparse
import json
from pathlib import Path
from evaluate import load_manifest, reviewed_domains, data_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="dataset")
    args = p.parse_args()
    root = Path(args.data_root)
    expected = json.loads((root / "expected-counts.json").read_text())
    if "counts" in expected:
        expected = expected["counts"]
    from PIL import Image
    results = {}
    for split in ("train", "val"):
        rows = load_manifest(root / (split + ".json"))
        for row in rows:
            with Image.open(data_path(root, row["image"])) as im:
                assert im.size == (row["width"], row["height"]), row["id"] + ": image dimensions"
            positive, known, blank, _ = reviewed_domains(row, root)
            actual = {"positive": int(positive.sum()), "negative": int((known & ~positive).sum())}
            want = expected[row["id"]]
            for key in actual:
                assert actual[key] == want[key + "_pixels"], (row["id"], key, actual[key], want[key + "_pixels"])
            results[row["id"]] = actual
        print(f"{split}: verified {len(rows)} source-count identities", flush=True)
    print(json.dumps({"verified_queries": len(results), "source_count_mismatches": 0}))


if __name__ == "__main__":
    main()
