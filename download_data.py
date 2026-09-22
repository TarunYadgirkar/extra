#!/usr/bin/env python3
"""Download and verify the public challenge dataset (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\\" in value:
        raise ValueError(f"Unsafe archive path: {value!r}")
    return p


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(root):
    checks = json.loads((root / "checksums.json").read_text())
    for name, item in checks.items():
        p = root.joinpath(*safe_relative(name).parts)
        if p.is_symlink() or not p.is_file() or p.stat().st_size != item["bytes"] or digest(p) != item["sha256"]:
            raise ValueError(f"Integrity check failed: {name}")
    print(f"Verified {len(checks)} dataset files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output)
        return
    release = json.loads((Path(__file__).parent / "data" / "release.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit("Output already exists. Use --verify-only, or choose a new --output directory.")
    required = release["bytes"] + release["unpacked_bytes"] + 128 * 1024 * 1024
    if shutil.disk_usage(args.output.parent).free < required:
        raise SystemExit(f"Need at least {required:,} free bytes for download and extraction.")
    with tempfile.TemporaryDirectory(prefix="hatch-download-", dir=args.output.parent) as temporary:
        archive = Path(temporary) / "dataset.zip"
        h = hashlib.sha256()
        count = 0
        request = urllib.request.Request(release["url"], headers={"User-Agent": "TruTec-Hatch-Challenge/1"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as f:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > release["bytes"]:
                    raise ValueError("Download exceeds pinned size")
                h.update(block)
                f.write(block)
        if count != release["bytes"] or h.hexdigest() != release["sha256"]:
            raise ValueError("Archive checksum/size mismatch; nothing installed.")
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        with zipfile.ZipFile(archive) as z:
            names = set()
            total = 0
            for member in z.infolist():
                relative = safe_relative(member.filename)
                if member.filename in names or member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Unexpected archive entry")
                names.add(member.filename)
                total += member.file_size
                if total > release["unpacked_bytes"]:
                    raise ValueError("Unpacked size exceeds pinned limit")
                target = stage.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
            if total != release["unpacked_bytes"]:
                raise ValueError("Unpacked size mismatch")
        verify(stage)
        stage.rename(args.output)
    print(f"Ready: {args.output}/train.json and {args.output}/val.json")


if __name__ == "__main__":
    main()
