"""Validate submission metadata only; never run commands or upload files."""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
from urllib.parse import urlsplit

FIELDS = {"schema_version", "repository", "commit", "inference_command", "environment",
          "weights", "validation_metrics", "hardware", "runtime_seconds", "peak_memory_mb",
          "time_spent_hours", "external_resources", "ai_tools"}


class SubmissionError(ValueError):
    """Submission metadata is incomplete or malformed."""


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise SubmissionError(name + " must be a nonempty string")
    if "REPLACE_" in value or value.strip().startswith("<"):
        raise SubmissionError(name + " still contains a placeholder")
    return value


def https_url(value, name):
    text(value, name)
    try:
        url = urlsplit(value)
        host = url.hostname
        _ = url.port
    except ValueError as exc:
        raise SubmissionError(name + " must be a valid HTTPS URL") from exc
    if (url.scheme != "https" or not host or url.username or url.password
            or any(c.isspace() for c in value) or host.endswith(".invalid")):
        raise SubmissionError(name + " must be a valid HTTPS URL")


def relative_file(value, name):
    text(value, name)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value or value.endswith("/") or str(path) == ".":
        raise SubmissionError(name + " must be a relative file path within the submission")


def nonnegative(value, name, nullable=False):
    if nullable and value is None:
        return
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise SubmissionError(name + " must be a finite nonnegative number" + (" or null" if nullable else ""))


def validate_submission(obj):
    if not isinstance(obj, dict):
        raise SubmissionError("Submission must be a JSON object")
    missing, extra = FIELDS - obj.keys(), obj.keys() - FIELDS
    if missing or extra:
        raise SubmissionError("Submission fields differ: missing=" + str(sorted(missing)) + "; unknown=" + str(sorted(extra)))
    if obj["schema_version"] != "1.0":
        raise SubmissionError("schema_version must be the string 1.0")
    https_url(obj["repository"], "repository")
    if not isinstance(obj["commit"], str) or not re.fullmatch(r"[0-9a-fA-F]{40}", obj["commit"]):
        raise SubmissionError("commit must contain exactly 40 hexadecimal characters")
    command = text(obj["inference_command"], "inference_command")
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise SubmissionError("inference_command has invalid quoting") from exc
    if not tokens or tokens[0].startswith("--"):
        raise SubmissionError("inference_command must name a program")
    for flag in ("--inputs", "--data-root", "--output-dir"):
        matches = [i for i, token in enumerate(tokens) if token == flag or token.startswith(flag + "=")]
        if len(matches) != 1:
            raise SubmissionError("inference_command must contain exactly one " + flag)
        index = matches[0]
        value = tokens[index].split("=", 1)[1] if "=" in tokens[index] else (tokens[index + 1] if index + 1 < len(tokens) else "")
        if not value or value.startswith("--"):
            raise SubmissionError(flag + " requires a value")
    relative_file(obj["environment"], "environment")
    relative_file(obj["validation_metrics"], "validation_metrics")
    text(obj["hardware"], "hardware")
    for key in ("runtime_seconds", "time_spent_hours"):
        nonnegative(obj[key], key)
    nonnegative(obj["peak_memory_mb"], "peak_memory_mb", nullable=True)
    if not isinstance(obj["weights"], list):
        raise SubmissionError("weights must be a list (empty when none are needed)")
    for weight in obj["weights"]:
        if not isinstance(weight, dict) or not {"url", "sha256"} <= set(weight) or set(weight) - {"url", "sha256", "bytes"}:
            raise SubmissionError("Each weight needs url and sha256, with optional bytes")
        https_url(weight["url"], "weight url")
        if not isinstance(weight["sha256"], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", weight["sha256"]):
            raise SubmissionError("Weight sha256 must contain exactly 64 hexadecimal characters")
        if "bytes" in weight and (type(weight["bytes"]) is not int or weight["bytes"] <= 0):
            raise SubmissionError("Weight bytes must be a positive integer")
    for key in ("external_resources", "ai_tools"):
        if not isinstance(obj[key], list):
            raise SubmissionError(key + " must be a list")
        for entry in obj[key]:
            text(entry, key + " entry")
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, help="Completed submission JSON")
    args = parser.parse_args()
    try:
        validate_submission(json.loads(Path(args.submission).read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        parser.exit(2, "Invalid submission: " + str(exc) + "\n")
    print("Submission metadata is valid. No command was executed or file uploaded.")


if __name__ == "__main__":
    main()
