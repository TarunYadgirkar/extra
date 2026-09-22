"""Atomic, SHA-256-verified training checkpoint persistence."""

from __future__ import annotations

import hashlib
import hmac
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch


def checkpoint_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a checkpoint file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return Path(f"{path}.sha256")


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Atomically save ``payload`` and an adjacent GNU-style SHA sidecar."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        digest = checkpoint_sha256(temporary)
        os.replace(temporary, destination)
        sidecar = _sidecar_path(destination)
        sidecar_temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
        sidecar_temporary.write_text(
            f"{digest}  {destination.name}\n",
            encoding="ascii",
        )
        os.replace(sidecar_temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _sidecar_digest(path: Path) -> str:
    sidecar = _sidecar_path(path)
    try:
        fields = sidecar.read_text(encoding="ascii").split()
    except FileNotFoundError as exc:
        raise ValueError(f"checkpoint SHA-256 sidecar is missing: {sidecar}") from exc
    if len(fields) < 1 or len(fields[0]) != 64:
        raise ValueError(f"checkpoint SHA-256 sidecar is malformed: {sidecar}")
    return fields[0].lower()


def load_verified_checkpoint(
    path: str | Path,
    expected_sha256: str | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Verify bytes before deserializing and return a checkpoint mapping.

    ``expected_sha256`` can come from a trusted manifest. If it is omitted,
    the adjacent ``.sha256`` sidecar is required.
    """

    checkpoint_path = Path(path)
    expected = (
        _sidecar_digest(checkpoint_path)
        if expected_sha256 is None
        else expected_sha256.lower()
    )
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected SHA-256 must contain 64 hexadecimal characters")
    actual = checkpoint_sha256(checkpoint_path)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected}, computed {actual}"
        )
    loaded = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return loaded
