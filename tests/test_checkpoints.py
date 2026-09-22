import hashlib
from pathlib import Path

import pytest
import torch

from hatchmatch.checkpoints import (
    checkpoint_sha256,
    load_verified_checkpoint,
    save_checkpoint,
)


def test_verified_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {}, "config": {"fold": 0}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert load_verified_checkpoint(path, digest)["config"]["fold"] == 0


def test_bad_expected_hash_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {}}, path)

    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_checkpoint(path, "0" * 64)


def test_saved_checkpoint_has_verified_sha_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "model.pt"

    digest = save_checkpoint(path, {"state_dict": {"weight": torch.tensor([2.0])}})

    assert digest == checkpoint_sha256(path)
    assert path.with_suffix(path.suffix + ".sha256").read_text().split()[0] == digest
    loaded = load_verified_checkpoint(path, digest)
    torch.testing.assert_close(loaded["state_dict"]["weight"], torch.tensor([2.0]))


def test_sidecar_tampering_is_rejected_when_expected_hash_is_omitted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(path, {"state_dict": {}})
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_checkpoint(path)
