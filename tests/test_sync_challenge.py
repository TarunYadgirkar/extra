from pathlib import Path

from scripts.sync_challenge import REQUIRED_FILES, validate_checkout


def test_validate_checkout_rejects_missing_files(tmp_path: Path) -> None:
    missing = validate_checkout(tmp_path)
    assert missing == sorted(REQUIRED_FILES)
