from pathlib import Path
import subprocess


REQUIRED_FILES = {
    "evaluate.py",
    "make_inputs.py",
    "validate_submission.py",
    "data/release.json",
}


def validate_checkout(root: Path) -> list[str]:
    return sorted(path for path in REQUIRED_FILES if not (root / path).is_file())


def sync(repo: str, revision: str, destination: Path) -> None:
    subprocess.run(
        ["git", "clone", "--filter=blob:none", repo, str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", revision],
        check=True,
    )
    missing = validate_checkout(destination)
    if missing:
        raise RuntimeError(f"challenge checkout is missing: {', '.join(missing)}")
