"""Snapshot/verify protected research files during frontend-only work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected():
    paths = list((ROOT / "src/component").rglob("*.py"))
    paths += [p for folder in ["results", "configs", "data/raw/chbmit"]
              for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix != ".pyc"]
    return {p.relative_to(ROOT).as_posix(): fingerprint(p) for p in sorted(paths)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["before", "after"])
    args = parser.parse_args()
    current = protected()
    if args.phase == "before":
        with (HERE / "protected_before.json").open("x", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        for path in list((ROOT / "demo").glob("*.py")) + [ROOT / ".streamlit/config.toml"]:
            destination = HERE / "before_code" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    else:
        before = json.loads((HERE / "protected_before.json").read_text())
        assert before == current, "Protected research files changed"
        with (HERE / "protected_after.json").open("w", encoding="utf-8") as f:
            json.dump({"unchanged": True, "files_verified": len(current), "sha256": current}, f, indent=2)
    print(f"{args.phase}: {len(current)} protected files verified")
