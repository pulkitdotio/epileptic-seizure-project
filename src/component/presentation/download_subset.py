"""Selected CHB-MIT files, metadata preflight and checksum-verified resumable transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from src.component.dataloaders.chbmit.download import parse_summary_text

BASE = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/"


def selected_files(files):
    selected = []
    for value in files:
        match = re.fullmatch(r"(chb\d{2})/(chb\d{2})_\d+[a-z]?\.edf", value)
        if not match or match[1] != match[2]:
            raise ValueError("Use explicit CHB-MIT files such as chb01/chb01_03.edf. Wildcards and paths outside the subset are rejected.")
        if value not in selected:
            selected.append(value)
    for case in sorted({value.split("/")[0] for value in selected}):
        selected.append(f"{case}/{case}-summary.txt")
    return selected


def load_manifest(path):
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        records = manifest.get("recordings", [])
        if not records or len({r["path"] for r in records}) != len(records):
            raise ValueError("Manifest needs unique selected recordings.")
        selected_files([r["path"] for r in records])
        for record in records + list(manifest.get("summaries", {}).values()):
            if type(record["size_bytes"]) is not int or record["size_bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
                raise ValueError("Manifest needs positive sizes and SHA-256 hashes.")
        for name in manifest.get("summaries", {}):
            if not re.fullmatch(r"(chb\d{2})/\1-summary\.txt", name):
                raise ValueError("Invalid manifest summary path.")
        return manifest
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid subset manifest: {exc}") from exc


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_plan(selected, destination, manifest=None):
    """HEAD only for EDFs; small summaries/checksum list fetched to validate the plan."""
    expected = {}
    if manifest:
        expected.update({r["path"]: r for r in manifest["recordings"]})
        expected.update(manifest.get("summaries", {}))
    else:
        with urllib.request.urlopen(BASE + "SHA256SUMS.txt", timeout=30) as response:
            content = response.read(200001)
        if len(content) > 200000:
            raise ValueError("Checksum list exceeds expected size.")
        expected = {line.split()[1].lstrip("*").removeprefix("./"): {"sha256": line.split()[0]}
                    for line in content.decode("ascii").splitlines()}
    plan, summaries = [], {}
    for value in selected:
        if value not in expected:
            raise ValueError(f"No official checksum for selected file: {value}")
        target = (destination / value).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Destination path escapes the selected dataset root.")
        with urllib.request.urlopen(urllib.request.Request(BASE + value, method="HEAD"), timeout=30) as response:
            size = int(response.headers.get("Content-Length", 0))
        if size <= 0 or ("size_bytes" in expected[value] and size != expected[value]["size_bytes"]):
            raise ValueError(f"Mirror size missing or changed: {value}")
        if target.exists() and (not target.is_file() or target.stat().st_size != size or sha256(target) != expected[value]["sha256"]):
            raise ValueError(f"Existing file size/checksum mismatch: {target}. Preserved; no overwrite.")
        partial = target.with_suffix(target.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Partial file larger than remote recording: {partial}")
        plan.append({"path": value, "size_bytes": size, "sha256": expected[value]["sha256"],
                     "existing": target.exists(), "resume_bytes": offset,
                     "transfer_bytes": 0 if target.exists() else size - offset})
        if value.endswith(".txt"):
            # Annotation preflight is a tiny metadata transfer, not an EDF download.
            with urllib.request.urlopen(BASE + value, timeout=30) as response:
                content = response.read(2 * 1024 * 1024 + 1)
            if len(content) != size or hashlib.sha256(content).hexdigest() != expected[value]["sha256"]:
                raise ValueError(f"Annotation size/checksum mismatch: {value}")
            summaries[value.split("/")[0]] = parse_summary_text(content.decode("utf-8-sig"))
    for item in plan:
        if item["path"].endswith(".edf"):
            case, name = item["path"].split("/")
            if name not in summaries[case]:
                raise ValueError(f"No matching annotation entry for {name}.")
            item["events"] = summaries[case][name]
            if manifest:
                declared = expected[item["path"]].get("events")
                if declared is not None and [list(e) for e in item["events"]] != declared:
                    raise ValueError(f"Annotation intervals changed for {name}.")
    return plan


def transfer_file(item, destination):
    target = destination / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    lock = target.with_suffix(target.suffix + ".download.lock")
    # Exclusive lock avoids two processes appending to the same resumed file.
    with lock.open("x"):
        pass
    try:
        if target.exists():
            raise ValueError(f"Refusing to overwrite existing file: {target}")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset < item["size_bytes"]:
            request = urllib.request.Request(BASE + item["path"], headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=30) as response:
                if offset and (response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{item['size_bytes'] - 1}/{item['size_bytes']}"):
                    raise ValueError("Mirror did not confirm the requested resume range; partial file preserved.")
                with partial.open("ab" if offset else "xb") as handle:
                    transferred = offset
                    while chunk := response.read(1024 * 1024):
                        if transferred + len(chunk) > item["size_bytes"]:
                            raise ValueError("Mirror exceeded advertised size; transfer stopped.")
                        handle.write(chunk)
                        transferred += len(chunk)
        if partial.stat().st_size != item["size_bytes"] or sha256(partial) != item["sha256"]:
            raise ValueError("Incomplete or checksum-invalid transfer. Partial preserved for inspection/resume.")
        partial.rename(target)  # Windows refuses an existing destination; never overwrite.
        print(f"VERIFIED {item['path']}: {item['size_bytes'] / 1024**2:.2f} MiB", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def download_selected(files, destination, download=False, max_mib=500, *, estimate=False, manifest=None):
    selected = selected_files(files)
    destination = Path(destination).resolve()
    pending = [value for value in selected if not (destination / value).exists()]
    for value in selected:
        print(("SKIP existing " if value not in pending else "SELECT ") + value)
    if not download and not estimate:
        print("Dry run: no network requests or downloads. Add --download to fetch these files and summaries.")
        return []
    if max_mib <= 0:
        raise ValueError("Download budget must be positive.")
    plan = metadata_plan(selected, destination, manifest)
    total = sum(item["transfer_bytes"] for item in plan)
    print(f"Verified metadata: {sum(p['size_bytes'] for p in plan) / 1024**2:.2f} MiB selected; "
          f"{total / 1024**2:.2f} MiB remaining transfer.")
    for item in plan:
        print(json.dumps(item))
    if not download:
        print("Estimate only: metadata fetched; no EDF or annotation files written. Approval required before --download.")
        return plan
    if total > max_mib * 1024 * 1024:
        raise ValueError(f"Selected files exceed {max_mib} MiB budget. Narrow the subset or explicitly raise --max-mib.")
    for item in plan:
        if not item["existing"]:
            transfer_file(item, destination)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--files", nargs="+")
    source.add_argument("--manifest", type=Path)
    p.add_argument("--patients", nargs="+", help="Filter an explicit manifest to these patients; never selects entire patient folders.")
    p.add_argument("--destination", type=Path, default=Path("data/raw/chbmit"))
    p.add_argument("--max-mib", type=int, default=500)
    p.add_argument("--download", action="store_true")
    p.add_argument("--estimate", action="store_true", help="Read mirror metadata and verify existing files, without downloading EDFs.")
    args = p.parse_args()
    try:
        manifest = load_manifest(args.manifest) if args.manifest else None
        files = args.files or [r["path"] for r in manifest["recordings"]]
        if args.patients:
            if not manifest or not set(args.patients) <= {r["path"].split('/')[0] for r in manifest["recordings"]}:
                raise ValueError("--patients must name selected patients present in an explicit manifest.")
            files = [f for f in files if f.split('/')[0] in args.patients]
        download_selected(files, args.destination, args.download, args.max_mib, estimate=args.estimate, manifest=manifest)
    except (ValueError, OSError) as exc:
        p.exit(2, f"Download unavailable: {exc}\n")


if __name__ == "__main__":
    main()
