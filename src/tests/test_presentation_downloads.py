"""HTTP fixtures verify transfer safeguards without contacting the dataset mirror."""
import hashlib
import io
import json
from pathlib import Path

import pytest

from src.component.presentation.download_subset import load_manifest, metadata_plan, download_selected, transfer_file
from src.component.presentation.train import discover
from src.component.presentation.pipeline import PreprocessConfig
from src.tests.test_presentation_pipeline import write_test_edf, summary


class Response(io.BytesIO):
    def __init__(self, content=b"", headers=None, status=200):
        super().__init__(content)
        self.headers, self.status = headers or {}, status


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    name = "chb01/chb01_01.edf"
    content = write_test_edf(tmp_path / name).read_bytes()
    text = summary("chb01_01.edf").encode()
    data = {name: content, "chb01/chb01-summary.txt": text}
    manifest = {"recordings": [{"path": name, "sha256": hashlib.sha256(content).hexdigest(),
                                "size_bytes": len(content), "events": [[4., 8.]]}],
                "summaries": {"chb01/chb01-summary.txt": {"size_bytes": len(text), "sha256": hashlib.sha256(text).hexdigest()}}}
    calls = []
    def urlopen(request, **kwargs):
        url = request if isinstance(request, str) else request.full_url
        relative = url.split("/1.0.0/")[-1]
        method = request.get_method() if not isinstance(request, str) else "GET"
        offset = request.get_header("Range") if not isinstance(request, str) else None
        calls.append((method, relative, offset))
        raw = data[relative]
        if method == "HEAD":
            return Response(headers={"Content-Length": str(len(raw))})
        if offset:
            start = int(offset.removeprefix("bytes=").removesuffix("-"))
            return Response(raw[start:], {"Content-Range": f"bytes {start}-{len(raw)-1}/{len(raw)}"}, 206)
        return Response(raw)
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return name, data, manifest, calls


def test_estimate_fetches_metadata_only_and_validates_annotations(mirror, tmp_path):
    name, _, manifest, calls = mirror
    destination = tmp_path / "new"
    plan = download_selected([name], destination, estimate=True, manifest=manifest)
    assert plan[0]["events"] == [(4., 8.)]
    assert not destination.exists()
    assert ("GET", name, None) not in calls
    assert ("HEAD", name, None) in calls
    manifest["recordings"][0]["events"] = [[5., 8.]]
    with pytest.raises(ValueError, match="intervals changed"):
        download_selected([name], destination, estimate=True, manifest=manifest)


def test_budget_checked_before_edf_transfer_and_skips_verified_files(mirror, tmp_path):
    name, data, manifest, calls = mirror
    destination = tmp_path / "new"
    with pytest.raises(ValueError, match="exceed"):
        download_selected([name], destination, download=True, max_mib=.0001, manifest=manifest)
    assert ("GET", name, None) not in calls and not destination.exists()
    (destination / name).parent.mkdir(parents=True)
    (destination / name).write_bytes(data[name])
    plan = download_selected([name], destination, download=True, max_mib=1, manifest=manifest)
    assert plan[0]["existing"] and ("GET", name, None) not in calls
    assert (destination / "chb01/chb01-summary.txt").read_bytes() == data["chb01/chb01-summary.txt"]


def test_resume_range_checksum_and_existing_preservation(mirror, tmp_path):
    name, data, manifest, calls = mirror
    destination = tmp_path / "new"
    target = destination / name
    target.parent.mkdir(parents=True)
    partial = target.with_suffix(".edf.part")
    partial.write_bytes(data[name][:300])
    plan = download_selected([name], destination, download=True, manifest=manifest)
    assert plan[0]["resume_bytes"] == 300 and ("GET", name, "bytes=300-") in calls
    assert target.read_bytes() == data[name] and not partial.exists()
    assert not target.with_suffix(".edf.download.lock").exists()
    target.write_bytes(b"user existing file")
    with pytest.raises(ValueError, match="Existing file size/checksum mismatch"):
        download_selected([name], destination, download=True, manifest=manifest)
    assert target.read_bytes() == b"user existing file"


def test_resume_rejects_wrong_range_and_checksum(mirror, tmp_path, monkeypatch):
    name, data, manifest, _ = mirror
    destination = tmp_path / "new"
    target = destination / name
    target.parent.mkdir(parents=True)
    partial = target.with_suffix(".edf.part")
    partial.write_bytes(data[name][:100])
    item = {"path": name, **manifest["recordings"][0]}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Response(data[name], status=200))
    with pytest.raises(ValueError, match="resume range"):
        transfer_file(item, destination)
    assert partial.read_bytes() == data[name][:100] and not target.exists()
    partial.write_bytes(b"x" * len(data[name]))
    with pytest.raises(ValueError, match="checksum-invalid"):
        transfer_file(item, destination)
    assert not target.exists() and partial.exists()


def test_training_manifest_excludes_unselected_edfs_and_rejects_incomplete(mirror, tmp_path):
    name, data, manifest, _ = mirror
    root = tmp_path / "train_data"
    (root / name).parent.mkdir(parents=True)
    (root / name).write_bytes(data[name])
    (root / "chb01/chb01-summary.txt").write_bytes(data["chb01/chb01-summary.txt"])
    # An incompatible extra file does not silently join an explicitly selected experiment.
    (root / "chb01/chb01_99.edf").write_bytes(b"unselected")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    frame, records = discover(root, PreprocessConfig(sfreq=128), ["C1", "C2"], 4, path)
    assert len(frame) == 3 and len(records) == 1
    manifest["recordings"][0]["path"] = "chb01/chb01_02.edf"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        discover(root, PreprocessConfig(sfreq=128), ["C1", "C2"], 4, path)


def test_invalid_manifest_paths(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"recordings": [{"path": "../file.edf", "size_bytes": 5, "sha256": "a"*64}]}))
    with pytest.raises(ValueError):
        load_manifest(path)
