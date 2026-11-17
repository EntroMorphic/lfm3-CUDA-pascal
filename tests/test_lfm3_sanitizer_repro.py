"""Host-only packet integrity/security regressions: no CUDA or subprocesses."""
import json
from pathlib import Path
import shutil

import pytest

from tools import check_lfm3_sanitizer_repro as repro


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def packet(tmp_path):
    destination = tmp_path / "packet"
    repro.build(REPO, destination)
    return destination


def test_exact_allowlist_and_failed_status(packet):
    manifest = repro.verify(packet)
    assert set(manifest["files"]) == set(repro.baseline_names()) | set(repro.EXTRAS)
    assert len(manifest["files"]) == 83
    assert manifest["status"] == "failed-controls-or-tool"
    assert manifest["integrated_initcheck_accepted"] is False
    summary = json.loads((packet / repro.EVIDENCE / "summary.json").read_text())
    assert summary["status"] == "failed-controls-or-tool"
    assert sum(r["classification"] == "unexpected-failure" for r in summary["records"]) == 3
    assert not any(p.suffix in (".safetensors", ".pt", ".bin") for p in packet.rglob("*"))


def test_altered_manifest_rejected(packet):
    path = packet / "packet-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "passed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        repro.verify(packet)


def test_unrecognized_manifest_fields_rejected(packet):
    path = packet / "packet-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["vendor_confirmed"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        repro.verify(packet)


@pytest.mark.parametrize("change", ["missing", "altered", "rehash"])
def test_negative_control_evidence_fail_closed(packet, change):
    name = repro.EVIDENCE + "tool-2/1x128-missing.sanitizer.log"
    path = packet / name
    if change == "missing":
        path.unlink()
    else:
        path.write_text("fabricated clean control\n")
        if change == "rehash":
            manifest_path = packet / "packet-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"][name] = repro.digest(path.read_bytes())
            manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        repro.verify(packet)


def test_no_arbitrary_repo_attachments_copied(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for name in repro.baseline_names() + list(repro.EXTRAS[1:]):
        name = repro.REPO_PATH.get(name, name)  # the repository's layout, not the packet's
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
    for name in ("model.safetensors", ".env", "credentials.json", "http-token.txt"):
        (root / name).write_text("private data must remain local")
    destination = tmp_path / "packet"
    repro.build(root, destination)
    assert not any((destination / name).exists() for name in ("model.safetensors", ".env", "credentials.json", "http-token.txt"))


@pytest.mark.parametrize("name", ["model.safetensors", "credentials.json", "unexpected/empty"])
def test_extra_packet_entries_rejected(packet, name):
    path = packet / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not allowed")
    with pytest.raises(ValueError):
        repro.verify(packet)


def test_manifest_traversal_rejected(packet):
    path = packet / "packet-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../../credentials"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        repro.verify(packet)


def test_symlink_and_existing_output_rejected(packet, tmp_path):
    with pytest.raises(ValueError):
        repro.build(REPO, packet)
    name = repro.EVIDENCE + "tool-0/1x128-kernel.application.log"
    path = packet / name
    outside = tmp_path / "outside"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        repro.verify(packet)


@pytest.mark.parametrize("content", ["Authorization: Bearer fake-secret", "-----BEGIN RSA PRIVATE KEY-----", "https://host/path?token=secret"])
def test_secret_content_rejected(tmp_path, content):
    (tmp_path / "evidence.log").write_text(content)
    with pytest.raises(ValueError):
        repro.read_local(tmp_path, "evidence.log")


def test_untrusted_read_paths_rejected(tmp_path):
    for name in ("../secret", "/etc/passwd", "a/../../secret", "a\\secret"):
        with pytest.raises(ValueError):
            repro.read_local(tmp_path, name)
