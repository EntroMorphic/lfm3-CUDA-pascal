import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pedantic_diagnostic", ROOT / "tools/prepare_lfm3_pedantic_diagnostic.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    shutil.copytree(ROOT / "native/include", root / "native/include")
    for relative in [builder.MAIN, builder.BATCH, builder.PROBE,
                     "native/src/lfm3_worker.cu", "native/src/safetensors.cpp",
                     "native/CMakeLists.txt", "LICENSE", "native/third_party/PYTORCH_LICENSE"]:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return root


def test_mirror_pins_hashes_without_touching_production(tmp_path):
    paths = [p for p in (ROOT / "native").rglob("*") if p.is_file() and not p.is_symlink()]
    before = {p: sha(p) for p in paths}
    output = tmp_path / "mirror"
    manifest = builder.prepare(ROOT, output)
    assert before == {p: sha(p) for p in paths}
    assert manifest == json.loads((output / "manifest.json").read_text())
    assert manifest["diagnostic_only"] is True
    assert manifest["full_acceptance"] is False
    assert manifest["expected_eager_checks"] == 6890
    assert manifest["goldens_may_change"] is False
    for relative, record in manifest["files"].items():
        assert record["original_sha256"] == record["copied_sha256"] == sha(ROOT / relative)
        assert record["modified_sha256"] == sha(output / relative)
        assert record["source_origin"] == str(ROOT / relative)
        if relative not in builder.REPLACEMENTS:
            assert (ROOT / relative).read_bytes() == (output / relative).read_bytes()
    main = (output / builder.MAIN).read_text()
    assert "lfm::finish_handle_setup(blas," in main
    assert "noexcept {cublasDestroy(handle);}" in main
    for relative, replacements in builder.REPLACEMENTS.items():
        original = (ROOT / relative).read_text()
        for before, (after, count) in replacements.items():
            original = original.replace(before, after)
            assert manifest["files"][relative]["replacements"][before]["count"] == count
        assert (output / relative).read_text() == original


def test_new_directory_exclusive(source, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("untouched")
    with pytest.raises(FileExistsError):
        builder.prepare(source, output)
    assert sentinel.read_text() == "untouched"


@pytest.mark.parametrize("relative", ["native/include/lfm/escape.hpp",
                                     "native/include/escape-dir", builder.MAIN])
def test_reject_symlinks(source, tmp_path, relative):
    outside = tmp_path / "outside"
    outside.write_text("private")
    link = source / relative
    if link.exists():
        link.unlink()
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        builder.prepare(source, tmp_path / "mirror")
    assert not (tmp_path / "mirror").exists()


def test_only_project_headers_and_explicit_sources_copied(source, tmp_path):
    for relative in ["model.safetensors", ".env", "native/src/secret.cu",
                     "native/include/data.bin", "native/include/config.json"]:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("MUST NOT COPY")
    output = tmp_path / "mirror"
    builder.prepare(source, output, with_probe=False)
    assert not (output / builder.PROBE).exists()
    assert "lfm3-decode-batch-probe" not in (output / "CMakeLists.txt").read_text()
    assert all(b"MUST NOT COPY" not in p.read_bytes() for p in output.rglob("*") if p.is_file())


@pytest.mark.parametrize("relative,token", [(builder.MAIN, "CUBLAS_COMPUTE_32F"),
                                           (builder.BATCH, "CUBLAS_GEMM_DEFAULT_TENSOR_OP")])
def test_count_drift_rejected_before_output_creation(source, tmp_path, relative, token):
    path = source / relative
    path.write_text(path.read_text() + f"\n// {token}\n")
    with pytest.raises(ValueError, match="expected"):
        builder.prepare(source, tmp_path / "mirror")
    assert not (tmp_path / "mirror").exists()


def test_exact_word_replacement_and_constexpr_build_metadata(source, tmp_path):
    path = source / builder.MAIN
    path.write_text(path.read_text() + "\n// CUBLAS_COMPUTE_32F_FAST_16BF\n")
    output = tmp_path / "mirror"
    manifest = builder.prepare(source, output)
    assert "CUBLAS_COMPUTE_32F_FAST_16BF" in (output / builder.MAIN).read_text()
    cmake = (output / "CMakeLists.txt").read_text()
    assert "cxx_std_20 cuda_std_20" in cmake
    assert "CUDA_ARCHITECTURES 110" in cmake
    assert "-O3 --fmad=false" in cmake
    assert "relaxed-constexpr" not in cmake
    assert "native/include" in cmake and "native/src/safetensors.cpp" in cmake
    assert str(source) not in cmake and str(output) not in cmake
    assert manifest["build_policy"]["relaxed_constexpr"] is False
    assert manifest["generated_files"]["CMakeLists.txt"]["sha256"] == sha(output / "CMakeLists.txt")
    assert (output / "native/include/lfm/handle_setup.hpp").read_bytes() == (
        source / "native/include/lfm/handle_setup.hpp").read_bytes()
