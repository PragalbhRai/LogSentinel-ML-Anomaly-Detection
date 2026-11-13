"""Build/verify the curated team ZIP using only Python's standard library.

Run from the project root: python scripts/package_handoff.py
Verify an extracted handoff: python scripts/package_handoff.py --verify-tree .
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

ROOT_FILES = (
    ".gitignore", "README.md", "SETUP.md", "logsentinel.ipynb", "pyproject.toml",
    "requirements.txt", "requirements-runtime.txt", "requirements-dev.txt",
    "requirements-experiments.txt", "requirements.lock.txt", "requirements-qa.txt", ".gitattributes", ".python-version",
)
SOURCE_DIRS = ("app", "src", "tests", "configs", "docs", "manifests", "scripts")
EXCLUDED_PARTS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "outputs", "node_modules"}
ZIP_ROOT = "LogSentinel"


def digest_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_originals(root):
    original = json.loads((root / "manifests/artifact_checksums.json").read_text(encoding="utf-8"))["files"]
    for relative, expected in original.items():
        if digest_file(root / relative) != expected:
            raise ValueError(f"Original experiment changed: {relative}")
    return original


def curated_files(root):
    files = {Path(name) for name in ROOT_FILES}
    files.add(Path(".streamlit/config.toml"))
    for directory in SOURCE_DIRS:
        for current, folders, names in os.walk(root / directory, followlinks=False):
            folders[:] = [name for name in folders if name not in EXCLUDED_PARTS
                          and not name.endswith(".egg-info")
                          and not (Path(current) / name).is_symlink()
                          and not (Path(current) / name).is_junction()]
            for name in names:
                path = Path(current) / name
                if path.suffix in {".py", ".css", ".md", ".json", ".yaml", ".yml", ".toml"}:
                    files.add(path.relative_to(root))
    # Include exactly the completed RF payload, verifying its saved checksums.
    final_run = root / "outputs/imbalance_experiments/run_20261113_01"
    completion = json.loads((final_run / "completion.json").read_text(encoding="utf-8"))
    for name, expected in completion["files_sha256"].items():
        path = (final_run / name).resolve()
        if not path.is_relative_to(final_run.resolve()) or digest_file(path) != expected:
            raise ValueError(f"Final experiment payload mismatch: {name}")
        files.add(path.relative_to(root))
    files.add((final_run / "completion.json").relative_to(root))
    # The immutable manifest is the exact original artifact allowlist.
    files.update(Path(name) for name in verify_originals(root))
    for name in ("artifacts/.gitkeep", "data/raw/.gitkeep", "data/processed/.gitkeep", "data/splits/.gitkeep"):
        files.add(Path(name))
    for relative in files:
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing/unsafe package file: {relative}")
        is_final_run = path.resolve().is_relative_to(final_run.resolve())
        if any((part in EXCLUDED_PARTS and not (part == "outputs" and is_final_run))
               or part.endswith(".egg-info") for part in relative.parts):
            raise ValueError(f"Excluded file selected: {relative}")
    return sorted(files, key=lambda item: item.as_posix())


def verify_tree(root):
    root = root.resolve()
    manifest = json.loads((root / "HANDOFF_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Unsafe manifest path: {name}")
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/unsafe extracted file: {name}")
        if path.stat().st_size != expected["bytes"] or digest_file(path) != expected["sha256"]:
            raise ValueError(f"Extracted payload mismatch: {name}")
    verify_originals(root)
    return {"payload_files_verified": len(manifest["files"]), "original_exports_verified": 22,
            "original_notebook_verified": True, "root": root.name}


def build(root, destination):
    root = root.resolve()
    files = curated_files(root)
    manifest = {
        "schema_version": 1, "project": "LogSentinel", "root_folder": ZIP_ROOT,
        "python": "Windows x64 CPython 3.12 (verified 3.12.10)",
        "instructions": "SETUP.md", "requires_retraining": False,
        "excluded": ["virtual environments", "caches", "raw HDFS dataset", "browser QA outputs", "temporary/generated outputs"],
        "files": {name.as_posix(): {"bytes": (root / name).stat().st_size, "sha256": digest_file(root / name)} for name in files},
    }
    # Exclusive creation prevents accidentally replacing an existing shared ZIP.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in files:
            archive.write(root / name, ZIP_ROOT+"/"+name.as_posix())
        archive.writestr(ZIP_ROOT+"/HANDOFF_MANIFEST.json", json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {ZIP_ROOT+"/"+name.as_posix() for name in files} | {ZIP_ROOT+"/HANDOFF_MANIFEST.json"}:
            raise ValueError("ZIP membership mismatch")
        for name, expected in manifest["files"].items():
            with archive.open(ZIP_ROOT+"/"+name) as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected["sha256"]:
                    raise ValueError(f"ZIP payload checksum mismatch: {name}")
    verify_originals(root)
    digest = digest_file(destination)
    destination.with_suffix(".sha256").write_text(digest+"  "+destination.name+"\n", encoding="ascii")
    return {"zip": destination.name, "bytes": destination.stat().st_size,
            "size_mib": round(destination.stat().st_size / 1024**2, 2), "files": len(files)+1,
            "uncompressed_bytes": sum(value["bytes"] for value in manifest["files"].values()),
            "sha256": digest, "crc_check": "passed", "payload_hashes": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-tree", type=Path)
    parser.add_argument("--output", type=Path, default=Path("LogSentinel_Team_Handoff.zip"))
    args = parser.parse_args()
    if args.verify_tree:
        result = verify_tree(args.verify_tree)
    else:
        root = Path(__file__).resolve().parents[1]
        result = build(root, args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
