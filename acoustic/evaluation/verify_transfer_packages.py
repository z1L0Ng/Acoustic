"""Static verifier for the B1 PAFA and B2 SG-SCL immutable packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


COMMON_FILES = [
    "acoustic/evaluation/__init__.py",
    "acoustic/evaluation/sprsound_inter.py",
    "acoustic/evaluation/verify_sprsound_inter_contract.py",
    "acoustic/evaluation/verify_transfer_packages.py",
]
METHOD_FILES = {
    "pafa": [
        "baseline/pafa/checkpoint_eval/__init__.py",
        "baseline/pafa/checkpoint_eval/bootstrap.py",
        "baseline/pafa/checkpoint_eval/run_sprsound_transfer.py",
        "baseline/pafa/checkpoint_eval/verify_sprsound_transfer.py",
        "baseline/pafa/checkpoint_eval/protocol.json",
        "baseline/pafa/checkpoint_eval/PACKAGE_STATUS.md",
        "baseline/pafa/checkpoint_eval/README.md",
        "baseline/pafa/checkpoint_eval/pafa_sprsound_transfer.ipynb",
    ],
    "sg_scl": [
        "baseline/sg_scl/checkpoint_eval/__init__.py",
        "baseline/sg_scl/checkpoint_eval/bootstrap.py",
        "baseline/sg_scl/checkpoint_eval/run_sprsound_transfer.py",
        "baseline/sg_scl/checkpoint_eval/verify_sprsound_transfer.py",
        "baseline/sg_scl/checkpoint_eval/protocol.json",
        "baseline/sg_scl/checkpoint_eval/PACKAGE_STATUS.md",
        "baseline/sg_scl/checkpoint_eval/README.md",
        "baseline/sg_scl/checkpoint_eval/sg_scl_sprsound_transfer.ipynb",
    ],
}
EXPECTED_ROOT_PREFIX = {
    "pafa": "pafa_sprsound_transfer_",
    "sg_scl": "sg_scl_sprsound_transfer_",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text())
    if notebook.get("nbformat") != 4:
        raise RuntimeError(f"invalid notebook format: {path}")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            if cell.get("execution_count") is not None or cell.get("outputs"):
                raise RuntimeError(f"notebook must be clean: {path}")


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    verified: dict[str, int] = {}
    for method, method_files in METHOD_FILES.items():
        files = COMMON_FILES + method_files
        missing = [relative for relative in files if not (root / relative).is_file()]
        if missing:
            raise FileNotFoundError(f"{method} package files missing: {missing}")
        notebook = root / next(relative for relative in method_files if relative.endswith(".ipynb"))
        verify_notebook(notebook)
        protocol = json.loads((root / f"baseline/{method}/checkpoint_eval/protocol.json").read_text())
        if protocol["target"]["events"] != 1429:
            raise RuntimeError(f"{method} target count drift")
        if protocol["target"]["event_id_sha256"] != "81a6b15783a01eb86abe218928884b41e7f975f64eedaefd546e2dbf3deba44b":
            raise RuntimeError(f"{method} target ID contract drift")
        combined_text = "\n".join((root / relative).read_text(errors="ignore") for relative in method_files)
        forbidden = ["/Users/", "/files1/", "results/"]
        found = [token for token in forbidden if token in combined_text]
        if found:
            raise RuntimeError(f"{method} nonportable path token(s): {found}")
        if EXPECTED_ROOT_PREFIX[method] not in combined_text:
            raise RuntimeError(f"{method} canonical result-root prefix absent")
        manifest_path = root / f"baseline/{method}/checkpoint_eval/package_manifest.sha256"
        manifest = read_manifest(manifest_path)
        if set(manifest) != set(files):
            raise RuntimeError(f"{method} package manifest membership mismatch")
        mismatches = {
            relative: (manifest[relative], sha256_file(root / relative))
            for relative in files
            if manifest[relative] != sha256_file(root / relative)
        }
        if mismatches:
            raise RuntimeError(f"{method} package hash mismatch: {mismatches}")
        verified[method] = len(files)
    print(f"checkpoint_transfer_packages_static_ok files={verified}")


if __name__ == "__main__":
    main()
