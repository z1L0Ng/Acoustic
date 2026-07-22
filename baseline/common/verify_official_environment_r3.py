"""Strict Release 3 environment gate for the five official reproductions."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings


BASE_RELEASE_COMMIT = "3f757adcc12fcc5b5e2f1058a593345f750de2a5"
SETUPTOOLS_VERSION = "80.9.0"
COMMON_IMPORTS = [
    "numpy", "torch", "torchaudio", "torchvision", "timm", "librosa",
    "scipy", "sklearn", "pkg_resources", "triton",
]
METHODS = {
    "patch_mix_cl": {
        "environment": "acoustic-patchmix-r3",
        "yaml": "baseline/patch_mix_cl/environment.linux-cu118.yml",
        "cuda": "11.8",
        "versions": {
            "torch": "2.0.1", "torchaudio": "2.0.2", "torchvision": "0.15.2",
            "triton": "2.0.0", "cmake": "3.26.4", "lit": "16.0.6",
            "setuptools": SETUPTOOLS_VERSION, "librosa": "0.9.2",
        },
        "imports": ["transformers", "safetensors", "cmake", "lit"],
    },
    "pafa": {
        "environment": "acoustic-pafa-r3",
        "yaml": "baseline/pafa/environment.linux-cu118.yml",
        "cuda": "11.8",
        "versions": {
            "torch": "2.0.1", "torchaudio": "2.0.2", "torchvision": "0.15.2",
            "triton": "2.0.0", "cmake": "3.26.4", "lit": "16.0.6",
            "setuptools": SETUPTOOLS_VERSION, "librosa": "0.9.2",
        },
        "imports": ["transformers", "einops", "cmake", "lit"],
    },
    "sg_scl": {
        "environment": "acoustic-sgscl-r3",
        "yaml": "baseline/sg_scl/environment.linux-cu118.yml",
        "cuda": "11.8",
        "versions": {
            "torch": "2.0.1", "torchaudio": "2.0.2", "torchvision": "0.15.2",
            "triton": "2.0.0", "cmake": "3.26.4", "lit": "16.0.6",
            "setuptools": SETUPTOOLS_VERSION, "librosa": "0.9.2",
        },
        "imports": ["cmake", "lit"],
    },
    "mvst": {
        "environment": "acoustic-mvst-r3",
        "yaml": "baseline/mvst/environment.linux-cu118.yml",
        "cuda": "11.8",
        "versions": {
            "torch": "2.0.1", "torchaudio": "2.0.2", "torchvision": "0.15.2",
            "triton": "2.0.0", "cmake": "3.26.4", "lit": "16.0.6",
            "setuptools": SETUPTOOLS_VERSION, "librosa": "0.9.2",
            "opencv-python-headless": "4.11.0.86",
            "opencv-python": "4.11.0.86", "cmapy": "0.6.6",
        },
        "imports": ["cv2", "cmapy", "cmake", "lit"],
    },
    "add_rsc": {
        "environment": "acoustic-addrsc-r3",
        "yaml": "baseline/add_rsc/environment.linux-cu121.yml",
        "cuda": "12.1",
        "versions": {
            "torch": "2.3.1", "torchaudio": "2.3.1", "torchvision": "0.18.1",
            "triton": "2.3.1", "setuptools": SETUPTOOLS_VERSION,
            "librosa": "0.9.2",
        },
        "imports": [],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def verify(method: str, project_root: Path, output: Path, cuda_mode: str) -> dict:
    spec = METHODS[method]
    project_root = project_root.resolve()
    environment_path = project_root / spec["yaml"]
    errors: list[str] = []

    yaml_text = environment_path.read_text()
    required_yaml_tokens = [
        f"name: {spec['environment']}",
        "- setuptools==80.9.0",
    ]
    if method != "add_rsc":
        required_yaml_tokens.extend(["- cmake==3.26.4", "- lit==16.0.6"])
    missing_tokens = [token for token in required_yaml_tokens if token not in yaml_text]
    if missing_tokens:
        errors.append(f"environment declaration missing: {missing_tokens}")

    active_environment = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    if active_environment != spec["environment"]:
        errors.append(
            f"active environment {active_environment!r} != {spec['environment']!r}"
        )
    if platform.system() != "Linux":
        errors.append(f"target gate requires Linux, found {platform.system()}")
    if sys.version_info[:2] != (3, 10):
        errors.append(f"Python must be 3.10, found {platform.python_version()}")

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if pip_check.returncode != 0:
        errors.append(f"pip check failed: {pip_check.stdout.strip()}")

    versions = {name: installed_version(name) for name in spec["versions"]}
    for name, expected in spec["versions"].items():
        if versions[name] != expected:
            errors.append(f"{name} version {versions[name]!r} != {expected!r}")

    import_results = {}
    for module_name in [*COMMON_IMPORTS, *spec["imports"]]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                module = import_module(module_name)
            import_results[module_name] = {
                "status": "ok",
                "version": getattr(module, "__version__", None),
                "warnings": [str(item.message) for item in caught],
            }
        except Exception as error:  # The receipt must preserve the exact import failure.
            import_results[module_name] = {
                "status": "failed", "error": f"{type(error).__name__}: {error}"
            }
            errors.append(f"import {module_name} failed: {type(error).__name__}: {error}")

    torch_module = sys.modules.get("torch")
    torch_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if torch_cuda_version != spec["cuda"]:
        errors.append(
            f"torch CUDA metadata {torch_cuda_version!r} != {spec['cuda']!r}"
        )
    cuda_receipt = {"mode": cuda_mode, "torch_cuda_version": torch_cuda_version}
    if cuda_mode == "runtime" and torch_module is not None:
        cuda_receipt["available"] = bool(torch_module.cuda.is_available())
        cuda_receipt["device_count"] = int(torch_module.cuda.device_count())
        if not cuda_receipt["available"] or cuda_receipt["device_count"] < 1:
            errors.append("CUDA runtime is not available")
        else:
            device = torch_module.device("cuda:0")
            left = torch_module.ones((8, 8), device=device)
            value = (left @ left).sum()
            torch_module.cuda.synchronize()
            cuda_receipt.update({
                "device_name": torch_module.cuda.get_device_name(0),
                "device_capability": list(torch_module.cuda.get_device_capability(0)),
                "finite_kernel": bool(torch_module.isfinite(value).item()),
            })
            if not cuda_receipt["finite_kernel"]:
                errors.append("CUDA finite-kernel check failed")

    triton_requirements = metadata.requires("triton") or []
    installed_packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    }
    receipt = {
        "status": "verified" if not errors else "failed",
        "release": "official-reproduction-release-3",
        "base_release_commit": BASE_RELEASE_COMMIT,
        "method": method,
        "environment": spec["environment"],
        "environment_spec": {
            "path": spec["yaml"], "sha256": sha256(environment_path),
        },
        "platform": {
            "system": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(), "prefix": sys.prefix,
        },
        "pip_check": {
            "returncode": pip_check.returncode,
            "output": pip_check.stdout.strip(),
        },
        "expected_versions": spec["versions"],
        "observed_versions": versions,
        "imports": import_results,
        "triton_declared_requirements": triton_requirements,
        "cuda": cuda_receipt,
        "installed_packages": dict(sorted(installed_packages.items())),
        "errors": errors,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if errors:
        raise ValueError(f"Release 3 environment verification failed: {errors}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cuda-mode", choices=["metadata", "runtime"], required=True,
        help="Use runtime on the allocated L40; metadata is for Linux solve auditing only.",
    )
    args = parser.parse_args()
    receipt = verify(args.method, args.project_root, args.output, args.cuda_mode)
    print(json.dumps({
        "status": receipt["status"], "method": args.method,
        "environment": receipt["environment"], "cuda_mode": args.cuda_mode,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
