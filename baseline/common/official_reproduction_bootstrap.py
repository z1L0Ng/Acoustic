"""Fresh-checkout bootstrap for official-reproduction server packages."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
from datetime import datetime
import re
from zoneinfo import ZoneInfo

from .install_official_compatibility import install_compatibility_patch


LABELS = ["normal", "crackle", "wheeze", "both"]
RELEASE_NAME = "official-reproduction-release-4"
BASE_RELEASE_COMMIT = "4172524a0e5d7b792de248820439f30874e2ae6d"
ENVIRONMENT_NAMES = {
    "patch_mix_cl": "acoustic-patchmix",
    "pafa": "acoustic-pafa",
}


@dataclass(frozen=True)
class MethodSpec:
    method: str
    repo_url: str
    commit: str
    checkpoint_filename: str
    checkpoint_url: str
    checkpoint_sha256: str
    checkpoint_provenance: str
    support_subdir: str


SPECS = {
    "patch_mix_cl": MethodSpec(
        method="patch_mix_cl",
        repo_url="https://github.com/raymin0223/patch-mix_contrastive_learning.git",
        commit="836b09fea1b70eb29fe0b25afa481286b56f5104",
        checkpoint_filename="audioset_10_10_0.4593.pth",
        checkpoint_url="https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1",
        checkpoint_sha256="dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f",
        checkpoint_provenance="current author-hosted runtime artifact; historical 2023 serialized-byte identity unresolved",
        support_subdir="data/icbhi_dataset",
    ),
    "pafa": MethodSpec(
        method="pafa",
        repo_url="https://github.com/wa976/PAFA.git",
        commit="e49e294d0db0d6af10ac46290512b9c85d3f71e1",
        checkpoint_filename="BEATs_iter3_plus_AS2M.pt",
        checkpoint_url="https://huggingface.co/mooneyko/BEATs/resolve/18cfdf9d43820b2db86c6dfde4ae2f7531d1f5ad/BEATs_iter3_plus_AS2M.pt",
        checkpoint_sha256="d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34",
        checkpoint_provenance="public mooneyko/BEATs compatibility mirror; Microsoft byte identity unverified",
        support_subdir="data",
    ),
}

RUN_ROOT_PATTERN = re.compile(r"^(patch_mix_cl|pafa)_\d{8}_\d{6}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_result_root_for_bootstrap(result_root: Path, method: str) -> Path:
    """Allow only the strict Release 4 environment receipt before bootstrap."""
    environment_receipt = result_root / "receipts" / "environment_r4.json"
    if result_root.exists():
        unexpected = [
            path for path in result_root.rglob("*")
            if path.is_file() and path != environment_receipt
        ]
        if unexpected:
            raise FileExistsError(
                f"result root contains files other than the Release 4 environment receipt: {unexpected}"
            )
    result_root.mkdir(parents=True, exist_ok=True)
    if not environment_receipt.is_file():
        raise FileNotFoundError(
            "run baseline.common.verify_official_environment_r4 before bootstrap: "
            f"{environment_receipt}"
        )
    receipt = json.loads(environment_receipt.read_text())
    if (
        receipt.get("status") != "verified"
        or receipt.get("release") != RELEASE_NAME
        or receipt.get("method") != method
        or receipt.get("base_release_commit") != BASE_RELEASE_COMMIT
    ):
        raise ValueError(f"invalid Release 4 environment receipt: {environment_receipt}")
    return environment_receipt


def run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command, cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def chicago_timestamp() -> str:
    return datetime.now(ZoneInfo("America/Chicago")).isoformat()


def validate_output_roots(
    project_root: Path, result_root: Path, cache_root: Path | None, method: str,
) -> tuple[Path, Path]:
    canonical_result = (project_root / "result").resolve()
    result_root = result_root.resolve()
    if result_root.parent != canonical_result:
        raise ValueError(f"result root must be directly under {canonical_result}: {result_root}")
    if not RUN_ROOT_PATTERN.fullmatch(result_root.name) or not result_root.name.startswith(f"{method}_"):
        raise ValueError(
            "result root must use result/<method>_YYYYMMDD_HHMMSS naming: "
            f"{result_root}"
        )
    cache_root = (cache_root or result_root / "cache").resolve()
    try:
        cache_root.relative_to(result_root)
    except ValueError as error:
        raise ValueError(f"cache root must be inside result root: {cache_root}") from error
    return result_root, cache_root


def discover_audio_dir(dataset_root: Path) -> Path:
    candidates: dict[Path, int] = {}
    for wav in dataset_root.rglob("*.wav"):
        candidates[wav.parent] = candidates.get(wav.parent, 0) + 1
    matches = []
    for parent, wav_count in candidates.items():
        annotations = [
            path for path in parent.glob("*.txt")
            if len(path.stem.split("_")) >= 5
        ]
        if wav_count == 920 and len(annotations) == 920:
            matches.append(parent)
    if len(matches) != 1:
        raise ValueError(f"expected one 920-WAV ICBHI directory, found {matches}")
    return matches[0].resolve()


def discover_split_file(dataset_root: Path) -> Path:
    candidates = list(dataset_root.rglob("ICBHI_challenge_train_test.txt"))
    valid = []
    for path in candidates:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) == 920:
            valid.append(path)
    if len(valid) != 1:
        raise ValueError(f"expected one 920-row official split, found {valid}")
    return valid[0].resolve()


def stable_recording_key(recording_id: str) -> str:
    return "_".join(recording_id.split("_")[:4])


def parse_split(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        recording_id, split = line.split()
        if split not in {"train", "test"} or recording_id in result:
            raise ValueError(f"invalid split row: {line}")
        result[recording_id] = split
    if len(result) != 920:
        raise ValueError(f"official split has {len(result)} recordings")
    return result


def label_name(crackle: int, wheeze: int) -> str:
    return LABELS[crackle + 2 * wheeze if not (crackle and wheeze) else 3]


def build_manifest(dataset_root: Path, output: Path) -> dict:
    audio_dir = discover_audio_dir(dataset_root)
    split_file = discover_split_file(dataset_root)
    split = parse_split(split_file)
    stable_split: dict[str, list[tuple[str, str]]] = {}
    for recording_id, split_name in split.items():
        stable_split.setdefault(stable_recording_key(recording_id), []).append((recording_id, split_name))

    rows = []
    aliases = []
    for wav in sorted(audio_dir.glob("*.wav"), key=lambda path: path.stem):
        recording_id = wav.stem
        if recording_id in split:
            split_id, split_name = recording_id, split[recording_id]
        else:
            candidates = stable_split.get(stable_recording_key(recording_id), [])
            if len(candidates) != 1:
                raise ValueError(f"cannot resolve split alias for {recording_id}: {candidates}")
            split_id, split_name = candidates[0]
            aliases.append({"audio_recording_id": recording_id, "split_recording_id": split_id})
        annotation = wav.with_suffix(".txt")
        if not annotation.exists():
            raise FileNotFoundError(annotation)
        with annotation.open(newline="") as handle:
            for index, fields in enumerate(csv.reader(handle, delimiter="\t")):
                if len(fields) != 4:
                    raise ValueError(f"invalid annotation row: {annotation}:{index + 1}")
                start, end, crackle, wheeze = fields
                crackle_i, wheeze_i = int(crackle), int(wheeze)
                label = label_name(crackle_i, wheeze_i)
                rows.append({
                    "cycle_id": f"{recording_id}__cycle_{index:03d}",
                    "recording_id": recording_id,
                    "split_recording_id": split_id,
                    "patient_id": recording_id.split("_")[0],
                    "group_id": recording_id.split("_")[0],
                    "audio_path": str(wav.resolve()),
                    "cycle_start_s": start,
                    "cycle_end_s": end,
                    "official_split": split_name,
                    "native_four_class_label": label,
                    "binary_label": "normal" if label == "normal" else "abnormal",
                })

    fields = list(rows[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    split_counts = {
        name: sum(row["official_split"] == name for row in rows)
        for name in ["train", "test"]
    }
    label_counts = {
        label: sum(row["native_four_class_label"] == label for row in rows)
        for label in LABELS
    }
    expected_split = {"train": 4142, "test": 2756}
    expected_labels = {"normal": 3642, "crackle": 1864, "wheeze": 886, "both": 506}
    if len(rows) != 6898 or split_counts != expected_split or label_counts != expected_labels:
        raise ValueError(
            f"manifest contract mismatch: rows={len(rows)} split={split_counts} labels={label_counts}"
        )
    return {
        "dataset_root": str(dataset_root),
        "audio_dir": str(audio_dir),
        "official_split_file": str(split_file),
        "official_split_sha256": sha256(split_file),
        "manifest_path": str(output.resolve()),
        "manifest_sha256": sha256(output),
        "cycles": len(rows),
        "recordings": 920,
        "unique_cycle_ids": len({row["cycle_id"] for row in rows}),
        "split_cycle_counts": split_counts,
        "label_counts": label_counts,
        "split_alias_used": bool(aliases),
        "split_aliases": aliases,
    }


def clone_source(spec: MethodSpec, destination: Path) -> dict:
    if destination.exists():
        if not (destination / ".git").exists():
            raise FileExistsError(f"non-git source destination exists: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", spec.repo_url, str(destination)])
    run(["git", "checkout", "--detach", spec.commit], cwd=destination)
    head = run(["git", "rev-parse", "HEAD"], cwd=destination)
    remote = run(["git", "remote", "get-url", "origin"], cwd=destination)
    status = run(["git", "status", "--short"], cwd=destination)
    if head != spec.commit or status:
        raise ValueError(f"source pin/cleanliness failure: head={head} status={status!r}")
    return {"repo_url": spec.repo_url, "remote": remote, "commit": head, "clean": True}


def materialize_checkpoint(
    spec: MethodSpec,
    destination: Path,
    checkpoint_path: Path | None,
    checkpoint_url: str | None,
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_kind: str
    source_value: str
    if checkpoint_path is not None:
        source = checkpoint_path.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            destination.unlink()
        try:
            os.link(source, destination)
            source_kind = "explicit_path_hardlink"
        except OSError:
            shutil.copy2(source, destination)
            source_kind = "explicit_path_copy"
        source_value = str(source)
    else:
        url = checkpoint_url or spec.checkpoint_url
        request = urllib.request.Request(url, headers={"User-Agent": "Acoustic-reproduction-bootstrap/1"})
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
        source_kind = "public_url_download"
        source_value = url
    observed = sha256(destination)
    if observed != spec.checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA mismatch for {spec.method}: expected={spec.checkpoint_sha256} observed={observed}"
        )
    return {
        "path": str(destination.resolve()),
        "size_bytes": destination.stat().st_size,
        "sha256": observed,
        "source_kind": source_kind,
        "source": source_value,
        "public_default_url": spec.checkpoint_url,
        "provenance": spec.checkpoint_provenance,
    }


def ensure_symlink(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() and path.resolve() == target.resolve():
            return
        raise FileExistsError(f"refusing to replace non-matching adapter path: {path}")
    path.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def build_adapter(spec: MethodSpec, source: Path, audio_dir: Path, checkpoint: Path, root: Path) -> dict:
    target = root / "data" / "icbhi_dataset"
    target.mkdir(parents=True, exist_ok=True)
    support = source / spec.support_subdir
    support_files = ["official_split.txt", "metadata.txt", "patient_diagnosis.txt", "patient_list_foldwise.txt"]
    for name in support_files:
        ensure_symlink(target / name, support / name)
    lines = (support / "official_split.txt").read_text().splitlines()
    author_split = {line.split("\t")[0]: line.split("\t")[1] for line in lines}
    stable_author: dict[str, list[str]] = {}
    for recording_id in author_split:
        stable_author.setdefault(stable_recording_key(recording_id), []).append(recording_id)
    audio_adapter = target / "audio_test_data"
    audio_adapter.mkdir()
    aliases = []
    for wav in sorted(audio_dir.glob("*.wav")):
        raw_id = wav.stem
        if raw_id in author_split:
            adapter_id = raw_id
        else:
            candidates = stable_author.get(stable_recording_key(raw_id), [])
            if len(candidates) != 1:
                raise ValueError(f"cannot map raw recording into author split: {raw_id}: {candidates}")
            adapter_id = candidates[0]
            aliases.append({"raw_recording_id": raw_id, "adapter_recording_id": adapter_id})
        ensure_symlink(audio_adapter / f"{adapter_id}.wav", wav)
        ensure_symlink(audio_adapter / f"{adapter_id}.txt", wav.with_suffix(".txt"))
    if len(list(audio_adapter.glob("*.wav"))) != 920:
        raise ValueError("author data adapter must contain exactly 920 WAV symlinks")
    pretrained = root / "pretrained_models"
    pretrained.mkdir(parents=True, exist_ok=True)
    ensure_symlink(pretrained / spec.checkpoint_filename, checkpoint)
    counts = {
        "train": sum(line.endswith("\ttrain") for line in lines),
        "test": sum(line.endswith("\ttest") for line in lines),
    }
    if counts != {"train": 539, "test": 381}:
        raise ValueError(f"author split counts mismatch: {counts}")
    cycle_counts = {"train": 0, "test": 0}
    for recording_id, split_name in author_split.items():
        annotation = audio_adapter / f"{recording_id}.txt"
        cycle_counts[split_name] += sum(
            bool(line.strip()) for line in annotation.read_text().splitlines()
        )
    if cycle_counts != {"train": 4142, "test": 2756}:
        raise ValueError(f"author adapter cycle counts mismatch: {cycle_counts}")
    return {
        "portable_run": str(root.resolve()),
        "raw_audio_target": str(audio_dir),
        "raw_policy": "920 read-only file symlinks; stable-key alias only where author split name differs",
        "audio_adapter_wavs": 920,
        "split_alias_used": bool(aliases),
        "split_aliases": aliases,
        "checkpoint_target": str(checkpoint.resolve()),
        "author_split_recording_counts": counts,
        "author_split_cycle_counts": cycle_counts,
        "support_files": {name: str((support / name).resolve()) for name in support_files},
    }


def bootstrap(
    *, method: str, project_root: Path, dataset_root: Path, result_root: Path,
    cache_root: Path | None, checkpoint_path: Path | None, checkpoint_url: str | None,
    device: str,
) -> dict:
    spec = SPECS[method]
    project_root = project_root.resolve()
    dataset_root = dataset_root.resolve()
    result_root, cache_root = validate_output_roots(
        project_root, result_root, cache_root, method
    )
    environment_gate = prepare_result_root_for_bootstrap(result_root, method)
    cache_root.mkdir(parents=True, exist_ok=True)
    source_root = result_root / "source" / "repo"
    source_receipt = clone_source(spec, source_root)
    compatibility_receipt = install_compatibility_patch(method, source_root)
    manifest_path = result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"
    data_receipt = build_manifest(dataset_root, manifest_path)
    checkpoint = result_root / "checkpoints" / spec.checkpoint_filename
    checkpoint_receipt = materialize_checkpoint(spec, checkpoint, checkpoint_path, checkpoint_url)
    adapter_receipt = build_adapter(
        spec, source_root, Path(data_receipt["audio_dir"]), checkpoint, result_root / "portable_run"
    )
    environment_spec = project_root / "baseline" / method / "environment.linux-cu118.yml"
    receipt = {
        "status": "fresh_checkout_bootstrap_ready",
        "created_at": chicago_timestamp(),
        "timezone": "America/Chicago",
        "minimum_release": RELEASE_NAME,
        "base_release_commit": BASE_RELEASE_COMMIT,
        "method": method,
        "project_root": str(project_root),
        "dataset_root": str(dataset_root),
        "result_root": str(result_root),
        "cache_root": str(cache_root),
        "device_requested": device,
        "source": source_receipt,
        "compatibility_patch": compatibility_receipt,
        "data": data_receipt,
        "checkpoint": checkpoint_receipt,
        "adapter": adapter_receipt,
        "environment_gate": {
            "path": str(environment_gate), "sha256": sha256(environment_gate),
        },
        "environment_spec": {
            "name": ENVIRONMENT_NAMES[method],
            "path": str(environment_spec),
            "project_relative_path": str(environment_spec.relative_to(project_root)),
            "sha256": sha256(environment_spec),
        },
        "environment": {
            "HF_HOME": str(cache_root / "huggingface"),
            "TORCH_HOME": str(cache_root / "torch"),
            "NUMBA_CACHE_DIR": str(cache_root / "numba"),
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
        },
    }
    receipts = result_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / "bootstrap_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_bootstrap(method: str, result_root: Path) -> dict:
    result_root = result_root.resolve()
    receipt_path = result_root / "receipts" / "bootstrap_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    spec = SPECS[method]
    source = result_root / "source" / "repo"
    checkpoint = result_root / "checkpoints" / spec.checkpoint_filename
    manifest = result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"
    errors = []
    if run(["git", "rev-parse", "HEAD"], cwd=source) != spec.commit:
        errors.append("source_commit")
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--", "main.py", "util/misc.py"],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if hashlib.sha256(diff).hexdigest() != receipt["compatibility_patch"]["diff_sha256"]:
        errors.append("compatibility_patch_diff")
    if sha256(checkpoint) != spec.checkpoint_sha256:
        errors.append("checkpoint_sha256")
    if sha256(manifest) != receipt["data"]["manifest_sha256"]:
        errors.append("manifest_sha256")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6898 or len({row["cycle_id"] for row in rows}) != 6898:
        errors.append("manifest_rows_or_ids")
    portable = result_root / "portable_run"
    audio_adapter = portable / "data" / "icbhi_dataset" / "audio_test_data"
    wav_links = list(audio_adapter.glob("*.wav"))
    txt_links = [
        path for path in audio_adapter.glob("*.txt")
        if len(path.stem.split("_")) >= 5
    ]
    if (
        len(wav_links) != 920
        or len(txt_links) != 920
        or not all(path.is_symlink() for path in [*wav_links, *txt_links])
    ):
        errors.append("data_adapter")
    if not (portable / "pretrained_models" / spec.checkpoint_filename).is_symlink():
        errors.append("checkpoint_adapter")
    if receipt["adapter"].get("author_split_cycle_counts") != {"train": 4142, "test": 2756}:
        errors.append("author_split_cycle_counts")
    environment_spec = Path(receipt["environment_spec"]["path"])
    if receipt.get("minimum_release") != RELEASE_NAME:
        errors.append("release_name")
    if receipt.get("base_release_commit") != BASE_RELEASE_COMMIT:
        errors.append("base_release_commit")
    if receipt["environment_spec"].get("name") != ENVIRONMENT_NAMES[method]:
        errors.append("environment_name")
    if not environment_spec.is_file() or sha256(environment_spec) != receipt["environment_spec"]["sha256"]:
        errors.append("environment_spec_sha256")
    environment_gate = Path(receipt["environment_gate"]["path"])
    if not environment_gate.is_file() or sha256(environment_gate) != receipt["environment_gate"]["sha256"]:
        errors.append("environment_gate_sha256")
    result = {
        "status": "verified" if not errors else "failed",
        "method": method,
        "minimum_release": RELEASE_NAME,
        "base_release_commit": BASE_RELEASE_COMMIT,
        "environment_name": ENVIRONMENT_NAMES[method],
        "result_root": str(result_root),
        "errors": errors,
        "cycles": len(rows),
        "unique_cycle_ids": len({row["cycle_id"] for row in rows}),
        "source_commit": spec.commit,
        "checkpoint_sha256": sha256(checkpoint),
    }
    (result_root / "receipts" / "bootstrap_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if errors:
        raise ValueError(f"bootstrap verification failed: {errors}")
    return result
