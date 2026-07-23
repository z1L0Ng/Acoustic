from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .config import LABEL_COLUMNS, MANIFEST_PATH, Protocol


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype={"cycle_id": str, "recording_id": str})
    required = {
        "cycle_id", "recording_id", "patient_id", "group_id", "official_split",
        "native_four_class_label", "binary_label",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if len(manifest) != 6898 or not manifest["cycle_id"].is_unique:
        raise ValueError("Manifest must contain 6,898 unique cycle IDs")
    counts = manifest["official_split"].value_counts().to_dict()
    if counts != {"train": 4142, "test": 2756}:
        raise ValueError(f"Unexpected official split counts: {counts}")
    return manifest


def validation_assignment(manifest: pd.DataFrame, protocol: Protocol) -> pd.DataFrame:
    assignment = manifest[["cycle_id", "recording_id", "patient_id", "official_split"]].copy()
    assignment["partition"] = np.where(assignment["official_split"].eq("test"), "test", "subtrain")
    train = manifest[manifest["official_split"].eq("train")].reset_index()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=protocol.validation_seed)
    _, val_pos = next(splitter.split(train, train["native_four_class_label"], train["patient_id"]))
    val_ids = set(train.iloc[val_pos]["cycle_id"])
    assignment.loc[assignment["cycle_id"].isin(val_ids), "partition"] = "validation"
    if set(assignment.loc[assignment["partition"].eq("validation"), "patient_id"]) & set(
        assignment.loc[assignment["partition"].eq("subtrain"), "patient_id"]
    ):
        raise AssertionError("Patient leakage between subtrain and validation")
    return assignment


def load_aligned_features(
    manifest: pd.DataFrame,
    feature_path: Path,
    feature_key: str,
    expected_dim: int,
    allow_pickle_metadata: bool = False,
    expected_sha256: str | None = None,
) -> tuple[np.ndarray, dict]:
    actual_sha256 = sha256_file(feature_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"Feature SHA256 mismatch: {actual_sha256}")
    with np.load(feature_path, allow_pickle=allow_pickle_metadata) as archive:
        required = {feature_key, "cycle_id", "native_four_class_label", "binary_label", "official_split"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Feature archive missing keys: {sorted(missing)}")
        if allow_pickle_metadata:
            for key in required.difference({feature_key}):
                value = archive[key]
                if value.dtype == object and not all(isinstance(item, str) for item in value.tolist()):
                    raise ValueError(f"Unsafe object metadata in {key}")
        ids = archive["cycle_id"].astype(str)
        if len(ids) != len(set(ids)):
            raise ValueError("Feature cycle IDs are not unique")
        if set(ids) != set(manifest["cycle_id"]):
            raise ValueError("Feature and manifest cycle ID sets differ")
        position = {cycle_id: index for index, cycle_id in enumerate(ids)}
        order = np.asarray([position[value] for value in manifest["cycle_id"]], dtype=np.int64)
        features = np.asarray(archive[feature_key][order], dtype=np.float32)
        if features.shape != (len(manifest), expected_dim):
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        for key, column in (
            ("native_four_class_label", "native_four_class_label"),
            ("binary_label", "binary_label"),
            ("official_split", "official_split"),
        ):
            actual = archive[key][order].astype(str)
            expected = manifest[column].astype(str).to_numpy()
            if not np.array_equal(actual, expected):
                raise ValueError(f"Feature metadata mismatch for {key}")
        metadata = {}
        for key in archive.files:
            if key not in {feature_key, "cycle_id", "native_four_class_label", "binary_label", "official_split"}:
                value = archive[key]
                if value.size <= 8:
                    metadata[key] = value.astype(str).tolist()
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity")
    return features, metadata


def split_arrays(
    features: np.ndarray,
    manifest: pd.DataFrame,
    assignment: pd.DataFrame,
    task: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    labels = manifest[LABEL_COLUMNS[task]].astype(str).to_numpy()
    output = {}
    for partition in ("subtrain", "validation", "test"):
        mask = assignment["partition"].eq(partition).to_numpy()
        output[partition] = (features[mask], labels[mask], manifest.loc[mask, "cycle_id"].to_numpy())
    return output


def fit_scaler(partitions: dict) -> tuple[StandardScaler, dict]:
    scaler = StandardScaler()
    # sklearn/Accelerate can overflow in float32 variance accumulation even for
    # bounded embeddings. Fit and transform in float64, then return float32.
    scaler.fit(partitions["subtrain"][0].astype(np.float64, copy=False))
    scaled = {
        name: (
            scaler.transform(values[0].astype(np.float64, copy=False)).astype(np.float32),
            values[1],
            values[2],
        )
        for name, values in partitions.items()
    }
    return scaler, scaled


def provenance_row(backbone: str, config: dict, feature_metadata: dict) -> dict:
    path = Path(config["path"])
    meta_path = path.with_name(path.stem + "_meta.json")
    extraction_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {
        "backbone": backbone,
        "feature_path_absolute": str(path.resolve()),
        "feature_path_relative": str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else str(path),
        "feature_key": config["key"],
        "shape": f"6898x{config['expected_dim']}",
        "sha256": sha256_file(path),
        "encoder": config["encoder"],
        "npz_metadata_json": json.dumps(feature_metadata, sort_keys=True),
        "extraction_meta_path": str(meta_path) if meta_path.exists() else "",
        "extraction_meta_json": json.dumps(extraction_meta, sort_keys=True),
        "allow_pickle_metadata": bool(config.get("allow_pickle_metadata", False)),
        "source_runner": str(config.get("source_runner", "")),
        "source_receipt": str(config.get("source_receipt", "")),
        "provenance_note": config.get("provenance_note", ""),
    }
