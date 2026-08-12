"""P4 official HeAR 1.0.0 two-second Keras serving binding.

HeAR model files are gated by the Health AI Developer Foundations terms.  This
module never downloads or authenticates.  A caller must provide an accepted,
immutable local SavedModel directory and its deterministic tree SHA256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .preflight import CandidateDimensionAdapter
from .window_encoder import (
    AdapterProvenance,
    FrozenWindowBackend,
    ProductionWindowEncoder,
    sha256_file,
)


HEAR_IDENTITY = "HeAR"
HEAR_MODEL_ID = "google/hear"
HEAR_MODEL_VERSION = "1.0.0"
HEAR_SOURCE_URL = "https://github.com/Google-Health/hear"
HEAR_CODE_LICENSE = "Apache-2.0"
HEAR_MODEL_LICENSE = "Health AI Developer Foundations terms of use"
HEAR_WINDOW_SAMPLES = 32_000


def saved_model_tree_sha256(path: Path) -> str:
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HeAR SavedModel directory is missing: {root}")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError("HeAR SavedModel directory is empty")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix()
        file_digest = sha256_file(item)
        digest.update(f"{relative}\0{item.stat().st_size}\0{file_digest}\n".encode())
    return digest.hexdigest()


class HeARWindowBackend(FrozenWindowBackend):
    """Keras serving bridge with an explicit DLPack PyTorch/TF boundary."""

    native_dim = 512

    def __init__(
        self,
        serving_signature: object,
        *,
        device: torch.device | str = "cpu",
        tensorflow_module: object | None = None,
    ) -> None:
        super().__init__()
        self.serving_signature = serving_signature
        self.tensorflow_module = tensorflow_module
        self.register_buffer(
            "device_anchor", torch.empty(0, device=torch.device(device)), persistent=False
        )

    def encode_valid_windows(
        self, waveform_windows: torch.Tensor, valid_samples: torch.Tensor
    ) -> torch.Tensor:
        if waveform_windows.device != self.device_anchor.device:
            raise RuntimeError("HeAR windows and explicit bridge device must match")
        if waveform_windows.ndim != 2 or waveform_windows.shape[1] != HEAR_WINDOW_SAMPLES:
            raise ValueError("HeAR requires exact [N,32000] two-second 16 kHz windows")
        if bool((valid_samples <= 0).any()) or bool((valid_samples > HEAR_WINDOW_SAMPLES).any()):
            raise ValueError("invalid HeAR valid_samples")
        tf = self.tensorflow_module
        if tf is None:
            try:
                import tensorflow as tf
            except (ImportError, OSError) as error:
                raise RuntimeError("HeAR production binding requires TensorFlow/tf_keras") from error
        dlpack = getattr(getattr(tf, "experimental", None), "dlpack", None)
        if dlpack is not None:
            tensor = dlpack.from_dlpack(
                torch.utils.dlpack.to_dlpack(waveform_windows.detach().contiguous())
            )
        elif waveform_windows.device.type == "cpu":
            tensor = tf.convert_to_tensor(waveform_windows.detach().numpy())
        else:
            raise RuntimeError(
                "HeAR CUDA bridge requires TensorFlow experimental.dlpack support"
            )
        output = self.serving_signature(x=tensor)
        if not isinstance(output, dict) or "output_0" not in output:
            raise RuntimeError("HeAR serving signature must return output_0")
        raw_output = output["output_0"]
        if dlpack is not None:
            values = torch.utils.dlpack.from_dlpack(
                dlpack.to_dlpack(raw_output)
            ).to(torch.float32)
        elif waveform_windows.device.type == "cpu":
            values = torch.from_numpy(raw_output.numpy()).to(torch.float32)
        else:
            raise RuntimeError("HeAR output cannot cross the framework boundary safely")
        if values.shape != (waveform_windows.shape[0], self.native_dim):
            raise RuntimeError(f"HeAR encoder returned {tuple(values.shape)}")
        return values


def _load_hear_saved_model(model_dir: Path, expected_tree_sha256: str) -> object:
    if len(expected_tree_sha256) != 64:
        raise ValueError("HeAR requires a full deterministic SavedModel tree SHA256")
    actual = saved_model_tree_sha256(model_dir)
    if actual != expected_tree_sha256.lower():
        raise RuntimeError(f"HeAR SavedModel tree SHA mismatch: {actual}")
    try:
        import tensorflow as tf
    except (ImportError, OSError) as error:
        raise RuntimeError("HeAR production binding requires TensorFlow/tf_keras") from error
    model = tf.saved_model.load(str(model_dir.resolve()))
    signatures = getattr(model, "signatures", None)
    if not signatures or "serving_default" not in signatures:
        raise RuntimeError("HeAR SavedModel lacks serving_default")
    return signatures["serving_default"]


def build_hear_window_encoder(
    model_dir: Path,
    model_revision: str,
    expected_tree_sha256: str,
    *,
    device: torch.device | str = "cpu",
) -> ProductionWindowEncoder:
    target = torch.device(device)
    if not model_revision:
        raise ValueError("HeAR requires an immutable accepted model revision")
    signature = _load_hear_saved_model(model_dir, expected_tree_sha256)
    backend = HeARWindowBackend(signature, device=target).to(target)
    provenance = AdapterProvenance(
        encoder_identity=HEAR_IDENTITY,
        source_url=HEAR_SOURCE_URL,
        source_revision=model_revision,
        source_license=HEAR_CODE_LICENSE,
        checkpoint_name=model_dir.name,
        checkpoint_source=f"{HEAR_MODEL_ID} model version {HEAR_MODEL_VERSION}",
        checkpoint_sha256=expected_tree_sha256,
        checkpoint_size_bytes=sum(
            item.stat().st_size for item in model_dir.rglob("*") if item.is_file()
        ),
        license_boundary=(
            "code repository is Apache-2.0; model use is governed separately by "
            "Health AI Developer Foundations terms"
        ),
    )
    return ProductionWindowEncoder(
        HEAR_IDENTITY,
        backend,
        provenance,
        dimension_adapter=CandidateDimensionAdapter(HEAR_IDENTITY).to(target),
    )
