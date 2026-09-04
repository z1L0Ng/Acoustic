"""PAFA-JH3.3 unit-level scalar log-fbank MVN diagnostic runner.

This isolated runner reuses the JH3 Soft Confidence Bridge training loop and
changes only the BEATs frontend normalization.  It keeps the canonical
128-bin Kaldi log-fbank, removes the fixed BEATs affine, and applies scalar
mean/variance normalization independently to each unit over all time/frequency
values.  The run remains test-exposed and is not a clean estimate or a PAFA
reproduction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from baseline.pafa import joint_hierarchy_soft_bridge as soft_bridge
from baseline.pafa.joint_hierarchy import PAFAJointHierarchyModel, PAFAJointHierarchyConfig


CONDITION = "PAFA_JH3_3_soft_bridge_mvn_seed42"
EVIDENCE_LABEL = (
    "icbhi_test60_spr_validation40_selected_soft_bridge_unit_scalar_fbank_mvn_diagnostic"
)
CHANGED_FILES = ("baseline/pafa/joint_hierarchy_soft_bridge_mvn.py",)


JH3_3_SELECTION = soft_bridge.SelectionSpec(
    condition=CONDITION,
    evidence_label=EVIDENCE_LABEL,
    icbhi_weight=0.6,
    spr_validation_weight=0.4,
    changed_files=CHANGED_FILES,
)


def _unit_scalar_mvn_log_fbank(source: torch.Tensor) -> torch.Tensor:
    import torchaudio

    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=128,
            sample_frequency=16000,
            frame_length=25,
            frame_shift=10,
        ).to(torch.float32)
        mean = fbank.mean()
        variance = (fbank - mean).square().mean()
        fbanks.append((fbank - mean) / torch.sqrt(variance + 1e-5))
    return torch.stack(fbanks, dim=0)


class UnitScalarMVNBEATs(nn.Module):
    """The author BEATs module with only its fbank normalization replaced."""

    def __init__(self, transfer_model: nn.Module) -> None:
        super().__init__()
        self.beats = transfer_model.beats
        self.cfg = transfer_model.cfg
        self.final_feat_dim = transfer_model.final_feat_dim

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        training: bool = False,
    ) -> torch.Tensor:
        fbank = _unit_scalar_mvn_log_fbank(source)

        if self.beats.spec_transform is not None and self.training:
            fbank = self.beats.spec_transform(fbank.unsqueeze(1)).squeeze(1)

        if padding_mask is not None:
            padding_mask = self.beats.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.beats.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.beats.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.beats.forward_padding_mask(features, padding_mask)

        if self.beats.post_extract_proj is not None:
            features = self.beats.post_extract_proj(features)

        features = self.beats.dropout_input(features)
        features, _ = self.beats.encoder(features, padding_mask=padding_mask)
        return features


def _build_components(
    config: PAFAJointHierarchyConfig,
    device: torch.device,
) -> tuple[PAFAJointHierarchyModel, nn.Module]:
    sys.path.insert(0, str(config.author_repo))
    try:
        from method.pafa import PAFALoss, ProjectionHead
        from models.beats import BEATsTransferLearningModel
    finally:
        sys.path.pop(0)

    transfer_model = BEATsTransferLearningModel(
        num_target_classes=4,
        model_path=str(config.checkpoint),
        ft_entire_network=True,
        spec_transform=None,
    )
    beats = UnitScalarMVNBEATs(transfer_model)
    projector = ProjectionHead(
        input_dim=beats.final_feat_dim,
        hidden_dim=None,
        output_dim=config.projection_output_dim,
        attention=config.projection_attention,
        proj_type="end2end",
        norm_type=config.projection_norm,
    )
    return PAFAJointHierarchyModel(beats, projector).to(device), PAFALoss().to(device)


_ORIGINAL_CONFIG_PAYLOAD = soft_bridge._config_payload


def _config_payload(
    config: PAFAJointHierarchyConfig,
    selection: soft_bridge.SelectionSpec = JH3_3_SELECTION,
) -> dict[str, object]:
    payload = _ORIGINAL_CONFIG_PAYLOAD(config, selection)
    payload.update(
        {
            "condition": selection.condition,
            "evidence_label": selection.evidence_label,
            "training_config_reused_from": "PAFA_JH3_1_soft_bridge_weighted_selection_seed42",
            "method_change": "unit-level scalar log-fbank MVN frontend",
            "frontend_normalization": {
                "waveform": "mono 16 kHz; 5 s repeat-pad/front-truncate",
                "fbank": (
                    "canonical BEATs torchaudio Kaldi log-fbank; num_mel_bins=128; "
                    "sample_frequency=16000; frame_length=25 ms; frame_shift=10 ms"
                ),
                "fixed_beats_affine": "not applied",
                "scope": "each unit independently over all T,F values",
                "mean": "mean over all T,F",
                "variance": "mean((F-mu)^2), unbiased=False",
                "transform": "(F-mu)/sqrt(variance+1e-5)",
                "waveform_peak_rms_normalization": "none",
                "spec_augment": "none",
                "dataset_specific_statistics": "none",
                "additional_normalization": "none",
            },
        }
    )
    return payload


def _install_isolated_overrides() -> None:
    soft_bridge._build_components = _build_components
    soft_bridge._config_payload = _config_payload


def _default_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    author_repo = (
        repo_root / "result/pafa_sprsound_transfer_20260722_235659/source/repo"
    )
    checkpoint = (
        repo_root
        / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt"
    )
    icbhi_audio = (
        repo_root
        / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database"
    )
    return author_repo, checkpoint, icbhi_audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--icbhi-audio-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    author_repo, checkpoint, icbhi_audio = _default_paths(args.repo_root)
    config = PAFAJointHierarchyConfig(
        repo_root=args.repo_root,
        author_repo=args.author_repo or author_repo,
        checkpoint=args.checkpoint or checkpoint,
        icbhi_audio_dir=args.icbhi_audio_dir or icbhi_audio,
        output_dir=args.output_dir
        or args.repo_root / "result/reproduce/pafa_joint_hierarchy" / CONDITION,
        device=args.device,
        cpu_threads=args.cpu_threads,
    )
    config.validate()
    if not args.run:
        print(
            __import__("json").dumps(
                {
                    "status": "READY_FOR_USER_START",
                    "execution_started": False,
                    "evidence_label": EVIDENCE_LABEL,
                    "config": _config_payload(config),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    _install_isolated_overrides()
    print(
        __import__("json").dumps(
            soft_bridge.run(config, selection=JH3_3_SELECTION),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
