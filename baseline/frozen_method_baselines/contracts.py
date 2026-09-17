"""Scientific contracts and asset paths for frozen method baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


SEEDS = (0, 1, 42)
TASKS = ("icbhi_flat4", "spr_binary", "hf_cas", "kauh_binary")


@dataclass(frozen=True)
class FrozenMethodPlan:
    method_id: str
    evidence_label: str
    status: str
    encoder_history: str
    encoder_checkpoint: str | None
    embedding_cache: str | None
    trainable_modules: tuple[str, ...]
    auxiliary_gradient_path: str
    claim_boundary: tuple[str, ...]
    tasks: tuple[str, ...] = TASKS
    seeds: tuple[int, ...] = SEEDS

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PAFA_METHOD_ENCODER = FrozenMethodPlan(
    method_id="pafa_trained_encoder_frozen_target_adaptation",
    evidence_label="method_trained_representation_target_supervised_heads",
    status="ASSETS_PRESENT_IMPLEMENTATION_PREPARED_LOCAL_ENVIRONMENT_HOLD",
    encoder_history=(
        "BEATs iter3+ AS2M full-fine-tuned by PAFA on ICBHI; accepted state is "
        "official-test-selected epoch 27 embedded in the saved container"
    ),
    encoder_checkpoint=".cache/checkpoints/pafa/server_epoch27/best.pth",
    embedding_cache=".cache/four_dataset_pafa_frozen_encoder/embeddings.npz",
    trainable_modules=(
        "per-task LayerNorm(768)-Linear(768,256)-ReLU-Dropout adapter",
        "per-task native classifier",
    ),
    auxiliary_gradient_path=(
        "none during downstream adaptation; PAFA PCSL/GPAL affected the encoder during "
        "its historical source training, and the frozen representation is the method object"
    ),
    claim_boundary=(
        "PAFA-trained frozen representation, not PAFA reproduction",
        "downstream heads are target-supervised",
        "three seeds vary downstream optimization only, not encoder training",
        "encoder source checkpoint is ICBHI-test-selected",
    ),
)


PCMCL_METHOD_ENCODER = FrozenMethodPlan(
    method_id="pcmcl_trained_encoder_frozen_target_adaptation",
    evidence_label="method_trained_representation_target_supervised_heads",
    status="HOLD_MISSING_TRAINED_PCMCL_CHECKPOINT_AND_MATCHED_CACHE",
    encoder_history=(
        "would require a PC-MCL-trained BEATs encoder from official 10-s multi-cycle, "
        "3-label and patient-matching training"
    ),
    encoder_checkpoint=None,
    embedding_cache=None,
    trainable_modules=(
        "per-task downstream adapter",
        "per-task native classifier",
    ),
    auxiliary_gradient_path=(
        "PC-MCL objectives would already be encoded in the frozen method-trained weights; "
        "no such local weights are currently available"
    ),
    claim_boundary=(
        "cannot substitute AudioSet-only BEATs weights",
        "cannot silently insert a new full PC-MCL source-training run",
        "not ready for Table 1 execution",
    ),
)


PCMCL_INSPIRED_ADAPTER = FrozenMethodPlan(
    method_id="pcmcl_inspired_frozen_embedding_adapter",
    evidence_label="frozen_audioset_beats_target_supervised_pcmcl_inspired_adapter",
    status="SUPERSEDED_BY_ICBHI5S_SOURCE_TRAINED_FIXED_TRANSFER",
    encoder_history="generic BEATs iter3+ AS2M AudioSet pretraining only",
    encoder_checkpoint=".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt",
    embedding_cache=(
        ".cache/four_dataset_representation_attribution/"
        "r1_beats_as2m_audioset_only/embeddings.npz"
    ),
    trainable_modules=(
        "shared LayerNorm(768)-Linear(768,256)-ReLU-Dropout adapter",
        "dataset-native heads",
        "3-label Normal/Crackle/Wheeze auxiliary head where labels are supported",
        "patient-pair classifier where true patient grouping and repeated units exist",
    ),
    auxiliary_gradient_path=(
        "native classification, 3-label and patient-matching losses all update the same "
        "shared 256-d adapter; the frozen encoder is unchanged"
    ),
    claim_boundary=(
        "PC-MCL-inspired adaptation, not PC-MCL reproduction",
        "embedding-pair operations are not raw-audio multi-cycle concatenation",
        "HF date proxy is not a patient ID",
        "three seeds vary adapter/head/pair sampling only",
    ),
)


PCMCL_FROZEN_RAW_CONCAT = FrozenMethodPlan(
    method_id="pcmcl_frozen_raw_concat_adaptation",
    evidence_label="frozen_audioset_beats_raw_concat_pcmcl_mechanism_adaptation",
    status="SUPERSEDED_BY_ICBHI5S_SOURCE_TRAINED_FIXED_TRANSFER",
    encoder_history="generic BEATs iter3+ AS2M AudioSet pretraining only",
    encoder_checkpoint=".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt",
    embedding_cache=None,
    trainable_modules=(
        "shared LayerNorm(768)-Linear(768,256)-ReLU-Dropout adapter",
        "Normal/Crackle/Wheeze pathology head",
        "same/different-patient head",
    ),
    auxiliary_gradient_path=(
        "the pathology and patient losses update the same adapter after the frozen "
        "encoder processes the complete 10-s raw concatenation"
    ),
    claim_boundary=(
        "retains raw multi-unit concatenation, 3-label supervision and patient matching",
        "changes the original full-fine-tuning recipe by freezing BEATs",
        "extends the original ICBHI-only method to a joint ICBHI/SPRSound core",
        "requires new composite-input features; pooled single-unit caches are invalid",
    ),
)


PCMCL_ICBHI5S_SOURCE_TRANSFER = FrozenMethodPlan(
    method_id="pcmcl_icbhi5s_source_trained_fixed_transfer",
    evidence_label="icbhi_test_selected_source_model_fixed_external_transfer",
    status="DESIGN_READY_SOURCE_TRAINING_NOT_AUTHORIZED",
    encoder_history=(
        "generic BEATs iter3+ AS2M initialization followed by full ICBHI-only "
        "5-s PC-MCL source training; no trained source checkpoint exists yet"
    ),
    encoder_checkpoint=None,
    embedding_cache=None,
    trainable_modules=(
        "entire BEATs encoder and frontend during ICBHI source training",
        "Normal/Crackle/Wheeze classifier",
        "same/different-patient classifier",
    ),
    auxiliary_gradient_path=(
        "N/C/W BCE and patient CE update the shared BEATs encoder during source "
        "training; the selected whole model is frozen for all target readouts"
    ),
    claim_boundary=(
        "ICBHI-only source-trained PC-MCL 5-s adaptation, not the paper's 10-s result",
        "SPRSound, HF and KAUH receive no target training, selection or threshold fit",
        "generic BEATs initialization is not a PC-MCL checkpoint",
        "three seeds are three independent source trainings",
    ),
)


DCASE_RESPIRATORY_ADAPTATION = FrozenMethodPlan(
    method_id="dcase_masked_crnn_respiratory_adaptation",
    evidence_label="frozen_beats_frame_fusion_missing_label_respiratory_adaptation",
    status="DESIGN_READY_FRAME_CACHE_AND_PROTOCOL_APPROVAL_HOLD",
    encoder_history="generic BEATs AudioSet pretraining used as frozen frame embeddings",
    encoder_checkpoint=".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt",
    embedding_cache=None,
    trainable_modules=(
        "official-style log-Mel CNN branch",
        "BEATs/CNN temporal fusion projection",
        "bidirectional GRU",
        "four-channel frame classifier and class-masked attention readout",
    ),
    auxiliary_gradient_path=(
        "eligible weak respiratory labels update the CNN, temporal fusion, BiGRU, "
        "classifier and attention modules; BEATs remains frozen"
    ),
    claim_boundary=(
        "DCASE-style respiratory adaptation, not DCASE Task 4 reproduction",
        "retains frame fusion and missing-class loss/attention masking",
        "omits Mean Teacher without an approved unlabeled respiratory pool",
        "requires new frame features; pooled single-unit caches are invalid",
    ),
)


PLANS = {
    plan.method_id: plan
    for plan in (
        PAFA_METHOD_ENCODER,
        PCMCL_METHOD_ENCODER,
        PCMCL_INSPIRED_ADAPTER,
        PCMCL_FROZEN_RAW_CONCAT,
        PCMCL_ICBHI5S_SOURCE_TRANSFER,
        DCASE_RESPIRATORY_ADAPTATION,
    )
}


def asset_status(repo_root: Path, plan: FrozenMethodPlan) -> dict[str, object]:
    checkpoint = repo_root / plan.encoder_checkpoint if plan.encoder_checkpoint else None
    cache = repo_root / plan.embedding_cache if plan.embedding_cache else None
    return {
        "method_id": plan.method_id,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "checkpoint_present": bool(checkpoint and checkpoint.is_file()),
        "cache_path": str(cache) if cache else None,
        "cache_present": bool(cache and cache.is_file()),
        "status": plan.status,
    }
