from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "dataset/processed/manifests/icbhi_2017_cycles.csv"
FEATURE_ROOT = REPO_ROOT / ".cache/features/icbhi_2017"
RESULT_ROOT = REPO_ROOT / "result/icbhi_frozen_downstream/architecture"
IMBALANCE_RESULT_ROOT = REPO_ROOT / "result/icbhi_frozen_downstream/loss"
EXTENSION_RESULT_ROOT = REPO_ROOT / "result/icbhi_frozen_downstream"
EXTENSION_ARCHITECTURE_ROOT = EXTENSION_RESULT_ROOT / "architecture_extension"
EXTENSION_IMBALANCE_ROOT = EXTENSION_RESULT_ROOT / "loss_extension"
IMBALANCE_LOSSES = (
    "unweighted_ce",
    "class_weighted_ce",
    "focal_loss",
    "class_balanced_ce",
)
LABELS = {
    "flat4": ["normal", "crackle", "wheeze", "both"],
    "binary": ["normal", "abnormal"],
}
LABEL_COLUMNS = {
    "flat4": "native_four_class_label",
    "binary": "binary_label",
}
CORE_FEATURES = {
    "ast": {
        "path": FEATURE_ROOT / "ast_frozen_features_full.npz",
        "key": "ast_cls",
        "expected_dim": 768,
        "encoder": "MIT/ast-finetuned-audioset-10-10-0.4593",
    },
    "clap": {
        "path": FEATURE_ROOT / "clap_frozen_features_full.npz",
        "key": "clap",
        "expected_dim": 512,
        "encoder": "laion_clap_htsat_tiny_630k_audioset_nonfusion",
    },
    "beats": {
        "path": FEATURE_ROOT / "beats_frozen_features_full.npz",
        "key": "beats",
        "expected_dim": 768,
        "encoder": "beats_iter3_plus_AS2M_pretrained_meanpool",
    },
}
EXTENSION_FEATURES = {
    "simple_acoustic": {
        "path": FEATURE_ROOT / "simple_acoustic_features.npz",
        "key": "X",
        "expected_dim": 114,
        "encoder": "handcrafted_114d_acoustic_summary",
        "result_name": "simple_acoustic",
        "allow_pickle_metadata": True,
        "expected_sha256": "b23af08e19425b7b113efdb2ed78cf12340c0ada8f7dcc26074ad08e6d0e7589",
        "provenance_note": "Local handcrafted feature baseline; object-dtype metadata contains audited Python strings only.",
    },
    "hear": {
        "path": FEATURE_ROOT / "hear_frozen_features_full.npz",
        "key": "hear_embedding",
        "expected_dim": 512,
        "encoder": "google/hear-pytorch",
        "result_name": "hear",
        "expected_sha256": "c1e6310c76cdb6a1959c678406a9ac0ec67282991c05acf75a03ee40a8b21a17",
    },
    "opera_ct_official_like": {
        "path": FEATURE_ROOT / "operaCT_official_like_full.npz",
        "key": "operaCT",
        "expected_dim": 768,
        "encoder": "operaCT_official_like",
        "result_name": "opera",
        "expected_sha256": "d6dd64b050edddf10d03b9788525a1af7bbd40d0a0c866215dfa69a487bdc57b",
        "provenance_note": (
            "OPERA official-repo feature path adapted to the local ICBHI cycle task; "
            "not an official OPERA paper ICBHI reproduction."
        ),
    },
}
FEATURES = {**CORE_FEATURES, **EXTENSION_FEATURES}


@dataclass(frozen=True)
class Protocol:
    protocol_version: str = "formal-downstream-v1"
    dataset: str = "ICBHI 2017"
    test_split: str = "official challenge split (not patient-independent)"
    validation_method: str = "StratifiedGroupKFold(n_splits=5, fold=0) on official train"
    validation_group: str = "patient_id"
    validation_seed: int = 20260712
    normalization: str = "StandardScaler fit on official subtrain only; float64 statistics, float32 model input"
    class_balance: str = "balanced class weights computed from official subtrain labels"
    decision_rule: str = "argmax; no test threshold search"
    lr_solver: str = "lbfgs"
    lr_c: float = 1.0
    lr_max_iter: int = 5000
    hidden_1: int = 256
    hidden_2: int = 128
    activation: str = "ReLU"
    dropout: float = 0.3
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 8
    selection_metric: str = "validation macro F1"
    seeds: tuple[int, ...] = (20260712, 20260713, 20260714)
    split_caveat: str = "Patients 156 and 218 occur in both official train and official test."

    def to_dict(self) -> dict:
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        return data


def imbalance_protocol(protocol: Protocol | None = None) -> dict:
    protocol = protocol or Protocol()
    return {
        **protocol.to_dict(),
        "protocol_version": "formal-imbalance-loss-v1",
        "task": "ICBHI cycle-level flat4 only",
        "architecture": "MLP2: input -> 256 -> 128 -> 4; ReLU; dropout=0.3",
        "controlled_variable": "training loss only",
        "losses": {
            "unweighted_ce": {
                "definition": "torch CrossEntropyLoss without class weights",
            },
            "class_weighted_ce": {
                "definition": "CrossEntropyLoss with balanced inverse-frequency weights from official subtrain",
                "formula": "w_c = N / (K * n_c)",
            },
            "focal_loss": {
                "definition": "multi-class focal cross entropy",
                "gamma": 2.0,
                "alpha": None,
            },
            "class_balanced_ce": {
                "definition": "Cui et al. effective-number weighting applied to cross entropy",
                "formula": "w_c = (1-beta) / (1-beta**n_c), normalized to mean 1",
                "beta": 0.9999,
            },
        },
        "excluded_variables": [
            "binary task", "sampler", "LungMix", "logit adjustment", "multi-label formulation",
            "temporal pooling", "test threshold search", "broad hyperparameter sweep",
        ],
        "scope": "Local controlled loss comparison; not a reproduction of any loss paper.",
    }
