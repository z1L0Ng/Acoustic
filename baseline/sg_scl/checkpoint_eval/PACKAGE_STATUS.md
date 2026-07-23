# Package Status

Status: `READY_FOR_IMMUTABLE_SNAPSHOT_CONDITIONAL_ON_SERVER_CHECKPOINT`

The SG-SCL-to-SPRSound B2 package has passed local compilation,
target-contract, notebook-cleanliness, path, and static package verification.
It deliberately has not run checkpoint inference locally because the verified
epoch-27 task checkpoint is server-owned and no checkpoint binary may be
invented or copied into Git.

Server execution remains gated by an explicit task-checkpoint path and SHA256,
bootstrap success, then an 8-event label-free smoke plus independent smoke
verification. The source method is metadata/device-aware during training, but
the author `validate()` path uses only the audio encoder and class classifier.
This package does not invent SPRSound device/domain labels.
