# Package Status

Status: `READY_FOR_IMMUTABLE_SNAPSHOT_CONDITIONAL_ON_SERVER_CHECKPOINT`

The PAFA-to-SPRSound B1 package has passed local compilation, target-contract,
notebook-cleanliness, path, and static package verification. It deliberately
has not run checkpoint inference locally because the accepted task-checkpoint
container is server-owned and no checkpoint binary may be invented or copied
into Git. The immutable contract distinguishes container epoch 100 from
selected best epoch 27 and pins size, SHA256, and embedded-best-state equality.

Server execution remains gated by an explicit task-checkpoint path and SHA256,
the audited BEATs backbone path and SHA256, bootstrap success, then an 8-event
label-free smoke plus independent smoke verification. Full 1,429-event
inference is a separate explicit command after those gates pass.
