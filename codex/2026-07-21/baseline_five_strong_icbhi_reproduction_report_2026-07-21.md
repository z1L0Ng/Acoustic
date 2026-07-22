# 五个 strong ICBHI 方法复现准备报告

## Executive summary

本轮完成的不是 Mac 本地 full numerical reproduction，而是五个方法的
fresh-checkout server-runnable 准备。Release 1 已冻结 Patch-Mix CL 和 PAFA；
Release 2 候选补齐 SG-SCL、MVST 和 ADD-RSC。五个方法都可在 Linux/CUDA
服务器从 tracked code 重建 source/data/checkpoint/smoke/full command，但目前
均没有新的 one-seed full 论文数值，不能声称复现了 paper mean 或 aggregate。

优先级保持：Patch-Mix CL -> PAFA -> SG-SCL -> MVST -> ADD-RSC。Patch-Mix
仍是第一个 L40 source anchor；PAFA 在其不可运行时作为 backup。SG-SCL 是
metadata/device-aware 分支，不是纯音频 baseline。MVST 只能按作者固定 random
file split 报告。ADD-RSC 的 paper/repo 矛盾无法消除，只能运行两条命名清楚的
bounded protocol。

## 当前状态

| Method | Runnable classification | Local gates | Full numerical status | Main caveat |
|---|---|---|---|---|
| Patch-Mix CL | official-like, test-selected | source/env/data/smoke/100-step profile passed | pending CUDA | current author-hosted state dict verified, historical byte identity unresolved |
| PAFA | official-like, test-selected | source/env/data/patient-ID/smoke/100-step profile passed | pending CUDA | BEATs mirror compatibility checkpoint; no repo license |
| SG-SCL | metadata-aware official-like, test-selected | fresh bootstrap + 8-cycle train/prediction smoke passed | pending CUDA | device metadata required; no repo license/task checkpoint |
| MVST | author-code fixed-random-file-split, test-selected | fresh bootstrap + five-view/fusion smoke passed | pending CUDA | not official ICBHI split; row-order identity is a hard gate |
| ADD-RSC | dual bounded reconstruction/execution | fresh bootstrap + both 8-cycle smokes passed | pending CUDA | no unique faithful protocol; paper/repo split, mel and WD conflict |

## Release 2 verified facts

### SG-SCL

Pinned source commit `66564609595090b61540595d3d27764c00553086` and
checkpoint SHA256 `dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f`.
All 6,898 cycles receive exactly one author-compatible device label. The official
split is 4,142/2,756. Eight-cycle smoke produced 768-d embeddings and four-class
logits with finite CE, MetaCL, total loss, and gradients. Prediction-derived
metric wiring verified. Final results must be labelled metadata-aware and
test-selected.

### MVST

Pinned source commit `51f93fa6ffa580d0819ccb59f861582927937264` and
checkpoint SHA256 `dc71a6d4d07aeb7e746547f72a141f404e4c167d660bf003179f3865e06a970c`.
The executable split is 4,213/2,685 cycles from 552/368 recordings after
`random.Random(1)`. All five views generated ordered ID-bearing 768-d features;
IDs and labels were identical before finite gated-fusion smoke. Only the 16x16
patch projection matches the checkpoint, exactly as handled by author code.

### ADD-RSC

Pinned source commit `e2b0f8213cb7ca451ef28757cb1329e17469fe72` and
compatibility checkpoint SHA256
`dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f`.
The `paper_declared_reconstruction` track uses official split, 64 mel bins and
weight decay 0.1; the `author_repo_default_official_like` track uses author
random-file split, 128 mel bins and weight decay `1e-6`. Both completed
eight-cycle finite loss/gradient and prediction-wiring smoke. Only the first may
be compared cautiously with the paper target, and even then it is a bounded
reconstruction rather than official faithful reproduction.

## Server execution policy

Each run must create a new America/Chicago timestamp root directly under
`result/`. Bootstrap must clone the pinned author repo, reconstruct the manifest
from `dataset/raw/icbhi_2017`, download or validate the pinned checkpoint, and
write new runtime receipts. No command may depend on a local historical
`result/` tree. Complete an L40 100-step timing/VRAM profile before a full run.

Full result acceptance requires sample-level IDs, logits/probabilities,
predictions, confusion matrix, Sp, Se, and ICBHI Score; prediction-derived metric
must match the author metric. Official-split methods require 2,756 unique test
cycles. MVST and ADD-RSC author-code tracks require their explicitly recorded
2,685-cycle random-file test partition. Test-per-epoch selection must remain
visible in titles and tables.

## Conservative conclusion

The five methods are packaged for server execution, not numerically reproduced.
The next rational action is one Patch-Mix seed on L40, followed by prediction
identity and paper-distribution review. Only if Patch-Mix has an unrecoverable
runtime/numerical blocker should PAFA become the primary run. The other three
packages are ready but must retain their metadata/split/config claim boundaries.
