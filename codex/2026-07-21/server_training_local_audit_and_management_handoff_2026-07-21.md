# Server Training Coordinator：本地审计与管理线程汇报

日期：2026-07-21

范围：只读审计 + server deployment 文档；未修改 baseline/dataset/model/Notion，未 stage/commit/push，未启动远程训练。

迁移后路径合同：`result/` 是唯一 canonical generated-output root；所有 active server run 使用 `result/<baseline>_<YYYYMMDD_HHMMSS>/`。legacy `results/` 只保留为下述 prohibited historical Git blob 的历史路径证据，不得重建。

## 结论

目前可以完成服务器侧 runbook、指令包、release manifest 模板、run receipt schema 和调度/验收规则；**Release 1 尚未 ready，不能让服务器开始 checkout/下载/训练**。

有两个独立硬门禁：

1. 当前五个 baseline package 和 common reproduction 文件仍是未跟踪并行改动，现有任何 Git SHA 都不包含它们；fresh-checkout rebuildability 正由本地 Baseline Reproduction 线程修复。
2. 当前 `main` 不可直接 push：`origin/main` 已 fetch，`main` 相对其线性 ahead 58、behind 0；待推历史包含一个 1,307,407,924-byte tracked derived feature blob。它在迁移前的 legacy Git path 是 `results/model_design_2026-07-08/icbhi_sequence_pooling/features/beats_sequence_full.npz`，blob `51131f763da6d7de7b3b2780c8db61f92250348a`，由 `8c630c8` 引入。当前本地副本已迁入 canonical `result/`，但目录迁移不会改变历史 blob。首次 push 已安全终止；除它外，`origin/main..main` 未发现 `>=90 MB` blob。

短期推荐方案 A：管理线程从 `origin/main` 新建干净 `codex/l40-training-release`，用新的轻量 commit 只带入必要 baseline/common/server/codex 文件，排除所有完整 feature/result/raw/checkpoint，并在 push 前扫描增量 objects。不要直接从当前 `main` 建分支后推，不要整段 cherry-pick 会携带大 blob 的历史，不要重试当前 main push。

方案 B 是经用户明确批准后重写 `main` 历史以移除该 blob。它会改变 58 个本地 commit SHA，需要备份、停写、force-push 协调、受影响引用盘点和消费者 reclone。它不是短期服务器 rollout 的前置条件，本线程不执行。

## 只读审计证据

### Git / ignore

- branch：`main`。
- HEAD：`6400127d65287dd0bd7150042892769e48881976`。
- `origin/main`：`b9180791c83e2633b6c4ea9d0cac5a0455ed553e`。
- ahead/behind：58/0，线性。
- 五个 baseline 目录、common official reproduction helpers、2026-07-21 contracts/reports 为 untracked；没有 staged changes。
- `.gitignore` 已忽略 `result/` 和 `dataset/raw/**`。完整结果继续通过 rsync/scp 或用户指定存储回传，不走 GitHub。

### 下载脚本 / 数据

- notebook `dataset/script/download_icbhi_2017.ipynb` 的实际入口是 `dataset/script/phase1_dataset_workflow.py icbhi_2017`。
- workflow 会下载五个官方文件、按已知 size 检查、解压并生成 `file_inventory.csv` 与 `checksums_sha256.txt`。
- ICBHI route 使用 unverified TLS context，且脚本没有内置 publisher SHA256；release manifest 必须批准 expected artifact hashes，服务器下载后逐项比较。
- 当前本地观察：920 WAV、922 TXT；cycle manifest 6,898 rows / unique IDs，official train/test 4,142/2,756。
- 当前本地观察的顶层 artifact SHA256：
  - main zip `e34e9913c43ee9da62af4faa7335d8eda7ce9883c315b69b07cfcbde28650e4c`
  - events `d13b5efdcf36635e09731607664f3eea4185b25b2606a742d01be398829d15e5`
  - demographic `7be825ef14639a2dc9f033139158ddd2d7368c26fb6c871fa7ebf4f21cfc63bc`
  - diagnosis `b3948b8d43eb81331fcd954e78c8545d4bc9a04b48762591377cd94f47697e20`
  - split `5afa11096c3988d8aaefa5164c053c873f718e730fbb286d482f516b01f05c52`
- 这些是本地观察值，不应误称 publisher-signed hashes。

### 五个 package

| 方法 | 当前本地证据 | clean-release 结论 |
|---|---|---|
| Patch-Mix CL | environment、adapter、smoke/profile helper、CUDA full command、gate report 存在；本机 CPU gate 已记录 | 候选优先级 1；仍需 fresh clone 重建 source/checkpoint/portable run/receipts 和 L40 smoke/profile；test-selected |
| PAFA | environment、adapter、checkpoint inspector、smoke/profile helper、CUDA full command、gate report 存在 | 候选优先级 2；BEATs official identity 未证实时只能 official-like；需 fresh clone + L40 gate |
| SG-SCL | 通用与 Linux CUDA 11.8 env、adapter、smoke、CUDA command、gate report 存在 | 候选优先级 3；需 fresh clone + L40 gate；metadata-aware，不是 audio-only |
| MVST | 通用与 Linux CUDA 11.8 env、五视图工具、fusion/export、CUDA pipeline、gate report 存在 | 本地 package 宣布 ready 后再排队；author random-file split，不是 official split；五视图 ID/order 是硬 gate |
| ADD-RSC | README、environment、notebook、source-data gate 存在 | `not_packaged`：无 data adapter、CUDA full launcher、eval/export 和独立 server gate report；两种矛盾 protocol 未形成唯一 faithful route |

共同问题：五个 `run_reproduction.py` 当前都只接受 `--gate source-data`。`official_reproduction_receipts.py` 写死 `result/*_20260721_171405`，而 result-local source/checkpoint/portable run 不会经 Git 同步。第一个 release 不能依赖这些本机目录。

### 环境

- Patch-Mix CL：`baseline/patch_mix_cl/environment.yml`，当前 SHA256 `f109aeb61c4adcd98719aa7b439623bd4e1758daf53c7f32f52f3cb81ad1dd84`。
- PAFA：`baseline/pafa/environment.yml`，当前 SHA256 `f6a24604f3c83d883809fa1e95d5f715638b1b3a495c507d80a7ad7bc4d1dacf`。
- SG-SCL L40：`baseline/sg_scl/environment.linux-cu118.yml`，当前 SHA256 `902017d77000d04d913d594a2ddb8fcfea8b3c765e5a5cdfe756e188e1923f9f`。
- MVST L40：`baseline/mvst/environment.linux-cu118.yml`，当前 SHA256 `63544c410ed7b4785ccdd4a03473a6cdff72521aea499cf7e57df297c3cf4864`。
- ADD-RSC：`baseline/add_rsc/environment.yml`，当前 SHA256 `118d05a9180ecc8068fd8c1b2c62574f726c2d0fb343a330c60991a4478a11b8`。

以上 hashes 来自仍在变化的未跟踪工作树，只能做 audit snapshot；release manifest 必须在 release commit 冻结后重新计算。

## 已创建的 server-owned artifacts

- `server/server_training_deployment_runbook_zh.md`
- `server/server_training_vscode_codex_instructions_en.md`
- `server/server_training_release_manifest.template.yaml`
- `server/server_training_run_receipt.schema.json`
- `codex/2026-07-21/server_training_local_audit_and_management_handoff_2026-07-21.md`

这些文件没有真实 password/token/server path；所有远程信息均保留为明确占位符。

## 仍缺的服务器信息

在任何服务器动作前，需要用户或管理线程提供：

1. 可 fetch 的 Git remote URL，以及 Release 1 的固定 40 位 SHA、release ID、manifest path/hash。
2. 服务器 host/登录方式（无需把密码或 token 发到文档）、repo release root、data root、result root、conda root、结果回传目标。
3. 一张或两张 L40；GPU 是否独占；GPU UUID/资源配额；允许的最长 wall time。
4. 使用 tmux 还是 Slurm；若 Slurm，提供 partition、account、QOS、GRES 名称、CPU/RAM/time limits。
5. 网络限制、proxy/allowlist，以及 GitHub、ICBHI、conda/PyPI、approved checkpoint/source hosts 的可达性。
6. checkpoint 的批准获取方式和 expected size/SHA256；PAFA compatibility mirror 是否获准做 official-like run。
7. data/result/conda 磁盘配额和本地 result archive 回传路径。
8. 是否先只发布 Patch-Mix，还是 Patch-Mix+PAFA+SG-SCL 同一 release；建议以实际 fresh-checkout gate 通过顺序决定，不等待 MVST/ADD-RSC。

## 推荐何时创建第一个 release commit

不要现在创建。满足以下条件后立即创建 Release 1；不需要等待五个方法全部 ready：

1. 管理线程已从 `origin/main` 建立干净 release branch，并确认 release 增量 object 中不存在 `51131f...`、该 1.307 GB path 或任何未批准的 `>=90 MB` blob。
2. 至少 Patch-Mix CL 在完全临时、干净 clone 中能从 release-tracked instructions 重建固定上游 source、checkpoint、adapter、receipts；不引用本机 timestamped result。
3. environment、dataset script、paper contract、checkpoint provenance 和所有 launch/eval commands 都进入 manifest 并有 hash。
4. smoke → 100-step → full → evaluation/export 命令均明确；L40 上未执行的阶段标为 pending，不能把 CPU/local gate冒充 server gate。
5. `result/`、raw、feature、checkpoint、cache 均不在 Git；release object size audit 和 `git diff --check` 通过。
6. 从远端重新 clone release branch，checkout 候选 SHA，HEAD/status/remote/manifest 验证通过。
7. 管理线程 review 并显式把至少一个 method status 设为 `ready`、`server_execution_authorized: true`，然后 push 该 release branch。

如果 Patch-Mix 先通过，就先发布 Patch-Mix-only Release 1；PAFA/SG-SCL 随后用新 SHA 发布 Release 2。服务器每次记录原 SHA，旧 receipt 不覆盖。

## 给管理线程的当前一句话状态

Server-side 文档与 receipt/manifest 契约已准备；当前 `main` 受 1.307 GB 历史 blob 阻塞且 baseline fresh-checkout rebuildability 仍在修复，短期应从 `origin/main` 建立干净 `codex/l40-training-release` 并在至少 Patch-Mix 的 clean-clone gate 通过后再创建、push 和授权 Release 1，现阶段不得启动远程训练。
