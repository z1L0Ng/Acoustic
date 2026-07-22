# Acoustic L40 训练部署与运行手册

版本：2026-07-21（第一阶段；仅 ICBHI 2017）

适用方法：Patch-Mix CL、PAFA、SG-SCL、MVST、ADD-RSC

角色边界：本地仓库负责方法实现、adapter、数据脚本与修复；服务器只负责固定版本的下载、环境构建、训练、评估、监控与结果导出。

## 1. 硬性原则

1. 服务器只能运行管理线程发布的固定 commit SHA 和同一 release manifest。当前本地 working tree、未提交文件、聊天中的代码片段都不是发布依据。
2. 方法代码、模型代码、adapter、数据脚本或修复必须先在本地完成 review，再由管理线程 commit/push。服务器 Codex 不得自行修改代码。
3. 每次运行使用新的、干净的 release checkout；运行期间不得 `pull`、切 branch 或改变 checkout。新修复使用新 SHA 和新 run ID，旧 receipt 永不覆盖。
4. `result/` 保存重型结果并保持 gitignored。GitHub 只同步代码、环境声明、release manifest 和轻量 receipt/report。
5. 所有 active generated output 使用 `result/<baseline>_<YYYYMMDD_HHMMSS>/`；`<baseline>` 必须是 `patch_mix_cl`、`pafa`、`sg_scl`、`mvst` 或 `add_rsc`。协议、seed 和 release SHA 写入 receipt，不另建第二个 output root。绝不创建 legacy `results/`。
6. compatibility checkpoint 可以支持 `official-like` 运行，但必须准确写明 provenance、hash 和 claim boundary；不得称为原始作者 checkpoint 或 faithful result。
7. 目标是按作者 repo/protocol 得到合理范围结果，不承诺精确复现 paper aggregate mean。必须同时记录 paper target、实际 protocol、test-selection、checkpoint identity 和 gap。
8. 第一优先级为 Patch-Mix CL、PAFA、SG-SCL；MVST、ADD-RSC 只在对应 method status 为 `ready` 后进入队列。不承诺 MVST 在次日早晨前完成。

## 2. 当前本地审计结论（只读快照）

审计日期为 2026-07-21。该快照只能用于决定 release gate，不能直接部署。

- branch：`main`。
- 本地 HEAD：`6400127d65287dd0bd7150042892769e48881976`。
- `origin/main`：`b9180791c83e2633b6c4ea9d0cac5a0455ed553e`。
- `origin/main` 已 fetch；本地 `main` 相对 `origin/main` 为线性 ahead 58、behind 0。
- 首次 push 已安全终止，未继续重试。待推历史包含单个 1,307,407,924-byte tracked blob；它在迁移前的 legacy Git path 是 `results/model_design_2026-07-08/icbhi_sequence_pooling/features/beats_sequence_full.npz`。blob `51131f763da6d7de7b3b2780c8db61f92250348a`，由 commit `8c630c8` 引入。GitHub 无法接受该单文件。当前本地副本已迁入 canonical `result/`，但目录迁移不会改变历史 blob。除它外，`origin/main..main` 未发现 `>=90 MB` blob。
- 五个新 baseline 目录、common reproduction helpers 和 `codex/2026-07-21/` 当前均为未跟踪并行改动；服务器从任何现有 Git SHA 都拿不到这些文件。
- `result/` 已由 `.gitignore` 忽略；`dataset/raw/**` 也已忽略。
- 五个 `run_reproduction.py` 当前都只支持 `--gate source-data`，不是 smoke/full launcher。
- common receipt 代码当前绑定本机时间戳 `result/*_20260721_171405`；这些 result-local 文件不会经 Git 到服务器。
- Patch-Mix CL、PAFA、SG-SCL、MVST 有 CUDA full shell 命令；ADD-RSC 暂无 CUDA full command、data adapter 或独立 server gate report。
- SG-SCL 和 MVST 有 `environment.linux-cu118.yml`；Patch-Mix CL、PAFA、ADD-RSC 只有通用 `environment.yml`，需在 L40 上实测 CUDA wheel、driver compatibility 和 kernel。
- 现有 ICBHI 下载入口是 `dataset/script/phase1_dataset_workflow.py icbhi_2017`；notebook 只是调用该脚本。脚本对五个官方文件校验已知 size，但没有内置 publisher SHA256，并对 ICBHI URL 使用未验证 TLS context。服务器生成的 SHA256 必须再与 release manifest 中管理线程批准的 expected hashes 比对。

因此，Release 1 **尚未 ready**。第一个 release commit 前必须先解决可推送 release 分支，再通过“从临时干净 clone 重建 server-local source/checkpoint adapter/receipts，并完成命令级 smoke”的 gate。不得把当前本机 `result/` 中已通过的 smoke/profile 当作 fresh checkout 已通过。

## 3. 发布生命周期

### 3.1 本地 review、commit、push（管理线程执行）

Baseline Reproduction 线程交付某方法后，管理线程完成：

1. 审核该方法目录、共享 common 文件、paper contract、environment 和 launch script。
2. 从临时干净 clone 验证：下载入口可调用；上游源码可按固定 commit 获取；checkpoint 可按批准 provenance 获取并验 hash；adapter 和 receipt 可在无本机 `result/` 的条件下重建；smoke/100-step/full 命令不引用本机绝对路径或写死时间戳。
3. 校验 `result/`、raw data、checkpoint、cache 未被纳入 Git。
4. 生成 `server_training_release_manifest.yaml` 的实例，逐方法设置 `ready`、`blocked` 或 `not_packaged`，填入全部 SHA256 和准确 claim。
5. 由管理线程统一 stage/commit/push。release commit SHA 必须是 40 位完整 SHA，且在远端可 fetch。

建议一个方法通过 server-runnable gate 后就创建一个明确 release；也可以合并多个 ready 方法，但 manifest 必须逐方法列状态。manifest 中 `ready` 不能仅表示本机 smoke 通过，而必须表示 fresh-checkout 可重建。

### 3.1.1 当前 GitHub 大文件阻塞与发布分支策略

不得直接重试推送当前 `main`：其未推送历史包含 GitHub 无法接收的 1.307 GB derived feature blob。也不得在未获得用户明确授权时重写 `main`。

| 方案 | 做法 | 优点 | 风险/代价 | 建议 |
|---|---|---|---|---|
| A. 干净 release 分支 | 从已 fetch 的 `origin/main` 创建 `codex/l40-training-release`，只把当前必要的 baseline/common/server/codex 轻量代码和声明作为新提交带入；显式排除 canonical `result/`、任何 legacy output tree、raw data、feature、checkpoint；验证新分支增量对象无大 blob后推送 | 不重写共享 `main`；最快解除服务器发布阻塞；release SHA 清晰 | 需要维护 release 分支与本地开发线的受控同步；不能整段 cherry-pick 含大 blob 的历史 | **短期推荐** |
| B. 重写 `main` 历史 | 经用户批准后，用历史清理工具从所有相关 commits 移除该 blob，再协调 force push/reclone | 长期统一主线，彻底移除不可推对象 | destructive；改变 58 个本地 commits 的 SHA；影响协作者、worktree、引用和 receipts；必须有备份、停写窗口和迁移计划 | 仅在用户明确批准、管理线程制定迁移方案后执行 |

方案 A 的关键安全点：

1. release branch 必须从 `origin/main` 出发，不能从当前 `main` 直接建分支后推送，也不能 cherry-pick `8c630c8` 或任何会携带该 blob 的聚合 commit。
2. 通过经过 review 的 path-level copy/patch 把必要文件形成新的、轻量 commit；禁止带入完整 feature/result。
3. 推送前枚举 `origin/main..codex/l40-training-release` 的所有 Git objects，要求不存在该 `51131f...` blob，且不存在未批准的 `>=90 MB` blob。
4. 在临时目录重新 clone 该远程 release branch，固定到候选 SHA，完成 fresh-checkout rebuildability gate。
5. 服务器只 checkout 管理线程宣布的 release branch 上固定 40 位 SHA；branch 名只用于 fetch，运行身份仍是 SHA + manifest hash。

方案 B 不是当前 server rollout 的前置条件。若以后执行，必须由用户显式批准，并由管理线程负责备份、受影响引用盘点、filter/rewrite 验证、force-push 协调和所有消费者 reclone；Server Training Coordinator 不执行这些操作。

候选 release commit 生成后，管理线程可用以下只读检查生成 object audit（命令本身不创建分支、不改历史、不 push）：

```bash
export RELEASE_SHA="<40_HEX_RELEASE_COMMIT_SHA>"
export BLOCKED_BLOB="51131f763da6d7de7b3b2780c8db61f92250348a"

git rev-list --objects "origin/main..${RELEASE_SHA}" \
  | git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)' \
  | awk '$2 == "blob" && $3 >= 90000000 {print}'

test -z "$(git rev-list --objects "origin/main..${RELEASE_SHA}" | awk -v blocked="${BLOCKED_BLOB}" '$1 == blocked {print}')"
```

第一条若输出任何行，都需要逐项审批；第二条必须成功。再将完整输出、检查时间和 release SHA 写入轻量 object audit report。

### 3.2 服务器固定 SHA checkout

优先使用每个 release 一个新目录，避免对旧 run 做 destructive reset。所有路径均由用户提供，不得猜测。

```bash
export ACOUSTIC_REMOTE_URL="<ACOUSTIC_GIT_REMOTE_URL>"
export RELEASE_SHA="<40_HEX_RELEASE_COMMIT_SHA>"
export RELEASE_ID="<RELEASE_ID>"
export SERVER_RELEASE_ROOT="<SERVER_RELEASE_ROOT>"
export RELEASE_DIR="${SERVER_RELEASE_ROOT}/${RELEASE_ID}"

test ! -e "${RELEASE_DIR}"
git clone --no-checkout "${ACOUSTIC_REMOTE_URL}" "${RELEASE_DIR}"
git -C "${RELEASE_DIR}" fetch --prune origin
git -C "${RELEASE_DIR}" cat-file -e "${RELEASE_SHA}^{commit}"
git -C "${RELEASE_DIR}" checkout --detach "${RELEASE_SHA}"
test "$(git -C "${RELEASE_DIR}" rev-parse HEAD)" = "${RELEASE_SHA}"
test -z "$(git -C "${RELEASE_DIR}" status --porcelain)"
git -C "${RELEASE_DIR}" remote get-url origin
```

将以下内容写入 bootstrap receipt：release ID、requested SHA、observed HEAD、branch/detached state、status porcelain、remote URL、checkout time、host。任何 mismatch 都停止，不运行训练。

## 4. L40/CUDA/driver/disk/network preflight

在创建环境或下载大文件前执行，并把完整输出保存到 `<RUN_ROOT>/receipts/preflight/`：

```bash
hostname
uname -a
date --iso-8601=seconds
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.free,temperature.gpu,pstate --format=csv
nvcc --version || true
df -h "<DATA_ROOT>" "<RESULT_ROOT>" "<CONDA_ROOT>"
df -ih "<DATA_ROOT>" "<RESULT_ROOT>" "<CONDA_ROOT>"
free -h
ulimit -a
conda --version
git --version
curl --version
```

通过条件：

- 识别到一张或两张 NVIDIA L40；记录 GPU UUID，不只记录 index。
- driver 能支持 environment 声明的 CUDA runtime；以安装后 `torch.version.cuda` 和真实 GPU kernel 为最终检查。
- 每张计划使用的 GPU 在启动时有足够显存，没有未知进程占用。100-step profile 决定该方法的实际显存门槛。
- `<DATA_ROOT>`、`<RESULT_ROOT>`、`<CONDA_ROOT>` 有经用户确认的空间预算和 inode 余量。MVST 有五个 encoder 和 feature archive，须单独留出更大预算。
- 能访问 Git remote、ICBHI 官方下载 host、conda channels、PyPI，以及 manifest 中批准的 checkpoint/source URL。只做 HEAD/小请求时也要记录 HTTP status；不得打印 token。
- NTP/系统时间正常。所有 receipt 同时写 UTC 和本地 timezone。

环境建立后，每个 env 必须执行 GPU check：

```bash
conda run -n "<ENV_NAME>" python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x=torch.randn(1024,1024,device="cuda"); print(float((x@x).mean()))'
```

若 `torch.cuda.is_available()` 为 false、kernel 失败或 device 不是预期 L40，停止该 env，不以 CPU full run 替代。

## 5. ICBHI 下载与验证

### 5.1 下载

只运行 release 中固定的数据脚本：

```bash
cd "<RELEASE_DIR>"
test "$(sha256sum dataset/script/phase1_dataset_workflow.py | awk '{print $1}')" = "<DATASET_SCRIPT_SHA256>"
conda run -n "<BOOTSTRAP_ENV>" python dataset/script/phase1_dataset_workflow.py icbhi_2017
```

脚本会写入 `dataset/raw/icbhi_2017/`，并在 `dataset/processed/icbhi_2017/` 生成 inventory 和 checksums。raw data 不得被方法脚本就地修改；adapter 只可使用只读 symlink 或在 result root 内建立派生缓存。

### 5.2 完整性和 inventory

```bash
cd "<RELEASE_DIR>/dataset/raw/icbhi_2017"
sha256sum -c ../../processed/icbhi_2017/checksums_sha256.txt

cd "<RELEASE_DIR>"
test "$(find dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database -maxdepth 1 -type f -name '*.wav' | wc -l | tr -d ' ')" = "920"
test "$(find dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database -maxdepth 1 -type f -name '*.txt' | wc -l | tr -d ' ')" = "922"
test "$(wc -l < result/baseline_overnight_2026-07-07/icbhi_2017/icbhi_2017_cycle_manifest.csv | tr -d ' ')" = "6899"
```

随后把五个顶层官方文件的 observed SHA256 与 release manifest 的 `dataset.expected_artifacts` 逐项比较。当前本地观察值可以作为管理线程候选证据，但不是 publisher 签名：

- `ICBHI_final_database.zip`: `e34e9913c43ee9da62af4faa7335d8eda7ce9883c315b69b07cfcbde28650e4c`
- `events.zip`: `d13b5efdcf36635e09731607664f3eea4185b25b2606a742d01be398829d15e5`
- demographic: `7be825ef14639a2dc9f033139158ddd2d7368c26fb6c871fa7ebf4f21cfc63bc`
- diagnosis: `b3948b8d43eb81331fcd954e78c8545d4bc9a04b48762591377cd94f47697e20`
- official split: `5afa11096c3988d8aaefa5164c053c873f718e730fbb286d482f516b01f05c52`

最终 data acceptance 还必须有 6,898 个 unique cycle IDs、official train/test 4,142/2,756、920 recordings，并保留已知 filename alias receipt。任一 count/hash 不符时隔离该下载目录，提交 failure report；不得静默修文件名或继续训练。

## 6. 五个 conda env 的 build/check

每个方法独立环境；不复用 base 环境，不在运行中更新包。使用 manifest 指定的 environment file 和 hash：

```bash
cd "<RELEASE_DIR>"
test "$(sha256sum "<ENVIRONMENT_FILE>" | awk '{print $1}')" = "<ENVIRONMENT_SHA256>"
conda env create -f "<ENVIRONMENT_FILE>"
conda run -n "<ENV_NAME>" python -V
conda run -n "<ENV_NAME>" python -m pip check
conda list -n "<ENV_NAME>" --explicit > "<RUN_ROOT>/receipts/environment/<METHOD>.conda-explicit.txt"
conda run -n "<ENV_NAME>" python -m pip freeze > "<RUN_ROOT>/receipts/environment/<METHOD>.pip-freeze.txt"
```

推荐 spec：

| 方法 | env name | release 应优先指向 | 当前限制 |
|---|---|---|---|
| Patch-Mix CL | `acoustic-patchmix` | `baseline/patch_mix_cl/environment.yml` | 需在 L40 验证实际 CUDA wheel/kernel |
| PAFA | `acoustic-pafa` | `baseline/pafa/environment.yml` | 同上；checkpoint identity 仍决定 claim |
| SG-SCL | `acoustic-sgscl` | `baseline/sg_scl/environment.linux-cu118.yml` | metadata-aware；不是 audio-only |
| MVST | `acoustic-mvst` | `baseline/mvst/environment.linux-cu118.yml` | 五视图；资源和时长更大 |
| ADD-RSC | `acoustic-addrsc` | `baseline/add_rsc/environment.yml` | 尚无 full launcher；只 build/check，不进入 full queue |

每个 env 的 acceptance：environment hash 匹配；创建成功；`pip check` 成功；核心 import 成功；真实 L40 kernel 成功；smoke 所需的 author repo/import 成功。若 lock/solver 结果不同，记录差异，不直接手改 environment 文件。

## 7. 方法运行状态和 claim

manifest 的 method status 是唯一调度依据：

- `ready`: fresh-checkout rebuild、checkpoint gate、smoke launcher、100-step launcher、full/eval/export 命令全部明确。
- `blocked`: 已知 blocker；只能做 manifest 明确允许的 preflight/build，不得 full run。
- `not_packaged`: 没有可执行 server package。
- `retired`: 保留历史 receipt，不再调度。

当前候选状态建议：

| 方法 | 候选排序 | release 前必须解决/确认 |
|---|---:|---|
| Patch-Mix CL | 1 | 干净 clone 重建 portable run；current author-hosted checkpoint hash；GPU smoke/profile；test-selected 标签 |
| PAFA | 2 | 干净 clone 重建；BEATs checkpoint provenance/hash；若为 mirror，只能 official-like；GPU smoke/profile |
| SG-SCL | 3 | 干净 clone 重建；device metadata 完整；GPU smoke/profile；not audio-only 标签 |
| MVST | 4 | package ready 后；五视图 ID/order gate；random-file split 标签；encoder resume 限制 |
| ADD-RSC | 5 | 先补 full launcher/adapter/eval/export；两种矛盾 protocol 分开，不能称 faithful |

## 8. 分阶段执行：smoke → 100-step → full → evaluation/export

每次方法执行使用 `RUN_ID=<method>_<YYYYMMDD_HHMMSS>`，对应唯一目录 `result/<method>_<YYYYMMDD_HHMMSS>/`。同一次运行的 smoke、profile、full 和 evaluation 都写入该目录；protocol、seed、release SHA 和 UTC/local timezone 写入 receipt。每阶段成功后写 milestone receipt；失败也写 receipt。

### 8.1 Smoke

- 只用 manifest 固定的小样本/8 cycles。
- 覆盖所有 4 类；需要 metadata 的方法覆盖 metadata domain。
- 验证 preprocessing shape、forward、loss、backward、optimizer step、finite values、cycle ID/export wiring。
- GPU smoke 必须实际在分配的 L40 上执行；记录 GPU UUID、peak VRAM、wall time。
- smoke metric 只能称 wiring result，不能称 numerical result。

### 8.2 100-step GPU profile

- 使用 full run 相同 batch size、precision、optimizer 和 model path，运行 100 个训练 step。
- 记录 warmup、mean/p50/p95 step time、peak allocated/reserved VRAM、host RSS、GPU utilization、temperature、power、loss finite/convergence warning。
- 用 observed throughput 估计 full run 时间和磁盘；不从本地 CPU profile 外推 L40。
- OOM 时只执行 manifest 预先批准的 fallback。改变 batch、gradient accumulation 或 precision 会改变 protocol，必须回本地审批，不能由服务器临时决定。

### 8.3 Full run

- 只有 smoke 和 100-step profile 都通过才启动。
- 启动前再次记录 HEAD/status/remote、manifest hash、dataset hash、env receipt、checkpoint hash、GPU UUID。
- 锁定 release directory，创建 `<RUN_ROOT>/RUNNING` marker；存在 marker 时禁止任何 Git 操作。
- 按 manifest 原样执行 launch command；stdout/stderr 使用 `tee` 保存，保留 shell exit code。

### 8.4 Evaluation/export

- 导出 checkpoint、per-cycle IDs/labels/logits/probabilities、confusion matrix、Sp、Se、Score、best epoch 和 selection trace。
- official split 方法要求 2,756 个 unique official-test cycle IDs、confusion total 2,756，repo metric 与 prediction-derived metric 在声明 tolerance 内一致。
- MVST 依据 author random-file split，expected test cycles 为 2,685；不得套用 official 2,756 gate。
- 记录 paper target 和 observed gap。若 paper 有 SD，项目内部 aligned gate 为对应 metric 的 `mean ± 2 SD`；无 SD 时暂用绝对 Score gap `<= 3`。
- aligned gate 不是“成功复现论文 aggregate mean”的声明。超出 gate 时如实记为 gap，回本地 audit；不在服务器扩大 grid 或修改方法。

## 9. 一张与两张 L40 的队列

### 一张 L40

严格串行：

1. Patch-Mix CL
2. PAFA
3. SG-SCL
4. MVST（仅 ready 后）
5. ADD-RSC（仅 ready 后）

同一时间只有一个 full/profile job 占 GPU。环境 build、checksum、轻量 export 可在不干扰训练 I/O/CPU 的条件下安排；若会争用磁盘或内存则等待。

### 两张 L40

- GPU0：Patch-Mix CL。
- GPU1：PAFA。
- 任一 GPU 空闲且对应前序 receipt 完整后，调度 SG-SCL。
- MVST 和 ADD-RSC 等 package ready，再进入首个空闲 GPU；MVST 五个 encoder 默认在单卡顺序运行，除非 manifest 明确批准多卡策略。
- 不把两个独立训练进程放到同一 GPU，也不由服务器自行改成 DDP。

## 10. tmux 与 Slurm 启动

### tmux

```bash
export METHOD="<METHOD>"
export RUN_TIMESTAMP="<YYYYMMDD_HHMMSS>"
export RESULT_ROOT="${RELEASE_DIR}/result"
export RUN_ID="${METHOD}_${RUN_TIMESTAMP}"
export RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
export GPU_INDEX="<GPU_INDEX>"
test "$(basename "${RESULT_ROOT}")" = "result"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/receipts"

tmux new-session -d -s "${RUN_ID}" \
  "cd '<RELEASE_DIR>' && CUDA_VISIBLE_DEVICES='${GPU_INDEX}' bash -lc '<MANIFEST_LAUNCH_COMMAND>' 2>&1 | tee '${RUN_ROOT}/logs/full.log'"
tmux list-sessions
tmux capture-pane -pt "${RUN_ID}" -S -200
```

服务器 Codex 必须单独记录真实退出码；若使用 pipeline，launcher 应包含 `set -o pipefail`。不要只以 tmux session 消失判断成功。

### Slurm

```bash
sbatch \
  --job-name="<RUN_ID>" \
  --gres=gpu:l40:1 \
  --cpus-per-task="<CPUS>" \
  --mem="<RAM>" \
  --time="<WALLTIME>" \
  --output="<RUN_ROOT>/logs/slurm-%j.out" \
  --error="<RUN_ROOT>/logs/slurm-%j.err" \
  --export=ALL,RELEASE_DIR="<RELEASE_DIR>",RUN_ROOT="<RUN_ROOT>",METHOD="<METHOD>" \
  "<RELEASE_DIR>/server/<APPROVED_SLURM_WRAPPER>"
```

Slurm partition/account/QOS/GRES 名称必须由用户提供。不得猜 `gpu:l40:1` 在目标集群有效；提交前先用集群只读命令确认并替换占位值。

## 11. Resume 与 failure handling

1. 首先停止自动重试，保留 log、partial checkpoint、exit code、`nvidia-smi` snapshot 和 receipt。
2. 分类：infrastructure、environment、data/checksum、checkpoint identity、OOM、NaN/numerical、upstream code、adapter/export、scheduler/preemption。
3. 只有 manifest 明确声明 `resume_supported: true` 且 server-side resume integration test 已通过，才可从 checkpoint resume。receipt 必须记录 parent run ID、checkpoint path/hash、原始 step/epoch 和新启动时间。
4. SG-SCL upstream resume 未完整恢复 classifier/projector；未通过 server integration test 前，中断后重启。MVST fusion 可按批准命令 resume；encoder resume 不支持，中断 encoder 重启该 encoder。
5. OOM、shape、missing key、数据 count、metric mismatch 等代码问题不得在服务器修复。输出最小 failure report，交给 Server Training Coordinator → 管理线程 → 本地 Baseline Reproduction 线程。
6. 本地修复产生新 release SHA。旧 run 标记 `failed` 或 `aborted` 并保留；新 run 使用新 ID，不覆盖。

最小 failure report：release SHA/manifest hash、method/protocol/seed/stage、exact command（脱敏）、env/checkpoint/dataset hashes、GPU UUID、exit code、最后 100–200 行 log、最小 traceback、已完成 checks、partial artifact 清单、是否可安全 resume、需要本地回答的一个具体问题。

## 12. Result 命名与 artifact list

目录规范：

```text
result/
  <method>_<YYYYMMDD_HHMMSS>/
    RUNNING
    server_run_receipt.json
    receipts/
      preflight/
      git_checkout.json
      dataset.json
      environment/
      checkpoint.json
      smoke.json
      profile_100.json
      evaluation.json
      artifact_inventory.sha256
    logs/
    checkpoints/
    predictions/
    metrics/
    exports/
```

`result/<method>_<YYYYMMDD_HHMMSS>/` 是唯一 active run root。不得生成或恢复 `results/`。同一秒发生命名冲突时停止并取得新的 timestamp，不能覆盖旧目录。

`server_run_receipt.json` 必须符合 `server/server_training_run_receipt.schema.json`，至少包含：run/release identity、Git remote/HEAD/status、manifest hash、host/GPU/driver/CUDA、environment path/hash/locks、dataset script/artifact/manifest hashes、checkpoint provenance/hash、method/protocol/selection/seed、exact commands、阶段状态/时间/退出码、resume lineage、paper target、observed metrics/gap/alignment、claim boundary 和 artifact inventory。

成功结束后删除 `RUNNING` marker，写最终 receipt，再对所有 artifact 生成 SHA256：

```bash
cd "<RUN_ROOT>"
find . -type f ! -name artifact_inventory.sha256 -print0 | sort -z | xargs -0 sha256sum > receipts/artifact_inventory.sha256
```

## 13. 结果回传

重型结果通过 rsync/scp 或用户指定存储回传，不进入 Git。先 dry-run：

```bash
rsync -avhn --partial --append-verify \
  "<SERVER_USER>@<SERVER_HOST>:<REMOTE_RESULT_ROOT>/<RUN_ID>/" \
  "<LOCAL_RESULT_ARCHIVE_ROOT>/<RUN_ID>/"
```

用户确认目标后执行：

```bash
rsync -avh --partial --append-verify --info=progress2 \
  "<SERVER_USER>@<SERVER_HOST>:<REMOTE_RESULT_ROOT>/<RUN_ID>/" \
  "<LOCAL_RESULT_ARCHIVE_ROOT>/<RUN_ID>/"
```

回传后在本地执行 `sha256sum -c receipts/artifact_inventory.sha256`。若只能用 scp：

```bash
scp -r "<SERVER_USER>@<SERVER_HOST>:<REMOTE_RESULT_ROOT>/<RUN_ID>" "<LOCAL_RESULT_ARCHIVE_ROOT>/"
```

不得在命令、receipt 或聊天中写真实密码/token。SSH key、host alias 和 storage credentials 由用户在服务器环境配置。

## 14. No-code-edit 与 escalation policy

服务器 Codex允许：读取固定 checkout；运行 manifest 命令；创建 result-local env/cache/symlink/log/receipt；监控；导出；回传用户批准的结果。

服务器 Codex禁止：编辑 tracked code；patch 上游 source；改变数据标签/split；更换 checkpoint 而不更新 manifest；临时改变 hyperparameter；`git commit/push/pull`；在运行中切 SHA；把重型 result 加入 Git；独立扩大实验 grid。

发现代码问题时：

1. 保持失败现场只读并写 failure report。
2. 暂停该方法队列；其他独立且 ready 的方法可继续。
3. 报告 Server Training Coordinator。
4. Coordinator 只做分流和汇报，不在服务器修代码。
5. 管理线程安排本地 Baseline Reproduction 线程修复、review、commit/push，并发布新 SHA/manifest。

## 15. Milestone 汇报节点

每个节点用简洁中文向管理线程汇报，并附 receipt path/hash：

1. 固定 SHA checkout 通过。
2. L40/CUDA/disk/network preflight 通过或 blocker。
3. ICBHI download/hash/inventory/cycle gate 通过。
4. 五个 env build/check 矩阵完成。
5. 每个方法 smoke 完成。
6. 每个方法 100-step profile 完成并给出 full ETA/VRAM。
7. full run 启动、关键 epoch/异常、完成。
8. evaluation/export/rsync 完成；报告 paper target、observed result、gap、aligned gate 和准确 claim。
