# LSAA归因消融：Native-only与without-PAFA

状态：**CODE READY，正式服务器运行已由用户批准；本实现线程未运行模型。** 两组均为test-selected三seed诊断，不是clean validation-only evidence。

## 冻结共同合同

- seeds：0/1/42；mono 16 kHz；5 s repeat-pad/front-truncate；无SpecAugment；
- BEATs iter3+ AS2M全量微调；batch 32；Adam lr `5e-5`、wd `1e-6`；cosine `eta_min_ratio=1e-3`；EMA beta 0.5；
- 最多50 epochs、patience 10、strict improvement、tie保留更早checkpoint；
- ICBHI official-test native Score逐epoch选模；原source-internal validation只用于既有C/W threshold；SPRSound terminal在selected checkpoint后读取；
- evidence label保留test-selected边界；历史Full与Native+attributes参照实际使用MPS/FP32，新消融使用CUDA/FP32并显式关闭AMP，不承诺跨后端逐位一致；
- 明确不使用PC-MCL的`lr=1e-4`、400 epochs或SpecAugment。

服务器相对`--repo-root /files1/Zilong/Acoustic`解析全部依赖：

- author source：`result/pafa_sprsound_transfer_20260722_235659/source/repo`；
- initial checkpoint：`.cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt`；
- ICBHI/SPR原始数据与现有manifest/split：仓库既有相对路径；
- Full参照：`result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/seed_{0,1,42}`；
- Native+attributes参照：既有`PAFA_BENCHMARK_4COND_* / native_attributes / seed_*`。

服务器不依赖上述历史result目录。三seed原始group→partition映射与预期sample/group支持量已版本化为`baseline/pafa/lsaa_attribution_source_splits.json`：Native+attributes映射已与正式Full每seed的epoch-1 validation sample IDs逐项相等核对。新run从canonical source metadata载入实际sample IDs后直接套用该映射，在模型构造前核对支持量，并在每个新seed目录写出完整`split_reference.json`；不调用sklearn重新划分。各seed ICBHI subtrain/validation分别为3636/506、2880/1262、3174/968，SPR均为5219/1437。

BEATs wrapper通过版本化的直接模块loader读取作者`models/beats.py`，不import作者`models/__init__.py`，避免无关AST/wget依赖；ProjectionHead/PAFALoss仍来自原作者PAFA模块，模型类与state keys不变。不提交原始数据、checkpoint或历史结果。

## Variant 1：Native-only

实现入口复用`table2_benchmark_controls.py`的Native+attributes模型构造。BEATs、PAFA projector、shared projector、ICBHI flat4 head、SPR binary head以及C/W heads均按相同顺序实例化；C/W heads不删除、不参与loss，也不用于报告。

每个homogeneous dataset batch：

```text
L_native_only = (1/3) * CE(native head) + PAFA(PCSL lambda=50, GPAL lambda=5e-4)
```

关闭C/W项后不重归一化。输出只报告ICBHI flat4与SPR Task1-1 native metrics、支持量、混淆和逐样本native NPZ。属性threshold、SPR C/W AUROC与HF均写为`not_applicable`，而不是0。

输出：`result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/native_only/seed_*`。

## Variant 2：LSAA without PAFA

实现入口复用`joint_hierarchy_main_multiseed.py`的正式Full模型、A/C/W分类、mask、batch、EMA、选模、threshold与native readout。模型仍实例化PAFA projector以保持架构和初始化顺序，但该mode直接跳过PAFA criterion调用：

```text
L_without_PAFA = 1.0 * L_hierarchical_classification
PCSL/GPAL criterion called = false
```

没有`0 * PAFA(...)`路径。分类项继续直接调用正式Full的`beats_nal_protocol.hierarchical_loss`，按实际eligible节点取mean；不替换成benchmark controls的固定逐节点`1/3`聚合。without-PAFA的每个total loss在backward前检查finite，异常立即终止。

selected checkpoint后输出ICBHI/SPR native结果及SPR C/W AUROC，再复用现行论文表格的固定外评：HF以每条15 s recording三个5 s窗口的`max(p_W)`为score，在任一D/Wheeze/Rhonchi/Stridor标注的957条eligible recording上计算CAS AUROC（Wheeze/Rhonchi/Stridor positive 661，D-only negative 296）；KAUH以B/D/E probability mean后的86位compatible patient Level1 balanced accuracy为主列。D/W单属性指标仅作secondary diagnostic，不能改名为CAS。HF/KAUH不参与训练、选模或threshold。每seed保存best与last，CAS与KAUH主终点成功后才把总状态写为complete；external失败时training summary保持`external_pending`，三seed汇总拒绝非complete或非有限指标。原Full默认的`early_stopped_epochwise_icbhi_test_selected`状态字符串保持不变。

输出：`result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/lsaa_without_pafa/seed_*`。

## 两张GPU持久队列

两个队列相互独立，各自在一张GPU按0→1→42执行；任一seed报错时对应队列停止，不自动重试或调参，另一队列不受影响。

```bash
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n acoustic-addrsc python \
  -m baseline.pafa.lsaa_attribution_ablations \
  --repo-root /files1/Zilong/Acoustic --variant native_only \
  --device cuda --queue --run \
  > logs/native_only_20260918.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n acoustic-addrsc python \
  -m baseline.pafa.lsaa_attribution_ablations \
  --repo-root /files1/Zilong/Acoustic --variant lsaa_without_pafa \
  --device cuda --queue --run \
  > logs/lsaa_without_pafa_20260918.log 2>&1 &
```

队列成功完成三seed后自动生成各自`multiseed_summary.json`（mean、sample SD、每seed状态）。只重建汇总可使用`--aggregate`，不会运行模型。

## 有依据的预算

历史MPS参照中，Native+attributes seeds 0/1/42分别完成25/23/27 epochs、耗时约244/258/258分钟；正式Full分别完成25/21/27 epochs、耗时约313/265/274分钟。若早停轮数相近，单队列三seed的历史MPS量级约12–15小时；50轮上界约为其近两倍。L40 CUDA尚无这两个新variant的实测耗时，预期更快但不承诺具体倍数；HF/KAUH external另有未计时开销。两卡并行wall time由较慢队列决定。

## Claim boundary

- Native-only回答显式C/W监督相对native heads+PAFA的增益，不产生合法属性/HF列；
- without-PAFA回答PCSL/GPAL相对相同LSAA分类系统的增益；
- 两组都保持原test-selected协议，不能升级为clean generalization evidence；
- 当前只有代码与合同准备，不是训练结果。
