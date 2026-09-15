# 第二轮可用的范围与证据说明

请在独立读完主PDF后再使用本文件。它帮助判断术语和结果范围，不代替正文应有的说明，也不是要求你接受作者解释。

## 当前研究范围

- 主要联合训练来源：ICBHI与SPRSound。原生终点分别为cycle-level四分类与event-level二分类。
- HF source-train只进入另外的辅助监督条件；HF source-test用于属性/标签组合的评测。
- KAUH当前作为外部评测，按患者聚合B/D/E滤波版本。使用兼容标签子集是本研究的范围选择，不是recording-level标注天然禁止训练。
- 当前方法名为LSAA；共享输出为Abnormality、Crackle、Wheeze概率，再使用任务读出。请判断稿件是否解释了每一步，而非仅凭本说明补齐。
- 核心结果沿用所披露的official-test-based checkpoint selection；外部目标的标签不用于模型选择或阈值拟合。可批评其对claim的限制，但需准确指出具体协议与主张之间的关系。

## 指标与对照的阅读提示

当前Table 1报告ICBHI Score、SPRSound official Score、HF CAS AUROC和KAUH patient binary BA。

SPRSound官方Score为AS与HS的均值；不要将二分类BA/AS直接当成该Score。KAUH评测支持为86位兼容患者，正常35、异常51；患者内三个滤波版本不是三个独立患者。

当前LSAA的HF CAS列是固定Wheeze分数对Wheeze/Rhonchi/Stridor标签union的排序诊断：957条eligible记录，661正、296负，三个5秒窗口取最大分数。它不等价于训练了一个CAS/Rhonchi/Stridor-native classifier。请检查其他行是否披露了足够信息，使同列比较可解释；不要自行假定所有行使用相同目标监督。

Native＋C/W是重新训练的原生任务头＋属性监督变体，不是只替换最终解码。Coarse SPR改变细监督、相对loss强度及相应校准条件；不能从该对照单独隔离“标签信息”的因果作用。

HF-on的旧seed42与它对应的历史HF-off配对；不能用主Full三seed中的fresh seed42替换这个参照。

## 提供和未提供的证据

`optional_evidence/`中是既有结果摘要的原样副本：
- 核心Full、单源、Coarse、Native对照；
- HF-on三个seed；
- CAS/KAUH补评估和配对统计。

它们用于核对数字与比较范围，不要求reviewer重复实验。如果内部摘要与稿件对同一结果的用途或证据范围表述不同，请明确指出并请求作者澄清，不自行替任一方补充解释。当前包没有五种frozen-encoder参考行的完整三seed原始日志、模型和预测；尤其本地训练任务尚未定位到BEATs这一行完整可复核产物。请把这一点作为核验材料限制，检查论文对来源的说明，不能仅凭缺文件断言数字错误。

公开方法资源（若需要独立核实）：
- PC-MCL论文：https://arxiv.org/abs/2601.17080
- PC-MCL代码：https://github.com/wa976/PC-MCL
- DCASE 2024 Task 4论文：https://arxiv.org/abs/2406.08056
- DCASE官方实现：https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2024_task4_baseline
- KAUH数据论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC7937981/

DCASE/PC-MCL的新对照未在本轮完成，不应视作Table 1中已存在的实验。代码公开、成品权重存在、获得可比的三seed结果是不同状态。

## 版本

主PDF来自用户指定的Downloads文件，当前源文件来自main工作区快照。已确认PDF含最新Introduction第二段的关键句，图件源码包为当前主目录文件；未另行编译证明全文完全一致。所有审阅优先引用PDF页码；源码仅辅助精确定位。如发现版本差异，请明确列出。
