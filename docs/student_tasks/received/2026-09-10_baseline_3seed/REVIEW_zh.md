# 本科生三seed基线汇总接收｜2026-09-10

状态：已收到学生报告的三seed汇总；用户已确认encoder frozen、5s输入及与主方法相同的输入处理。已交给写作更新Table1背景块。本次文件不含逐seed原始metrics/config，尚不能独立复算均值和sample SD；当前按来稿汇总值记录。

用户确认（2026-09-10）：本文件采用frozen encoder、5s设置，input方法与main method一致。此前根据旧材料提出的2s/1s猜测已被本条更正，不应继续用于新结果的比较说明。该确认覆盖输入处理与encoder训练范围，不等于其他head、优化器或数据集训练方式全部相同。

来源文件已原样保存为同目录main.tex。报告作者未在文件中署名；按内容与此前25组单数据集基线对接，不据此推断学生当前GPU资源。

## 本次材料包含

AST、BEATs、PANNs、OPERA-CT、HeAR五个模型，分别覆盖ICBHI flat4、SPRSound binary、SPRSound event7、KAUH nine-class、HF four-label temporal，共25行配置。caption声明每行是seeds 0/1/42的mean ± sample SD。HF行另外采用five-fold probability ensemble；fold不等同于独立seed。

## Table 1更新数据

以下按来稿原值记录，单位为百分数；两数据集是分别训练/选择的native任务结果，不是一个联合checkpoint的双端点。

| 模型 | ICBHI Score | SPR inter binary official Score |
|---|---:|---:|
| AST | 51.40 ± 0.48 | 90.20 ± 0.18 |
| BEATs | 51.68 ± 1.19 | 92.37 ± 0.35 |
| PANNs | 52.08 ± 0.16 | 91.23 ± 0.55 |
| OPERA-CT | 39.96 ± 1.68 | 79.45 ± 0.00 |
| HeAR | 49.50 ± 0.62 | 87.89 ± 0.31 |

本材料用于将论文Table1的五行旧seed42背景值更新为mean±SD，并同时更新caption、分块标题、正文举例和来源说明。设置采用用户已确认的frozen encoder、相同5s输入处理；不存在此前所写的2s与5s输入差异。每个数据集仍为单独的native任务背景模型，不能仅因输入一致就称为与主方法完全匹配的训练对照。逐seed原始metrics/config尚未随本次文件提供，保留为复核材料缺项，不再将已确认的输入设置列为待确认。

## 适用范围

- ICBHI在来稿中明确official-test selection；SPR intra/inter separately selected，正文若采用必须保留。
- SPR event7不替代我们binary原生端点或C/W属性AUROC。
- KAUH是有本地训练/validation的九分类，与我们86 compatible patients的外部四分类/二分类不同，不能直接混表比较。
- HF是I/E/CAS/DAS四标签时间任务、五折概率集成，与我们D/W recording AUROC和positive-interval recall不同。来稿event F1为1.11–2.32%，需要其事件匹配/后处理定义解释；仅凭汇总不能判定原因。
- 文件没有PAFA、SG-SCL、Patch-Mix的新增多seed强方法复现，也没有固定ICBHI模型到其他数据集的transfer结果。
- 本文件不含Figure 1的声学特征/分布图或class-matched分析；该项不因收到此表而勾选完成。
- 本地四条件full-finetuning队列继续按既有授权运行，此表不替代其中的ICBHI-only、SPRSound-only、Coarse或Native+attributes对照。

## 表内核对

已确认五组各五行、caption的seed声明与mean±sample SD记法。ICBHI Score与(SP+SE)/2、SPR/KAUH Score与(AS+HS)/2的均值关系仅见两位小数舍入量级差异；不据此声称逐seed计算已复现。来稿说明seed42源数字仅保留两位小数，后续应尽量使用逐seed未舍入原值重新汇总。没有编译、外发、Git或Notion操作。
