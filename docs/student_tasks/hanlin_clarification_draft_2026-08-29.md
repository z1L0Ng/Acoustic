Hanlin 澄清消息草稿

Hanlin，你现在需要先把 original benchmark reproduction 做清楚，再进入 cross-dataset adaptation。对于每个直接对照，必须分别列出 paper score、local score 和 delta，并说明对应的 dataset、task、split、metric 和 evaluation protocol。请注意，冻结 encoder 后在 ICBHI 上重新训练一个 new head，只能算我们的 local downstream baseline，不能直接称为原论文复现。按照目前的文献审计，PAFA（BEATs + CE）是优先核对的一条 direct baseline；请先确认 AST 和 BEATs 目前已有的 paper、代码、checkpoint、evaluation script 或结果材料，以及预计什么时候可以完成第一版。cross-dataset adaptation 放在对应的 original reproduction 确认之后，不需要现在并行扩展。
