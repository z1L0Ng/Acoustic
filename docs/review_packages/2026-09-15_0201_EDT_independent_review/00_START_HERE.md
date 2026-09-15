# 独立论文评审材料包

主审稿件是用户指定的最新 Overleaf 导出：`manuscript/current_paper.pdf`，原文件名 `ICASSP_2026_acoustic_disease (29).pdf`。请以它判断当前投稿文本、图表、页码与排版。PDF保持原样，未重新编译或修改。

## 建议发送方式

最少发送三个文件：主PDF、`01_REVIEW_PROMPT.md`、`02_ADVISOR_REVISION_BRIEF.md`。复制提示词全文作为给独立AI的消息。原始反馈和源码可按需要追加。

为了减少先入为主，最好先只给PDF，让AI形成第一轮阅读判断，再给老师反馈做第二轮核对。如果一次上传整个包，请仍要求它按提示词的两阶段顺序阅读。新对话不要继承此前写作AI的memory或聊天历史。

## 材料说明

- `01_REVIEW_PROMPT.md`：可以直接复制的完整提示词。
- `02_ADVISOR_REVISION_BRIEF.md`：老师修改要求的结构化索引，强调可检查的完成标准。
- `03_REVIEW_CONTEXT.md`：第二轮使用的任务与证据范围说明；辅助材料不能替代正文缺失的解释。
- `manuscript/current_paper.pdf`：主审PDF，共5页；Conclusion从第4页延伸至第5页，之后是References。
- `manuscript/current_pdf_text.txt`：按PDF页序提取的文本，仅方便搜索；视觉结论以PDF为准。
- `manuscript/source/`：当前main工作区的源码、参考文献和原图副本，包含尚未提交的Introduction修改。
- `manuscript/current_source_text.txt`：带源文件名和行号的LaTeX/BibTeX文本，便于AI定位。
- `source_feedback/`：9/13、9/14完整中文会议记录，以及Stephen后续英文反馈原文。
- `optional_evidence/`：既有核心实验、HF-on与CAS/KAUH补评估的结果摘要，用于按需核对数字，属于实验材料，不是评审结论。

## 版本和阅读边界

PDF生成时间元数据为2026-09-15 05:58:52 UTC（01:58:52 EDT）。当前仓库HEAD为`0fe9c99`；Introduction第二段后续修改尚未提交。已确认PDF含这次第二段的关键表述，但未重新编译源码来声明所有文件逐字等同。若源码和PDF有差异，审阅以PDF为准，并列出差异。

老师的批评针对当时的旧稿；请独立判断本PDF有没有解决问题。原始记录中的旅行安排、午夜deadline、旧完成状态和候选名称不是当前稿的事实。9/14记录中Honey已校正为Hanlin、Wit已校正为Wade。

当前作者规则禁止DONE/\revisiondone完成标记；旧反馈或工作流中要求标红DONE的内容不再生效。请勿在任何建议改稿中恢复。

本包不包含写作任务聊天、历史审稿AI结论、数据集音频或模型权重。后续正文若再修改，请相应替换主PDF和源码副本。

材料包整理时间：2026-09-15T02:06:42-04:00。源文件名和行号只用于定位，不属于论文内部注释。
