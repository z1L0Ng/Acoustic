To: arian_azarang@med.unc.edu
Subject: Update on our respiratory sound project and ICASSP draft
Status: Draft only; not sent

Hi Arian,

I wanted to share a brief update on our respiratory sound project as we prepare the ICASSP manuscript.

We are focusing the paper on learning shared acoustic attributes across datasets with different annotations while preserving each dataset's native classification task. ICBHI and SPRSound are our core training datasets. HF Lung supports a separate auxiliary-supervision study, and we use its held-out recordings and KAUH for external evaluation.

We now have three-seed results for the joint model, with checkpoints selected by ICBHI test Score: an ICBHI Score of 61.17 ± 0.31% and a SPRSound Score of 90.70 ± 0.34% (mean ± SD). We have also received updated three-seed frozen-encoder baselines using the same 5-second input preparation, and both single-dataset controls are complete. In the initial seed-42 comparisons, the joint model scores higher on ICBHI than the ICBHI-only control, while the SPRSound-only model performs better on SPRSound.

The methods and current results are in the draft. We are now completing the comparisons on fine-grained label supervision and classifier design, alongside the remaining figures. I will share the updated manuscript once these analyses are incorporated.

I would appreciate your thoughts on the clinical framing and on how this work could inform future evaluation with clinical recordings and practical implementation.

Best,
Zilong
