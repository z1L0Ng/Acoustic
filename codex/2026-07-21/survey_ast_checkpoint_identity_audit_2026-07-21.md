# AST AudioSet Checkpoint Identity Audit

Date: 2026-07-21

Scope: narrow provenance audit for the original AST checkpoint expected by Patch-Mix CL:
`audioset_10_10_0.4593.pth`.

## Executive Verdict

- Verdict: `unresolved`.
- The AST official repository does provide author-attributable download paths for the 0.4593 AudioSet model, but it does not publish a cryptographic checksum, torch state-dict signature, or release asset for `audioset_10_10_0.4593.pth`.
- There are two author-attributable Dropbox paths in the AST codebase:
  - README / batch download script: `ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth`.
  - runtime loader: `cv4knew8mvbrnvq/audioset_0.4593.pth`, saved locally as `audioset_10_10_0.4593.pth`.
- A 1-byte HTTP range probe on 2026-07-21 showed different total byte counts for those two Dropbox objects:
  - README object: `352,587,700` bytes, Dropbox ETag-like value `1624555547023059d`.
  - code-loader object: `352,587,836` bytes, Dropbox ETag-like value `1624426438138010d`.
- Because the two official Dropbox objects differ in byte size and no checksum/state-dict signature is published, they cannot be treated as bitwise-identical without further evidence.
- The Hugging Face `MIT/ast-finetuned-audioset-10-10-0.4593` model has a documented Transformers conversion script that pulls the README Dropbox URL and removes unused ImageNet head tensors. That supports an intended conversion path, but it still does not prove the cached HF model is bitwise/state-dict equivalent to either current AST Dropbox artifact.
- The local HF-to-author conversion proves only functional/format compatibility for the used AST tensors: converted tensors load exactly into the author layout and author-vs-HF logits are numerically close. It does not verify original checkpoint identity.

## Primary Source Evidence

| Source | Evidence | Status |
|---|---|---|
| AST official repo README at `31088be8a3f6ef96416145c4b8d43c81f99eba7a` | Lists "Full AudioSet, 10 tstride, 10 fstride, with Weight Averaging (0.459 mAP)" and links `https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1`. Also provides author OneDrive/Tencent alternatives for manual download of `audioset_10_10_0.4593.pth`. | Official author source; no checksum or byte size. |
| AST official repo `egs/audioset/download_models.sh` at same commit | Downloads `https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1` to `../../pretrained_models/audioset_10_10_0.4593.pth`. | Official author source; matches README URL. |
| AST official repo `src/models/ast_models.py` at same commit | If `../../pretrained_models/audioset_10_10_0.4593.pth` is absent, downloads `https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1` and saves it as `audioset_10_10_0.4593.pth`. | Official author source; URL differs from README/script. |
| AST GitHub releases API | `https://api.github.com/repos/YuanGongND/ast/releases` returned an empty release list on 2026-07-21. | No release asset/checksum found. |
| AST issue search | Searches for `audioset_10_10_0.4593.pth`, `audioset_0.4593.pth`, and `checksum audioset` found usage/debug threads but no author-published checksum. | No authoritative checksum found. |
| Patch-Mix official repo at `836b09fea1b70eb29fe0b25afa481286b56f5104` | `models/ast.py` uses the AST runtime-loader URL `cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1` and saves it as `./pretrained_models/audioset_10_10_0.4593.pth`. | Patch-Mix follows the AST loader path, not the README/script URL. |
| Hugging Face model card/API | `MIT/ast-finetuned-audioset-10-10-0.4593`, revision `f826b80d28226b62986cc218e5cec390b1096902`; model card says the AST team did not write the model card and it was written by Hugging Face. API lists model as public with `ASTForAudioClassification`. | Useful converted model; not original-author provenance. |
| Hugging Face file headers | `pytorch_model.bin`: `346,445,675` bytes, linked ETag `9ca280ba0276a0d2243f7dd561c72b10cc7c3955a2b1f12e42e93983f24b8ca4`; `model.safetensors`: `346,404,948` bytes, linked ETag `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d`. | HF artifact identity only; not original `.pth` identity. |
| Transformers conversion script | The official Transformers script maps `ast-finetuned-audioset-10-10-0.4593` to the AST README Dropbox URL `ca0b...`, loads it, removes `module.v.head.*` / `module.v.head_dist.*`, renames keys, and verifies a small logits slice. | Supports a deterministic intended conversion route from README artifact to HF format, but does not provide original checksum or prove the cached HF snapshot was produced from the current artifact. |

## Exact Source URLs

- AST README pretrained model section:
  `https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/README.md#pretrained-models`
- AST README direct 0.4593 link:
  `https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1`
- AST README OneDrive folder:
  `https://mitprod-my.sharepoint.com/:f:/g/personal/yuangong_mit_edu/ErLKkiP-GwVMgdsCeGEjsmoBMtGvXMkX3tCj5_I0E7ikNA?e=JE9Om8`
- AST README OneDrive file:
  `https://mitprod-my.sharepoint.com/:u:/g/personal/yuangong_mit_edu/EWrY3raql55CqxZNV3cVSkABaoU7pXQxAeJXudE1PTNzQg?e=gwEICj`
- AST batch download script:
  `https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/egs/audioset/download_models.sh#L11`
- AST runtime loader:
  `https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/src/models/ast_models.py#L122-L126`
- AST runtime-loader Dropbox URL:
  `https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1`
- Patch-Mix AST loader:
  `https://github.com/raymin0223/patch-mix_contrastive_learning/blob/836b09fea1b70eb29fe0b25afa481286b56f5104/models/ast.py#L120-L129`
- HF model card:
  `https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593`
- HF model API snapshot:
  `https://huggingface.co/api/models/MIT/ast-finetuned-audioset-10-10-0.4593`
- Transformers original-to-HF conversion script:
  `https://github.com/huggingface/transformers/blob/9ed46fb37cf4c7f885677ad194d2797265e89186/src/transformers/models/audio_spectrogram_transformer/convert_audio_spectrogram_transformer_original_to_pytorch.py#L14-L227`
- AST releases API checked:
  `https://api.github.com/repos/YuanGongND/ast/releases`
- AST checksum issue search checked:
  `https://api.github.com/search/issues?q=repo:YuanGongND/ast+checksum+audioset`

## Observed Artifact Metadata

| Artifact | Attributable source | Probe method | Observed size / checksum |
|---|---|---|---|
| `audioset_10_10_0.4593.pth` | AST README / `download_models.sh` Dropbox URL | `curl -L --range 0-0 --dump-header` on 2026-07-21 | `Content-Range: bytes 0-0/352587700`; Dropbox ETag-like `1624555547023059d`; no SHA256. |
| `audioset_0.4593.pth`, saved as `audioset_10_10_0.4593.pth` | AST `ast_models.py` and Patch-Mix `models/ast.py` Dropbox URL | `curl -L --range 0-0 --dump-header` on 2026-07-21 | `Content-Range: bytes 0-0/352587836`; Dropbox ETag-like `1624426438138010d`; no SHA256. |
| HF `pytorch_model.bin` | `MIT/ast-finetuned-audioset-10-10-0.4593` revision `f826b80d28226b62986cc218e5cec390b1096902` | `curl -L --head` on 2026-07-21 | `346445675` bytes; linked ETag `9ca280ba0276a0d2243f7dd561c72b10cc7c3955a2b1f12e42e93983f24b8ca4`. |
| HF `model.safetensors` | Same HF revision | `curl -L --head` on 2026-07-21 | `346404948` bytes; linked ETag `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d`. |
| Local HF-to-author converted checkpoint | Local compatibility conversion from HF `model.safetensors` | Prior local conversion receipt/report | SHA256 `e7efffdc9d7a63eab33b5919f578333a07c9873b9c93c6979e42ff54f24d4d4d`; 155 converted tensors. |

Notes:

- Dropbox ETag-like values above are not treated as cryptographic checksums.
- The two official Dropbox paths have different observed byte counts, so a bitwise-identity claim is unsafe.
- A full download from official AST Dropbox/OneDrive followed by `sha256sum` would produce a local checksum, but without an author-published checksum it would still verify only the currently served object, not historical identity unless authors confirm the object is unchanged.

## HF Conversion Equivalence Assessment

The cached HF model can not currently be proven bitwise-equivalent or full state-dict-equivalent to the original AST `.pth`.

What is supported:

- The Transformers conversion script is primary-source Hugging Face code and explicitly maps `ast-finetuned-audioset-10-10-0.4593` to the AST README Dropbox URL.
- The script removes the unused ImageNet classifier keys before saving HF format:
  `module.v.head.weight`, `module.v.head.bias`, `module.v.head_dist.weight`, `module.v.head_dist.bias`.
- Local reverse conversion from HF `model.safetensors` to the author key layout loaded exactly for the converted tensors, and author-layout logits matched HF logits within max absolute difference `2.956390380859375e-05`.

What is not supported:

- No source ties the HF snapshot to a specific original `.pth` checksum.
- No source proves the HF snapshot was produced from the current AST Dropbox object rather than an earlier Dropbox object.
- No source proves equivalence between the AST README Dropbox object and the AST runtime-loader/Patch-Mix Dropbox object.
- Because the HF representation omits unused ImageNet head tensors, it cannot be a full bitwise/state-dict equivalent to the original author `.pth`; at best it can be equivalent for the downstream-used AST encoder and AudioSet classifier tensors after deterministic key conversion.

Gate interpretation for Patch-Mix:

- `audioset_pretrain=True` in Patch-Mix should be treated as requiring the author-style `.pth` produced by the AST runtime-loader path unless the authors clarify otherwise.
- The local HF-derived checkpoint is acceptable only as an `official-like HF converted compatibility checkpoint`.
- It should not be reported as the original `audioset_10_10_0.4593.pth`.

## Minimum Author Clarification Needed

To mark the gate as `verified original` or `verified deterministic conversion`, ask the AST and/or Patch-Mix authors for the following:

1. The SHA256 and exact byte size for the intended `audioset_10_10_0.4593.pth`.
2. Whether the AST README Dropbox object `ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth` and the AST/Patch-Mix runtime-loader object `cv4knew8mvbrnvq/audioset_0.4593.pth` are intended to be identical, serialization-equivalent, or different checkpoints.
3. A minimal state-dict signature for the intended original checkpoint:
   - total key count;
   - whether `module.v.head.*` and `module.v.head_dist.*` are present;
   - shapes and checksums, or at least tensor sums, for `module.v.pos_embed`, `module.v.patch_embed.proj.weight`, `module.v.blocks.0.attn.qkv.weight`, and `module.mlp_head.1.weight`.
4. If using the HF checkpoint is acceptable, confirmation of the exact deterministic conversion route:
   original source URL + original checksum + conversion script/commit + expected HF revision/file checksum.

## Optional PAFA Note: BEATs Iter3+ AS2M

This was only a lightweight look-ahead check for PAFA.

- The official Microsoft UniLM BEATs README lists `BEATs_iter3+ (AS2M)` as a pre-trained model with a OneDrive URL:
  `https://github.com/microsoft/unilm/blob/833df7e7832e5064a281131ee64a481afa8e5b95/beats/README.md#pre-trained-and-fine-tuned-tokenizers-and-models`
- The README also shows the intended PyTorch loading contract: checkpoint contains `cfg` and `model`, then `BEATsConfig(checkpoint['cfg'])`, `BEATs(cfg)`, and `load_state_dict(checkpoint['model'])`.
- No checksum was found in this lightweight pass. If PAFA reproduction depends on a non-Microsoft mirror, treat it as official-like until the Microsoft OneDrive artifact is downloaded and hashed, or an author-published checksum is found.

## Final Gate Status

Patch-Mix AST checkpoint identity remains blocked:

- `verified original`: no.
- `verified deterministic conversion`: no.
- `unresolved`: yes.

Recommended reproduction label until clarified:

- `official-like / HF-converted AST compatibility checkpoint`, not `official faithful original checkpoint`.
