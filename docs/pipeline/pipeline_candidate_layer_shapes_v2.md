# Pipeline candidate layer and shape registry v2

**Rule:** every number below is tied to an official paper/repository, an accepted local contract, or an inspected local implementation/receipt. `S` is a symbolic dimension. `TBD/HOLD` means that the precise value or interface was not confirmed and must not be guessed.

## 1. Shape notation

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `T` | waveform samples after collation |
| `tau` | spectrogram time frames |
| `F` | spectrogram/fbank bins |
| `L` | acoustic token/time-step length |
| `D` | representation width |
| `K` | native output classes/channels |
| `P` | patients represented in a batch |

## 2. Native data/parser/head work cards

| Work | Variant | Stage | Operation / kernel / stride / blocks | Input | Output | Mask / time-map behavior | Source and verification |
|---|---|---|---|---|---|---|---|
| `W01` | ICBHI official 2017 | raw parser | WAV + cycle TXT row | recording waveform + `(start,end,c,w)` | one variable cycle `[T_i]` + flat4 target | retains source `(start,end)` | accepted local contract; fixed |
| `W01` | native head | classifier | task projection + Linear `D→4` | pooled `[B,D]` | logits `[B,4]` | one row per cycle | fixed contract; current R0 head is `LN(768)+Linear(768,4)` |
| `W02` | HF source benchmark | high-pass/STFT | order 10, 80 Hz; Hanning 256, hop 64 | `[B,60000]` at 4 kHz | log magnitude `[B,938,129]` | fixed 938-frame grid maps to 15 s | PLOS paper; paper-faithful |
| `W02` | HF source benchmark | feature concat | 129 spectrogram + 60 MFCC/delta/delta2 + 4 band energy | `[B,938,*]` | `[B,938,193]` | per-feature min-max | PLOS paper; paper-faithful |
| `W02` | RNN family | recurrent + time-distributed FC | LSTM/GRU/Bi variants; source figure gives cells, text confirms Bi simplified 128 vs full 256 | `[B,938,193]` | one task `[B,938,1]` | fixed grid; four one-vs-rest models | paper; no official code/checkpoint located |
| `W02` | CNN-RNN family | CNN downsample + recurrent + time-distributed FC | exact conv kernels are figure-only/not recovered as code | `[B,938,193]` | one task `[B,469,1]` | fixed half-rate grid; postprocess to events | paper; layer internals `TBD`, reference only |
| `W03` | SPR BioCAS2022 | event parser | interval crop by milliseconds | recording + event row | event waveform `[T_i]` | retains event time lineage | accepted local contract; fixed |
| `W03` | current event heads | two independent heads | `LN(768)+Linear` | pooled `[B,768]` | binary `[B,2]`; seven `[B,7]` | inter/intra evaluated separately | local R0 implementation/smoke; engineering only |
| `W03` | recording head | one native head | task projection + Linear `D→5` | recording pooled `[B,D]` | raw record logits `[B,5]` | Poor Quality remains a class | accepted contract; not active in R0 |
| `W04` | KAUH v3 | parser/grouping | WAV + workbook row; B/D/E filter prefix | recording `[T_i]` | record + raw sound string + P-number | replicas share patient group | accepted local contract; fixed |
| `W04` | raw sound head | classifier | task projection + Linear `D→9` | recording pooled `[B,D]` | logits `[B,9]` | group by P-number | fixed safe native contract; implementation HOLD |
| `W04` | diagnosis branch | separate patient task | aggregation/head `TBD` | 3 filtered recordings + patient row | patient diagnosis logits `K_diag=TBD` | requires patient aggregation | HOLD; raw diagnosis normalization unapproved |

## 3. Encoder/backbone candidates

### `W09` — AST base384, current executable R0 variant

**Variant:** AudioSet `audioset_10_10_0.4593`, local legacy-canonical identity pinned; `input_fdim=128`, `input_tdim=798`, patch/stride `16×16 / 10×10`.

| Stage | Operation | Blocks | Input | Output | Mask / time-map | Verification |
|---|---|---:|---|---|---|---|
| frontend input | local fbank image | — | `[B,1,798,128]` | transpose `[B,1,128,798]` | none | local protocol + smoke |
| patch embed | Conv2d `1→768`, kernel `16×16`, stride `10×10` | 1 | `[B,1,128,798]` | grid `[B,768,12,79]` → `[B,948,768]` | no token mask | local AST code |
| source positions | AudioSet grid `12×101`, time center-crop to 79 | 1 | `[1,1212,768]` acoustic positions | `[1,948,768]` | current 798-frame path uses crop, not interpolation | local AST code |
| special tokens | prepend CLS + DIST | 2 tokens | `[B,948,768]` | `[B,950,768]` | specials have no acoustic time map | local AST code |
| encoder | DeiT-Base block: MHA 12 heads; MLP `768→3072→768` | ×12 | `[B,950,768]` | `[B,950,768]` | no padding mask | timm/AST implementation |
| pooling | LayerNorm; `(CLS+DIST)/2` | 1 | `[B,950,768]` | pooled `[B,768]` | acoustic tokens not returned | local AST wrapper; pooled-only |
| native heads | LN + Linear | 3 heads | `[B,768]` | ICBHI 4; SPR 2 and 7 | dataset-routed | smoke verifies shapes only |

### `W10` — BEATs iter3+ AS2M

**Variant:** local AudioSet-only checkpoint receipt, 12 layers, `D=768`, 250 tensor keys, frozen in four-dataset diagnostics.

| Stage | Operation | Blocks | Input | Output | Mask / time-map | Verification |
|---|---|---:|---|---|---|---|
| waveform | mono 16 kHz | — | `[B,T]` | `[B,T]` | official API accepts `[B,T]` padding mask | official code + local source |
| fbank | Kaldi, 128 bins, 25-ms frame, 10-ms shift; normalize mean 15.41663/std `2×6.55582` | 1 | `[B,T]` | `[B,tau,128]` | waveform mask downsampled to fbank mask | official/local BEATs code |
| patch embed | Conv2d `1→E`, square kernel/stride `p=input_patch_size` | 1 | `[B,1,tau,128]` | `[B,L0,E]` | `p` is checkpoint-configured; decoded value not persisted in accepted receipt → `p=TBD` | exact code; checkpoint value HOLD |
| projection | LN(`E`), optional Linear `E→768` | 1 | `[B,L0,E]` | `[B,L0,768]` | padding mask downsampled again | official/local code |
| position | grouped Conv1d position, kernel 128, groups 16 | 1 | `[B,L0,768]` | `[B,L0,768]` | no source-time map emitted | official/local code |
| encoder | Transformer; MHA 12 heads; FFN `768→3072→768` | ×12 | `[B,L0,768]` | tokens `[B,L0,768]` | returns padding mask | code + local receipt |
| local diagnostic pooling | mean over frames, then mean over 5-s windows for long recordings | — | `[B,L0,768]` | `[B,768]` | discards temporal alignment | completed frozen-feature extraction only |

Local sequence receipt confirms ICBHI frame tensors with `D=768` and variable lengths 10–800; it does not by itself supply a generic four-dataset `time_map`.

### `W12` — OPERA-CT

**Variant:** official OPERA-CT HTS-AT/Swin encoder; the official extractor example uses `input_sec=8`, `dim=768`.

| Stage | Operation | Blocks | Input | Output | Mask / time-map | Verification |
|---|---|---:|---|---|---|---|
| audio/frontend | official OPERA preprocessing to log-mel/model image | — | 16-kHz waveform/window | model grid `[B,1,256,256]` | public extractor pads/splits; no exported token mask/time map | official repo |
| patch embed | Conv2d patch 4, stride `4×4`, width 96 | 1 | `[B,1,256,256]` | `[B,64×64,96]` | no v2 time map | official config/code |
| Swin stage 1 | window attention 8, heads 4, MLP ratio 4 | ×2 | `[B,64×64,96]` | block output same; merge → `[B,32×32,192]` | — | official code |
| Swin stage 2 | heads 8, MLP `192→768→192` | ×2 | `[B,32×32,192]` | merge → `[B,16×16,384]` | — | official code |
| Swin stage 3 | heads 16, MLP `384→1536→384` | ×6 | `[B,16×16,384]` | merge → `[B,8×8,768]` | — | official code |
| Swin stage 4 | heads 32, MLP `768→3072→768` | ×2 | `[B,8×8,768]` | `[B,8×8,768]` | — | official code |
| OPERA feature | LN/pooling through Cola wrapper | — | stage output | pooled `[B,768]` for OPERA-CT example | public v2 adapter not implemented | official repo; candidate |

The `256×256` model grid is an internal reshaped spectrogram grid, not 256 frequency bins from the source recordings.

### `W13` — HeAR 1.0.0

| Stage | Operation | Blocks | Input | Output | Mask / time-map | Verification |
|---|---|---:|---|---|---|---|
| serving input | 2-s mono at 16 kHz | — | `[B,32000]` | internal spectrogram patches | no padding input in published serving example | official model card |
| encoder | masked-autoencoder based on ViT-L | `N=TBD` | internal patches | internal representation | layer/patch dimensions not exposed by accepted serving contract; do not infer from generic ViT-L | official model card; internals HOLD |
| service output | embedding | — | internal | `[B,512]` | no tokens/mask/time map | official model card; pooled-only |

### `W11` — PANNs Cnn14

**Variant:** official `Cnn14`; symbolic `tau` keeps the waveform duration explicit.

| Stage | Operation | Blocks | Input | Output | Mask / time-map | Verification |
|---|---|---:|---|---|---|---|
| frontend | STFT → 64-bin log-mel → BN; SpecAugment in training | — | waveform `[B,T]` | `[B,1,tau,64]` | no public padding mask | official code |
| ConvBlock 1 | 2× Conv3×3 s1, 1→64; avg pool 2×2 | ×1 | `[B,1,tau,64]` | `[B,64,floor(tau/2),32]` | — | official code |
| ConvBlock 2 | 2× Conv3×3, 64→128; pool 2×2 | ×1 | previous | `[B,128,floor(tau/4),16]` | — | official code |
| ConvBlock 3 | 2× Conv3×3, 128→256; pool 2×2 | ×1 | previous | `[B,256,floor(tau/8),8]` | — | official code |
| ConvBlock 4 | 2× Conv3×3, 256→512; pool 2×2 | ×1 | previous | `[B,512,floor(tau/16),4]` | — | official code |
| ConvBlock 5 | 2× Conv3×3, 512→1024; pool 2×2 | ×1 | previous | `[B,1024,floor(tau/32),2]` | — | official code |
| ConvBlock 6 | 2× Conv3×3, 1024→2048; pool 1×1 | ×1 | previous | `[B,2048,floor(tau/32),2]` | — | official code |
| pooling/embedding | mean frequency; time max+mean; Linear 2048→2048 | — | feature map | `[B,2048]` | temporal map discarded | official code; pooled-only |
| source head | Linear 2048→`K_AudioSet` + sigmoid | — | `[B,2048]` | `[B,K_AudioSet]` | replace with native head only after protocol gate | official code |

## 4. Input/complete reference candidates

### `W05` — RespireNet

| Stage | Operation | Blocks | Input | Output | Mask / time-map | Verification |
|---|---|---:|---|---|---|---|
| native preprocessing | blank-region clipping → smart padding → optional concatenation augmentation | — | ICBHI cycle waveform | fixed image `[B,3,H,W]` | no mask; ICBHI-only | official repo/paper |
| backbone stem | torchvision ResNet34: Conv7×7 s2 + max pool | 1 | `[B,3,H,W]` | `[B,64,H/4,W/4]` | — | official repo code |
| residual stages | BasicBlock widths 64/128/256/512, stage blocks `[3,4,6,3]`, downsample at stages 2–4 | ×16 blocks | stem | pooled `[B,512]` | — | `torchvision.resnet34` selected by official repo |
| replacement FC | Dropout → Linear 512→128 → ReLU → Dropout → Linear 128→128 → ReLU | 1 | `[B,512]` | `[B,128]` | — | official repo code |
| classifier | Linear `128→K` | 1 | `[B,128]` | `[B,K]` | native ICBHI `K=4` in reference | official repo |

### `W07` — MVST

| Stage | Operation | Blocks | Input | Output | Mask / time-map | Verification |
|---|---|---:|---|---|---|---|
| spectrogram | STFT/mel resize | — | cycle waveform | `[B,1,256,1024]` | fixed grid; no mask | official paper/repo |
| five views | AST/DeiT-Base with patch shapes `(256,1),(128,2),(64,4),(32,8),(16,16)` and matching non-overlap strides | 5 parallel encoders | `[B,1,256,1024]` | five pooled embeddings `[B,768]` | no time map returned | official paper/card; exact branch configs in repo |
| gated fusion | for each view `G_i=sigmoid(x_i W_i)`, `W_i 768×768`; sum `Σ G_i⊙x_i` | 5 gates | five `[B,768]` | fused `[B,768]` | pooled only | official repo `gated_fusion.py` |
| classifier | Linear `768→4` | 1 | `[B,768]` | `[B,4]` | ICBHI-only reference | official repo |

## 5. Adapter/domain candidates

| Work | Variant/stage | Operation / blocks | Input | Output | Mask / time-map | Source and status |
|---|---|---|---|---|---|---|
| `W15` | Metadata-SCL projector | exact paper-code layer widths not reverified in this audit; supervised-contrastive representation conditioned/grouped by age/sex | `[B,D]` + eligible metadata | `[B,D_p]`, `D_p=TBD` | not a temporal operator | official paper/repo; HOLD where metadata absent |
| `W16` | SG-SCL audio backbone | AST base384, same 768-d family | fbank `[B,1,tau,128]` | pooled `[B,768]` | no token mask in inspected implementation | local source checkout; paper-faithful reference |
| `W16` | domain projector | Linear `768→768`, BN, ReLU, Linear `768→D_p`; repo default `D_p` is argument | `[B,768]` | `[B,D_p]` | stethoscope ID is training side-channel | official/local source; candidate/HOLD |
| `W16` | class/domain heads | native Linear `768→K`; optional domain Linear/projector | `[B,768]` | class `[B,K]`, domain `[B,K_domain]` | device-aware is not domain-free generalization | official/local source |
| `W18` | BEATs backbone | `W10` tokens | `[B,T]` | `[B,L,768]` | padding mask supported in code | local accepted source |
| `W18` | attention pooling | Linear `768→1` + softmax over L + weighted sum | `[B,L,768]` | `[B,768]` | source code does not propagate a v2 time map | official/local source |
| `W18` | projection | Linear `768→H`, BN/LN, ReLU, Linear `H→D_p` | `[B,768]` | `[B,D_p]` | exact `H,D_p` are CLI-configured → symbolic | official/local source |
| `W18` | PCSL/GPAL | patient centroids `[P,D_p]`; within/between variance + global centroid alignment | `[B,D_p]` + patient IDs | scalar loss | patient IDs required | official/local source; source checkpoint test-selected |

## 6. Fusion/aggregation candidates

| Work | Variant/stage | Operation / blocks | Input | Output | Mask / time-map | Source and status |
|---|---|---|---|---|---|---|
| `W07` | MVST gated fusion | five sigmoid gates + weighted sum | 5×`[B,768]` | `[B,768]` | pooled only | official repo; paper reference |
| `W24` | BTS audio-only | Hugging Face `ClapAudioModelWithProjection` | CLAP input features | audio `[B,512]` | CLAP preprocessing contract required | official repo; candidate |
| `W24` | BTS multimodal concat | CLAP audio `[B,512]` + text `[B,512]`; concatenate | 2×`[B,512]` | `[B,1024]` → Linear `1024→K` | text masks handled by CLAP; no acoustic time map | official repo; HOLD for P-Shared |
| `W24` | BTS multimodal add | `alpha*text+(1-alpha)*audio` | 2×`[B,512]` | `[B,512]` → Linear `512→K` | metadata parity/shortcut gate | official repo; HOLD |
| `W30` | diagnosis audio frontend | 16-kHz max 10-s; BEATs | `[B,160000]` | fixed `[B,496,768]` in released config | audio dropout; no v2 time map declared | official repo/config; preprint reference |
| `W30` | projection/composition | Linear `768→H_longformer`; text ≤128 tokens + 496 audio placeholders; global audio anchors stride 4 | audio tokens + text | sequence length `≤4096`, width `H_longformer` | Longformer attention/global masks | official repo; exact hidden follows named checkpoint |
| `W30` | classifier | Longformer sequence classification head | composed sequence | diagnosis logits `[B,K_respagent]` | heterogeneous derived labels | official repo; HOLD, not native-head evidence |

## 7. Decoder/head reference candidate

### `W21` — DeepBreath Cnn10Att

| Stage | Operation | Blocks | Input | Output | Mask / time-map | Verification |
|---|---|---:|---|---|---|---|
| frontend | 4 kHz; 5-s segment; log-mel FFT256/hop64, 32 mels, 250–750 Hz | — | `[B,20000]` | `[B,1,tau,32]` | 30-s max recording, segmented | official config |
| ConvBlock 1 | 2×Conv3×3, 1→64; avg pool2×2 | 1 | input | `[B,64,tau/2,16]` | — | official code |
| ConvBlock 2 | 2×Conv3×3, 64→128; pool2×2 | 1 | previous | `[B,128,tau/4,8]` | — | official code |
| ConvBlock 3 | 2×Conv3×3, 128→256; pool2×2 | 1 | previous | `[B,256,tau/8,4]` | — | official code |
| ConvBlock 4 | 2×Conv3×3, 256→512; pool2×2 | 1 | previous | `[B,512,tau/16,2]` | — | official code |
| ConvBlock 5 | 2×Conv3×3, 512→1024; pool1×1 | 1 | previous | `[B,1024,tau/16,2]` | — | official code |
| temporal features | frequency mean; local max+avg; Linear `1024→1024` | 1 | map | `[B,tau/16,1024]` | interpolation ratio 16 | official code |
| attention head | two Conv1d `1024→K` (attention and class); weighted sum | 1 | `[B,1024,tau/16]` | clip `[B,K]`, frame `[B,tau,K]` | source interpolation/pad restores input frame count | official code |
| disease composition | four separately trained one-vs-rest models | 4 models | recording segments | control/pneumonia/wheezing/bronchiolitis outputs | disease task, not acoustic native task | official repo/paper; external reference |

## 8. Loss/supervision candidates

| Work | Stage | Operation / blocks | Input | Output | Eligibility / status |
|---|---|---|---|---|---|
| `W14` | Patch-Mix | choose/mix AST patch subsets, retain lambda and paired targets | AST patches `[B,L,768]`, `y_a,y_b` | mixed pooled features/logits | ICBHI paper reference; not HF gap interpolation |
| `W14` | mixed CE | `lambda*CE(pred,y_a)+(1-lambda)*CE(pred,y_b)` | `[B,K]`, targets | scalar | only native eligible labels |
| `W14` | contrastive projector | Linear `768→768`, BN, ReLU, Linear `768→768` | `[B,768]` | `[B,768]` | official/local source; candidate |
| `W15` | metadata SupCon | metadata-grouped positive/negative construction | embeddings + eligible metadata | scalar | HOLD for datasets without parity |
| `W16` | stethoscope/domain SupCon | audio class + device-aware contrastive/domain objective | embeddings + device IDs | scalar | ICBHI device only unless equivalent metadata exists |
| `W18` | PAFA | `0.1*PCSL + 0.1*GPAL` in source defaults | projected `[B,D_p]` + patient IDs | scalar | patient IDs required; HF blocked |
| `W26` | adaptive differential denoise | released `DiffTransformerLayer`: RMSNorm → differential MHA (`d_model=256`, 8 heads, depth parameter 6) + AFNO1D + SwiGLU residuals | spectrogram sequence `[B,N,256]` | denoised `[B,N,256]` + denoise features | exact source code; device-hardcoded behavior/repo contradictions noted elsewhere |
| `W26` | bias-denoise loss | label smoothing size 4, smoothing 0.20; released total `0.5*denoise + 0.5*CE` | denoise logits/features + flat4 target | scalar | ICBHI-only candidate; not selected |

## 9. Interface closure summary

| Candidate | `pooled` | `tokens` | `token_mask` | `time_map` | v2 capability |
|---|---:|---:|---:|---:|---|
| Current AST `W09` | `[B,768]` | not returned | no | no | `pooled_only` |
| BEATs `W10` official/local code | mean can produce `[B,768]` | `[B,L,768]` | returned when provided | no | HOLD before `temporal_capable` |
| OPERA-CT `W12` extractor | `[B,768]` | internal only | no exported contract | no | `pooled_only` in v2 |
| HeAR `W13` | `[B,512]` | no public output | no | no | `pooled_only` |
| PANNs Cnn14 `W11` | `[B,2048]` | no public output | no | no | `pooled_only` |
| MVST `W07` | `[B,768]` | no | no | no | `pooled_only` |
| DeepBreath `W21` | clip logits | task-specific frame sequence | source-specific | source-specific fixed grid | external temporal reference, not generic backbone |

No candidate currently closes the full generic `tokens + token_mask + time_map` interface without an additional audited adapter. Therefore the v2 HF generic temporal branch remains **HOLD**, while the paper-faithful `W02` fixed-grid branch remains a separate reference.
