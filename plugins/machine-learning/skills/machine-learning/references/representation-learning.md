# Representation & Self-Supervised Learning

How to learn reusable representations without (many) labels, and how to adapt pretrained models efficiently.
This is the engine behind foundation models: **pretrain once on cheap unlabeled data, transfer everywhere.**
The "what should the representation be invariant to?" question from [deep-learning.md](deep-learning.md) §1 is
the heart of this file.

---

## 1. The paradigm and why it dominates

Labels are expensive; raw data is abundant. **Self-supervised learning (SSL)** invents a supervisory signal
*from the data itself* (a "pretext task"), learns a representation, then transfers to downstream tasks with
little labeled data. This is how LLMs, CLIP, DINOv2, and modern foundation models are built. Transfer learning
from a strong pretrained backbone is the **default starting point** for almost any applied DL problem — training
from scratch is for when you're studying pretraining itself.

**How to evaluate a representation** (decoupled from any one task):
- **Linear probe:** freeze the backbone, train a linear classifier on top. Measures linear separability /
  representation quality. The standard SSL yardstick.
- **k-NN probe:** classify by nearest neighbors in feature space — no training, tests metric structure.
- **Fine-tuning:** unfreeze and adapt — measures transfer ceiling, but conflates representation quality with
  adaptation.
- **Few-shot / low-data transfer:** the real test of whether the representation generalizes data-efficiently.
- Report linear-probe AND fine-tuning; a representation can win one and lose the other.

## 2. The families of SSL objectives

### Contrastive (instance discrimination)
Pull together representations of two augmented views of the same example ("positives"), push apart different
examples ("negatives"). The loss is **InfoNCE** (a mutual-information lower bound):
$\mathcal{L}=-\log\frac{\exp(\text{sim}(z_i,z_i^+)/\tau)}{\sum_j \exp(\text{sim}(z_i,z_j)/\tau)}$.
- **SimCLR** — needs large batches (many negatives) and strong augmentations; the augmentation choice *is* the
  prior (it defines what the representation ignores).
- **MoCo** — a momentum encoder + queue decouples negatives from batch size (works with modest hardware).
- **Key knobs:** temperature $\tau$, augmentation strength, projection head (discard it for downstream), and
  enough negatives. **Augmentation design is the most important decision** — it encodes the invariances.

### Non-contrastive / self-distillation (no negatives)
Avoid the need for negatives (and the collapse risk) via architectural tricks:
- **BYOL / SimSiam** — predict one view's representation from another with a momentum target / stop-gradient to
  prevent collapse (no negatives at all).
- **DINO / DINOv2 / DINOv3** — self-distillation with a momentum teacher (student matches a sharpened,
  centered teacher distribution over views; centering+sharpening prevents collapse). **DINOv2** added
  curated-data pretraining + iBOT-style patch-level masking and produces excellent general-purpose *frozen*
  visual features. **DINOv3** (Meta, 2025) scales to a 7B-param ViT on 1.7B curated images and introduces
  **Gram anchoring** to stop dense-feature degradation over long schedules; smaller models are *distilled* from
  the 7B teacher. It delivers sharp, high-quality **dense** features that beat weakly/fully-supervised baselines
  on segmentation, depth, and matching — the current default frozen vision backbone, especially when you need
  dense (per-pixel/per-patch) features, not just a global embedding.
- **Barlow Twins / VICReg** — explicitly decorrelate/regularize feature dimensions (invariance + variance +
  covariance terms) to prevent collapse; principled, batch-size robust, no momentum encoder or stop-gradient
  needed. **JEPA** (I-JEPA, V-JEPA 2) predicts *latent* representations of masked regions rather than pixels —
  a non-generative middle ground between MAE and contrastive that yields strong semantic features and underpins
  Meta's world-model line.

### Masked / generative prediction
Mask part of the input, reconstruct it:
- **Masked Language Modeling (BERT)** — the original at-scale SSL for text; bidirectional understanding (vs.
  autoregressive GPT for generation — see [transformers-llms.md](transformers-llms.md)).
- **Masked Autoencoders (MAE)** — mask ~75% of image patches, reconstruct pixels; scalable, simple, strong.
  **BEiT** masks tokens instead of pixels.
- **Masked modeling generalizes** to audio (wav2vec 2.0 / HuBERT / w2v-BERT / BEATs — masked prediction of
  quantized/clustered units; the backbone of modern speech SSL and ASR pretraining), video, graphs, and time
  series. The recipe: corrupt → reconstruct → keep the encoder.
- **Contrastive vs. masked, a rule of thumb:** contrastive/self-distillation (DINO, CLIP) yields features with
  strong *global/semantic* linear separability (great linear probes, retrieval, zero-shot); masked
  reconstruction (MAE, BERT) yields features that often *fine-tune* better and capture local/dense structure.
  Pick by whether you need frozen-feature quality or a fine-tuning ceiling — and report both probes (§1).

### Multimodal / cross-modal alignment
- **CLIP** — contrastively align image and text encoders on web image–caption pairs into a **shared embedding
  space** via a symmetric InfoNCE over the in-batch similarity matrix, enabling zero-shot classification
  (compare image to text-prompt embeddings) and powering text-to-image models and VLMs. The softmax loss
  couples to batch size (needs large batches / all-gather negatives). **SigLIP / SigLIP 2** replace it with a
  **pairwise sigmoid** loss (each image–text pair judged independently as match/non-match) — decoupled from
  batch size, more efficient, and stronger at small/medium batch; SigLIP 2 adds captioning + self-distillation
  + dense-feature objectives. The shared space is the product: retrieval, zero-shot, grounding, and
  conditioning generative models all fall out of it. **Note on VLMs:** modern vision-language models typically
  pair a CLIP/SigLIP-style *vision encoder* with an LLM via a projector (LLaVA-style); the contrastive encoder
  supplies the visual tokens.

## 3. Embeddings as the deliverable

Often the representation *is* the product (search, retrieval, recommendation, RAG, dedup, clustering, anomaly
detection).
- **Text embeddings:** the SOTA shifted from BERT-scale encoders (Sentence-BERT, E5, GTE, BGE) to
  **LLM-backbone embedding models** trained with contrastive + instruction tuning — **Qwen3-Embedding**,
  **Gemini Embedding**, Voyage, Jina, NV-Embed, and similar top MTEB(/MMTEB) as of 2026. They are
  **instruction-aware** (prepend a task/query instruction; queries and documents often use different prompts)
  and asymmetric query/doc handling matters. Choose by *your* retrieval benchmark, not the leaderboard
  aggregate (MTEB is partially saturated/contaminated).
- **Matryoshka Representation Learning (MRL)** is now standard: train so that truncating the embedding to the
  first $k$ dims still works, giving one model many dimensionalities. Truncating to ~256 typically costs only a
  few % accuracy for ~4× storage/latency savings — use it for coarse-to-fine retrieval (cheap shortlist on
  short vectors, rerank on full). Pairs well with **binary/int8 quantization** of vectors for large indices.
- **Metric learning** (when you need a tuned similarity space): triplet loss, contrastive loss, ArcFace/
  CosFace (margin-based, SOTA for face/fine-grained recognition), and hard-negative mining (which dominates
  results — mining strategy > loss choice; in-batch + mined hard negatives is the standard recipe).
- **Retrieval at scale:** approximate nearest neighbor (FAISS, HNSW, ScaNN, DiskANN), often **bi-encoder
  retrieve → cross-encoder rerank** for quality, increasingly **hybrid (BM25 + dense, RRF-fused)**. Normalize
  embeddings; pick cosine vs. dot product deliberately. **Late-interaction** models (ColBERT/ColPali —
  per-token vectors with MaxSim) trade index size for higher recall and are strong for hard retrieval and
  document/visual-document search.
- **Evaluating embeddings:** Recall@k / MRR / nDCG on held-out queries; MTEB/MMTEB for text. Watch for
  **train/test leakage via near-duplicate items** and benchmark contamination of LLM-backbone embedders (a
  recurring [data.md](data.md) / [evaluation-statistics.md](evaluation-statistics.md) theme).

## 4. Transfer learning: how to adapt a pretrained model

A spectrum from cheapest to most expressive:
1. **Zero-shot / prompting** — no training (CLIP zero-shot, LLM prompting).
2. **Linear probe / feature extraction** — freeze backbone, train a small head. Cheap, robust, great for small
   data, avoids catastrophic forgetting.
3. **Parameter-efficient fine-tuning (PEFT)** — adapt with a tiny fraction of parameters (§5).
4. **Full fine-tuning** — update all weights. Most expressive, most compute/memory, risk of overfitting (small
   data) and catastrophic forgetting.
- **Choosing:** small data / large model → probe or PEFT; abundant data / domain far from pretraining → full
  fine-tune; serving many tasks → PEFT adapters (swap per task). Discriminative LR (smaller for lower layers)
  and gradual unfreezing help full fine-tuning.

## 5. Parameter-Efficient Fine-Tuning (PEFT) — the default for large models

Full fine-tuning of large models is expensive and produces a full-size checkpoint per task. PEFT updates a tiny
set of parameters and matches full fine-tuning on most tasks.
- **LoRA** (Low-Rank Adaptation): freeze weights $W$, learn a low-rank update $\Delta W = \tfrac{\alpha}{r}BA$
  ($B$ init zero so training starts at the base model). Train ~0.1–1% of parameters; **adapters are tiny and
  swappable** (serve many tasks on one base — vLLM/S-LoRA hot-swap adapters per request); merge into $W$ at
  inference for zero added latency. The dominant method. Knobs: rank $r$, scaling $\alpha$ (a common heuristic
  is $\alpha=2r$, but LR matters more), **which modules** (target *all linear* layers — attention *and* MLP —
  not just q/v; this is the single biggest quality lever), dropout, and a **separate (higher) LR for the LoRA
  params** (LoRA+). LoRA is **highly LR-sensitive** — sweep it before reaching for fancier variants.
- **QLoRA:** LoRA on top of a **4-bit (NF4) quantized** frozen base + double quantization + paged optimizers →
  fine-tune large models on a single GPU with little quality loss. The standard recipe on modest hardware;
  slightly slower per step than fp16 LoRA due to dequantization.
- **LoRA variants worth knowing (2025–26):** **rsLoRA** (rank-stabilized scaling $\alpha/\sqrt{r}$ — makes
  large ranks actually train; essentially free, prefer it when supported); **DoRA** (decompose $\Delta W$ into
  magnitude + direction — closes much of the LoRA↔full gap, especially at low rank, for ~little extra cost,
  now common as QLoRA+DoRA); **PiSSA / LoRA-GA / OLoRA** (SVD/gradient-aware *initialization* of $A,B$ from the
  base weights for faster, better convergence). Caveat from controlled studies: **well-tuned vanilla LoRA
  often matches the fancy variants** under matched budgets — the gains are real but modest and LR-dominated.
  Practical 2026 starting point: rank 16–32, target all-linear, rsLoRA on, DoRA if you need the last bit and
  can afford it; raise rank (32–64) when adapting far from the base distribution.
- **Other PEFT:** adapter layers (bottleneck MLPs inserted in blocks), prefix/prompt tuning (learn soft prompt
  tokens — cheap but weaker), $(IA)^3$, BitFit (bias-only). Hugging Face `peft` implements these.
- **When PEFT vs. full:** PEFT for most fine-tuning, especially large models, multi-task serving, and limited
  compute; full fine-tuning when you have lots of in-domain data and the budget, or when adapting the base
  distribution substantially (new modality/language). Note LoRA tends to *preserve* base capabilities better
  (less catastrophic forgetting) but can underperform full FT on tasks requiring large representational change.

## 6. Pitfalls specific to representation/transfer learning

- **Augmentation = the prior.** In contrastive/SSL, your augmentations *define* what the model treats as
  nuisance. Color-jitter teaches color invariance — bad if color is the label. Choose augmentations to match
  downstream invariances; this dominates results.
- **Representation collapse.** Non-contrastive methods can collapse to constant outputs; the stop-gradient/
  momentum/decorrelation tricks exist to prevent it — monitor feature variance.
- **Probe vs. fine-tune disagreement.** A frozen representation can linear-probe poorly but fine-tune well (or
  vice versa). Report both; don't claim "better representations" from one number.
- **Pretrain–downstream leakage / contamination.** If your downstream test data (or near-duplicates) was in the
  pretraining corpus, transfer results are inflated — the foundation-model version of test-set leakage. Critical
  for honest evaluation (see [evaluation-statistics.md](evaluation-statistics.md), [data.md](data.md)).
- **Distribution shift between pretrain and target.** Web-pretrained features can transfer poorly to
  specialized domains (medical, satellite, scientific) — measure, and consider domain-specific continued
  pretraining.
- **Catastrophic forgetting** during fine-tuning — the model loses pretrained capabilities; PEFT, replay, lower
  LR, and KL/regularization to the base mitigate it.

## 7. Reach-for table

| Goal | Reach for |
|---|---|
| General visual features off the shelf | DINOv3 (frozen), esp. for dense features; CLIP/SigLIP for text-aligned |
| Image–text alignment / zero-shot / retrieval | SigLIP 2 (sigmoid loss) > CLIP |
| Pretrain on unlabeled images | DINOv3/iBOT (self-distill) or MAE/JEPA (masked); JEPA for latent prediction |
| Pretrain on unlabeled text | Masked LM (understanding) or autoregressive (generation) |
| Adapt a large model cheaply | QLoRA (+DoRA), target all-linear, rsLoRA on; sweep LR first |
| Small labeled data + big pretrained model | Linear probe or PEFT (not full fine-tune) |
| Semantic search / RAG | LLM-backbone embedding model (instruction-aware) + MRL truncation + ANN + reranker; hybrid BM25+dense |
| Hard / document-visual retrieval | Late interaction (ColBERT / ColPali) |
| Fine-grained similarity (faces, products) | Metric learning (ArcFace) + hard-negative mining |

**Canonical references:** Chen et al. 2020 (SimCLR); He et al. 2020 (MoCo); Grill et al. 2020 (BYOL); Bardes
et al. 2022 (VICReg); Caron et al. 2021 / Oquab et al. 2023 / Siméoni et al. 2025 (DINO/DINOv2/DINOv3);
Assran et al. 2023 (I-JEPA); He et al. 2022 (MAE); Devlin et al. 2018 (BERT); Radford et al. 2021 (CLIP);
Zhai et al. 2023 / Tschannen et al. 2025 (SigLIP / SigLIP 2); Kusupati et al. 2022 (Matryoshka); Khattab &
Zaharia 2020 (ColBERT); Hu et al. 2021 (LoRA); Dettmers et al. 2023 (QLoRA); Liu et al. 2024 (DoRA); Kalajdzievski
2023 (rsLoRA); Meng et al. 2024 (PiSSA).
