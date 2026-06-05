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
- **DINO / DINOv2** — self-distillation with a momentum teacher; **DINOv2** produces excellent general-purpose
  *frozen* visual features (strong dense/semantic features, great linear probes) — a top vision backbone to
  reach for off the shelf.
- **Barlow Twins / VICReg** — explicitly decorrelate/regularize feature dimensions to prevent collapse;
  principled, batch-size robust.

### Masked / generative prediction
Mask part of the input, reconstruct it:
- **Masked Language Modeling (BERT)** — the original at-scale SSL for text; bidirectional understanding (vs.
  autoregressive GPT for generation — see [transformers-llms.md](transformers-llms.md)).
- **Masked Autoencoders (MAE)** — mask ~75% of image patches, reconstruct pixels; scalable, simple, strong.
  **BEiT** masks tokens instead of pixels.
- **Masked modeling generalizes** to audio (wav2vec 2.0 / HuBERT), video, graphs, and time series. The recipe:
  corrupt → reconstruct → keep the encoder.

### Multimodal / cross-modal alignment
- **CLIP** — contrastively align image and text encoders on web image–caption pairs into a **shared embedding
  space**, enabling zero-shot classification (compare image to text-prompt embeddings) and powering text-to-
  image models and VLMs. **ALIGN, SigLIP** (sigmoid loss, more efficient) are the modern variants. The shared
  space is the product: retrieval, zero-shot, grounding, and conditioning generative models all fall out of it.

## 3. Embeddings as the deliverable

Often the representation *is* the product (search, retrieval, recommendation, RAG, dedup, clustering, anomaly
detection).
- **Text embeddings:** sentence/document encoders (Sentence-BERT, E5, GTE, BGE, and instruction-tuned
  embedding models) for semantic search and RAG. Choose by your retrieval benchmark (MTEB), not vibes.
- **Metric learning** (when you need a tuned similarity space): triplet loss, contrastive loss, ArcFace/
  CosFace (margin-based, SOTA for face/fine-grained recognition), and hard-negative mining (which dominates
  results — mining strategy > loss choice).
- **Retrieval at scale:** approximate nearest neighbor (FAISS, HNSW, ScaNN), often **bi-encoder retrieve →
  cross-encoder rerank** for quality. Normalize embeddings; pick cosine vs. dot product deliberately.
- **Evaluating embeddings:** Recall@k / MRR / nDCG on held-out queries; MTEB for text. Watch for **train/test
  leakage via near-duplicate items** (a recurring [data.md](data.md) theme).

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
- **LoRA** (Low-Rank Adaptation): freeze weights $W$, learn a low-rank update $\Delta W = BA$ (rank $r$ small).
  Train ~0.1–1% of parameters; **adapters are tiny and swappable**; merge into $W$ at inference for zero added
  latency. The dominant method. Knobs: rank $r$, $\alpha$ (scaling), which modules (attention proj + MLP),
  dropout.
- **QLoRA:** LoRA on top of a **4-bit quantized** frozen base (NF4 + paged optimizers) → fine-tune large models
  on a single GPU with little quality loss. The standard recipe for fine-tuning big models on modest hardware.
- **Other PEFT:** adapter layers (bottleneck MLPs inserted in blocks), prefix/prompt tuning (learn soft prompt
  tokens), $(IA)^3$, BitFit (bias-only), DoRA. Hugging Face `peft` implements these.
- **When PEFT vs. full:** PEFT for most fine-tuning, especially large models, multi-task serving, and limited
  compute; full fine-tuning when you have lots of in-domain data and the budget, or when adapting the base
  distribution substantially (e.g., new modality/language).

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
| General visual features off the shelf | DINOv2 (frozen) or a strong supervised/CLIP backbone |
| Image–text alignment / zero-shot / retrieval | CLIP / SigLIP |
| Pretrain on unlabeled images | MAE (generative) or DINO/iBOT (self-distill) |
| Pretrain on unlabeled text | Masked LM (understanding) or autoregressive (generation) |
| Adapt a large model cheaply | LoRA / QLoRA (PEFT) |
| Small labeled data + big pretrained model | Linear probe or PEFT (not full fine-tune) |
| Semantic search / RAG | Text embedding model (pick via MTEB) + ANN + reranker |
| Fine-grained similarity (faces, products) | Metric learning (ArcFace) + hard-negative mining |

**Canonical references:** Chen et al. 2020 (SimCLR); He et al. 2020 (MoCo); Grill et al. 2020 (BYOL);
Caron et al. 2021 / Oquab et al. 2023 (DINO/DINOv2); He et al. 2022 (MAE); Devlin et al. 2018 (BERT);
Radford et al. 2021 (CLIP); Zhai et al. 2023 (SigLIP); Hu et al. 2021 (LoRA); Dettmers et al. 2023 (QLoRA).
