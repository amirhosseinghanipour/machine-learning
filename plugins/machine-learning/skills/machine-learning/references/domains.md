# Domains: Applied ML by Field

Per-domain SOTA, defaults, evaluation, and the gotchas that bite. The cross-cutting rigor (splitting, leakage,
significance) in [evaluation-statistics.md](evaluation-statistics.md) and [data.md](data.md) applies
everywhere — this file adds what's *specific* to each field. Current to 2026.

---

## 1. Computer Vision

- **Backbones:** ViT and modern CNNs (ConvNeXt) both strong; **CNNs win at small/medium data** (convolutional
  prior), **ViTs win with scale/pretraining**. Off-the-shelf features: **DINOv2** (excellent frozen features),
  CLIP/SigLIP (image-text). Default to a pretrained backbone (see [representation-learning.md](representation-learning.md)).
- **Tasks & metrics:** classification (top-1/top-5); **detection** (mAP @ IoU thresholds — DETR-family
  transformers and YOLO-family real-time detectors); **segmentation** (mIoU/Dice — semantic; panoptic quality;
  **Segment Anything (SAM/SAM 2)** for promptable/zero-shot masks); pose, depth, tracking, optical flow.
- **Generation:** latent diffusion / flow matching, increasingly DiT backbones; video generation is the active
  frontier (see [generative-models.md](generative-models.md)).
- **Gotchas:** augmentation is high-leverage (and must be label-preserving — don't flip when orientation is the
  label); **shortcut learning** (background/texture/watermark instead of object — test with backgrounds swapped);
  robustness to corruptions/OOD (ImageNet-C/-R/-A); class imbalance in detection (focal loss); resolution and
  preprocessing must match between train and inference. Evaluate on the actual deployment imagery, not just
  curated benchmarks.

## 2. Natural Language Processing

- **Almost everything is an LLM now** (see [transformers-llms.md](transformers-llms.md)): generation, QA,
  summarization, classification, extraction, translation — via prompting, fine-tuning (LoRA), or RAG. But
  **smaller fine-tuned encoders (DeBERTa-class) or even linear/GBM on TF-IDF still win** on narrow, high-volume
  classification at a fraction of the cost — benchmark them as baselines.
- **Embeddings/retrieval:** sentence embedding models (pick via **MTEB**) + ANN + reranker for search/RAG (see
  [representation-learning.md](representation-learning.md)).
- **Evaluation:** task-specific verifiable metrics where possible; BLEU/ROUGE/chrF for MT/summarization are weak
  surface metrics — supplement with human/LLM-judge (with the bias caveats in
  [evaluation-statistics.md](evaluation-statistics.md)); for classification use macro-F1 with imbalance care.
- **Gotchas:** tokenization effects (multilingual fairness, numbers, code); benchmark **contamination** (huge for
  LLMs — use held-out/fresh sets); spurious cues / annotation artifacts in NLI-style datasets (models exploit
  them); domain/temporal shift in text; multilingual ≠ English-only performance — evaluate per language.

## 3. Speech & Audio

- **ASR (speech-to-text):** Whisper-class encoder-decoder and self-supervised (wav2vec 2.0/HuBERT) encoders;
  Conformer (conv+attention) and increasingly SSM/Mamba-based models for long audio. Metric: **WER/CER** (and be
  careful with normalization/text-normalization, which can swing WER).
- **TTS / audio generation:** neural codec + autoregressive or diffusion/flow-matching synthesis (very strong,
  near-human); voice cloning raises consent/deepfake concerns. Metrics: MOS (human), intelligibility via ASR-WER,
  speaker similarity, FAD.
- **Other:** speaker ID/diarization, keyword spotting, music/audio tagging, source separation, sound event
  detection.
- **Gotchas:** features (log-mel spectrograms) and SpecAugment; **speaker leakage** (same speaker across
  train/test inflates results — split by speaker); variable-length batching/padding; sample-rate and codec
  consistency; noise/reverb robustness; far-field vs. close-talk mismatch. Evaluate on realistic acoustic
  conditions.

## 4. Time Series & Forecasting

- **Strong simple baselines are essential and often win:** naive/seasonal-naive (predict last value / last
  season), exponential smoothing (ETS), ARIMA, **and gradient-boosted trees on lag/calendar features (often the
  practical winner)**. Always beat these before claiming a deep model helps — many "deep forecasting beats
  classical" claims fail under honest baselines (the M-competitions are sobering).
- **Deep/Modern:** DeepAR (probabilistic RNN), N-BEATS/N-HiTS, Temporal Fusion Transformer, and **time-series
  foundation models** (TimesFM, Chronos, Moirai — pretrained, strong zero-shot forecasting, a 2024–2026
  development). SSM/Mamba models suit long sequences.
- **Evaluation:** **strictly temporal splits** + rolling-origin/backtesting CV — a random split leaks the future
  and is the #1 time-series error. Metrics: MAE/RMSE, MASE (scale-free, vs. naive), sMAPE, **pinball/quantile
  loss** for probabilistic forecasts (forecast *distributions*, not just points). Report at the relevant horizon.
- **Gotchas:** non-stationarity/drift; look-ahead bias in features (any feature using future info); leakage via
  global normalization across the time boundary; hierarchical/coherent forecasts (reconciliation); irregular
  sampling and missing timestamps; **don't shuffle**.

## 5. Tabular (the GBM stronghold)

- **Gradient-boosted trees (XGBoost/LightGBM/CatBoost) are SOTA** on most tabular problems — start here, not with
  deep nets (see [classical-ml.md](classical-ml.md)). Deep tabular (FT-Transformer, TabNet) and **TabPFN**
  (in-context Bayesian inference, excellent on *small* tabular) have niches; AutoGluon for strong stacked
  baselines fast.
- **Where the wins are:** feature engineering, leakage prevention, categorical handling, and calibration — not
  model architecture.
- **Gotchas:** **leakage is the dominant risk** (target/temporal/group — audit every feature, see
  [data.md](data.md)); high-cardinality categoricals (target-encode inside folds or use CatBoost); imbalance
  (right metric + threshold); distribution shift between collection and deployment; trees don't extrapolate
  (detrend/difference for trends).

## 6. Recommender Systems

- **Approaches:** collaborative filtering (matrix factorization, ALS), **two-tower / dual-encoder** retrieval +
  ranking (the industrial standard), sequential/session-based (transformers — SASRec/BERT4Rec), graph-based
  (see [graph-geometric.md](graph-geometric.md)), and feature-rich rankers (DLRM, gradient-boosted). Increasingly
  LLM-augmented (semantic IDs, generative retrieval).
- **Evaluation:** ranking metrics (Recall@k, **nDCG**, MAP, MRR, hit-rate) offline, but **offline–online
  divergence is severe** — the gold standard is online **A/B testing** (and counterfactual/off-policy evaluation
  to estimate online impact offline). Optimize for the business metric (engagement/retention/revenue), not just
  rating prediction.
- **Gotchas:** **temporal split** (predict future interactions, never random — a classic leakage trap);
  cold-start (new users/items); popularity bias and feedback loops (the system shapes the data it later trains
  on); **exposure/selection bias** (you only observe feedback on shown items — debias with IPS); position bias;
  fairness/diversity/filter-bubble concerns.

## 7. Multimodal & Vision-Language Models (VLMs)

- **Architecture trend (2025–2026):** from "vision encoder + adapter bridge → frozen LLM" toward **single
  transformers trained from scratch on mixed-modality (interleaved image/text/audio/video) data** and
  unified/native multimodal models. Strong open VLMs (Qwen-VL-class and peers) rival proprietary frontier models
  on multimodal benchmarks; capabilities now include agentic UI/screen understanding, long-context video, and
  visual reasoning.
- **Training:** contrastive alignment (CLIP/SigLIP) for the shared space; instruction tuning on multimodal data;
  RLHF/RLVR for reasoning (see [transformers-llms.md](transformers-llms.md)).
- **Evaluation:** multimodal benchmarks (MMMU, MMBench, document/chart/OCR VQA, video QA); the same
  **contamination** and **LLM-judge** caveats apply; test grounding (does it actually use the image vs. answer
  from language priors? — counterfactual image swaps reveal this).
- **Gotchas:** modality imbalance (one modality dominates / is ignored), hallucinating objects not in the image,
  resolution/tiling for high-res images and documents, and alignment/safety across modalities.

## 8. Scientific ML (AI for Science)

- **Where ML is transforming science:** protein structure/design (AlphaFold-class, equivariant — see
  [graph-geometric.md](graph-geometric.md)), molecular property prediction & generation (flow matching/diffusion
  over molecules, see [generative-models.md](generative-models.md)), **ML interatomic potentials / universal
  force fields** (NequIP/MACE-class, near-DFT accuracy at MD speed), weather/climate (GraphCast/neural weather
  models now competitive with numerical NWP), **physics-informed and operator learning** (PINNs, Fourier/DeepO
  neural operators for PDEs), and ML-accelerated simulation/discovery.
- **Domain-specific rigor (stricter than typical ML):** physical constraints and symmetries belong in the
  architecture (equivariance, conservation laws); **out-of-distribution generalization is the whole point**
  (predicting *novel* molecules/materials) → use scaffold/structural/temporal splits, not random (random splits
  hugely overestimate — the [graph-geometric.md](graph-geometric.md) lesson); uncertainty quantification matters
  for deciding which experiment to run next (active learning, see [probabilistic-ml.md](probabilistic-ml.md));
  validate against held-out experiments/simulations and respect known physics. Domain experts must be in the
  loop — a metric win that violates physics is a bug.

## 9. Cross-domain reach-for summary

| Domain | Default first move | Don't-skip gotcha |
|---|---|---|
| Vision | Pretrained backbone (DINOv2/ViT/ConvNeXt) + fine-tune/probe | Shortcut learning; label-preserving augmentation |
| NLP | LLM (prompt/LoRA/RAG); GBM/encoder baseline for narrow tasks | Contamination; tokenization; per-language eval |
| Speech/Audio | Whisper/wav2vec2 + fine-tune | Speaker leakage (split by speaker); acoustic realism |
| Time series | Seasonal-naive + GBM on lags; foundation model for zero-shot | Temporal split + backtesting; look-ahead bias |
| Tabular | LightGBM/XGBoost/CatBoost (+ AutoGluon) | Leakage audit; categorical encoding in-fold |
| Recommenders | Two-tower retrieve + rank | Temporal split; A/B test; exposure bias |
| Multimodal/VLM | Strong open VLM + instruction tune | Grounding (uses the image?); hallucination |
| Scientific ML | Equivariant/operator model with physics priors | Scaffold/structural split; UQ; respect physics |

**Canonical references (entry points):** ImageNet/ConvNeXt/ViT/DINOv2/SAM (vision); the LLM canon in
[transformers-llms.md](transformers-llms.md) (NLP); Whisper, wav2vec 2.0 (speech); Makridakis M4/M5 competitions
& Chronos/TimesFM (forecasting); Grinsztajn et al. 2022 (tabular); two-tower/DLRM & off-policy evaluation
(recsys); Qwen-VL/SigLIP (multimodal); AlphaFold, NequIP/MACE, GraphCast, Fourier Neural Operator (sci-ML).
