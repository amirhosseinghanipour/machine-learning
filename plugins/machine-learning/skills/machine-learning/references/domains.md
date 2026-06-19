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

The fastest-moving applied area in 2026: the field has reorganized around **discrete audio tokens + (codec)
language models** and **SSL encoders**, with **flow matching** now the default for high-fidelity synthesis.

- **ASR (speech-to-text):** the landscape has moved past Whisper. **Whisper large-v3 / turbo** is still the
  robust multilingual default and a fine baseline, but on the **Open ASR Leaderboard** it is no longer SOTA:
  **NVIDIA Canary-Qwen** (a Speech-Augmented LM = FastConformer encoder + frozen LLM decoder) leads English
  WER (~5.6%) trained on an order of magnitude less data, with **Parakeet-TDT** (CTC/TDT decoder) the
  speed/throughput champion (RTFx in the thousands) at slightly higher WER; **IBM Granite Speech** and
  **Qwen3-ASR** are also sub-Whisper on English. Architecture rule of thumb: **Conformer/FastConformer encoder
  + attention decoder** = best accuracy but slow; **CTC/TDT decoders** = far faster, slightly worse, streamable.
  Backbones often start from SSL encoders (wav2vec 2.0 / HuBERT / **WavLM** — WavLM is the strongest general
  speech encoder, **w2v-BERT 2.0** for massively multilingual). Metric: **WER/CER**, but **text normalization
  dominates** — whisper-normalizer vs. NeMo vs. raw can swing WER by points; **fix and report the normalizer**,
  and for long-form report a segmentation-robust metric. Watch **hallucinated transcripts on silence/noise**
  (a known Whisper failure) and **language-confusion** in multilingual decoding.
- **Self-supervised audio (the encoders to know):** wav2vec 2.0/HuBERT (contrastive/masked-prediction),
  **WavLM** (adds denoising/overlap modeling — best for speaker/diarization tasks), **BEATs**/**EAT** (general
  audio events), **Whisper encoder** as a strong supervised feature extractor. Benchmark frozen features on
  **SUPERB / SUPERB-SG**. Default: don't train an audio model from scratch — probe or fine-tune one of these.
- **Neural audio codecs & tokens (the substrate for everything generative):** **RVQ** codecs **SoundStream →
  EnCodec → DAC** (Descript, best reconstruction/codebook use) produce *acoustic* tokens; **Mimi** (Kyutai,
  streaming, **semantic-distilled RVQ-1** from an SSL teacher) and **SpeechTokenizer/X-Codec(2)/WavTokenizer/
  DualCodec** add **semantic–acoustic disentanglement** and **low frame rates** (12.5–25 Hz, even single
  codebook) so an LM can model speech cheaply. Key axes when choosing a codec: **frame rate, #codebooks,
  semantic vs. acoustic content, streaming capability, reconstruction (SI-SDR / ViSQOL / mel-distance)**.
- **TTS / audio generation (near-human, two dominant paradigms):**
  - **Codec/neural LMs:** **VALL-E(2)**-style — autoregressive over codec tokens, zero-shot voice cloning from a
    short prompt; descendants and open systems (**XTTS, Fish-Speech, CosyVoice/CosyVoice2, Kokoro** for
    lightweight) are production-grade.
  - **Flow-matching / diffusion, non-autoregressive:** **F5-TTS** (DiT + conditional flow matching, no
    duration model, fast and high-quality), **E2/Voicebox/Matcha-TTS**, **NaturalSpeech 3** — now the
    quality/controllability frontier; flow matching also powers many codec vocoders.
  - **Full-duplex / spoken dialogue:** **Moshi** (Kyutai) and speech-LLM stacks integrate ASR+TTS+LM for
    real-time conversation. Voice cloning raises **consent/deepfake** concerns — watermark, gate, and red-team
    (codec-synthesized-speech deepfake detection is its own benchmark). Metrics: **MOS / CMOS** (human, the gold
    standard) plus **UTMOS/DNSMOS** (predicted MOS — convenient but gameable), intelligibility via **ASR-WER**,
    **speaker-similarity (SIM)** via a verification embedding, **FAD** (choose the embedding deliberately — VGGish
    is dated; CLAP/PANN-based FAD is more reliable).
- **Other tasks:** **speaker diarization** — end-to-end neural (**NVIDIA Sortformer/Sortformer-v2** streaming,
  EEND) now rivals/beats the classic **pyannote 3.1** clustering pipeline; hybrids (DiariZen = WavLM embeddings +
  pyannote clustering) are strong. Metric **DER** (with collar + overlap handling stated). Also speaker
  verification (EER/minDCF), keyword spotting, music/audio tagging, **source separation** (SI-SDRi; SepFormer/
  TF-GridNet), sound event detection. **Audio-text / understanding:** CLAP embeddings; audio-LLMs (Qwen2-Audio,
  SALMONN, Audio Flamingo) for captioning/QA.
- **Gotchas (speech-specific, high-yield):** features are **log-mel spectrograms** (state n_fft/hop/n_mels and
  keep them identical train↔infer) + **SpecAugment**; **speaker leakage is the #1 trap** — the *same speaker*
  (or recording session/channel) across train/test inflates everything → **split by speaker, and ideally by
  session/device**; **variable-length batching** (bucket by length; pad and **mask** so padding doesn't enter
  the loss/attention); **sample-rate and codec consistency** (resample to a single rate; MP3/Opus artifacts
  differ from the training distribution); **noise/reverb robustness** and **far-field vs. close-talk** mismatch;
  **channel/microphone bias** as a confound; for generative models, **prosody/duration** evaluation needs more
  than WER. Evaluate on realistic acoustic conditions, not just clean read speech (LibriSpeech-clean flatters).

## 4. Time Series & Forecasting

- **Strong simple baselines are essential and often win:** naive/seasonal-naive (predict last value / last
  season), exponential smoothing (ETS), ARIMA, **and gradient-boosted trees on lag/calendar features (often the
  practical winner)**. Always beat these before claiming a deep model helps — many "deep forecasting beats
  classical" claims fail under honest baselines (the M-competitions are sobering).
- **Deep/Modern:** DeepAR (probabilistic RNN), N-BEATS/N-HiTS, **PatchTST** (patching + channel-independence —
  a strong, simple transformer baseline), and **time-series foundation models** (pretrained, zero-shot):
  **Chronos/Chronos-2** (tokenize-and-quantize, T5-style; Chronos-2 adds multivariate/covariate support),
  **TimesFM(-2.5)** (decoder-only, patched), **Moirai/Moirai-MoE** (any-variate attention + MoE),
  **MOMENT, Time-MoE**, and even **TabPFN-TS** (tabular in-context model, surprisingly strong zero-shot). They
  are genuinely useful for **cold-start/zero-shot** but **a tuned local model or GBM on lags often still wins**
  when you have history — benchmark, don't assume. A sobering result: on broad benchmarks several TSFMs still
  fail to beat **AutoETS/AutoTheta** at some frequencies. SSM/Mamba models suit very long sequences.
- **Evaluation:** **strictly temporal splits** + rolling-origin/backtesting CV — a random split leaks the future
  and is the #1 time-series error. **Beware TSFM contamination** — public benchmarks (ETT, traffic, M-series)
  may be in the pretraining corpus, so "zero-shot" wins can be recall; prefer fresh/private series or the
  contamination-aware **GIFT-Eval**. Metrics: MAE/RMSE, **MASE** (scale-free, vs. naive — the M-competition
  standard), sMAPE, **pinball/quantile loss + CRPS / coverage** for probabilistic forecasts (forecast
  *distributions* and check calibration, not just points). Report at the relevant horizon and reconcile
  hierarchies.
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

- **Where ML is transforming science:** protein/biomolecular structure (**AlphaFold 3** and open re-impls like
  **Boltz-1/2, Chai-1** — a **diffusion** head over a Pairformer, predicting *complexes* with ligands/nucleic
  acids/ions, not just single chains; equivariant priors — see [graph-geometric.md](graph-geometric.md)),
  protein/molecule **design** (RFdiffusion, ESM-class protein LMs), molecular property prediction & generation
  (flow matching/diffusion over molecules, see [generative-models.md](generative-models.md)), **ML interatomic
  potentials / universal force fields** — now **foundation models**: **MACE-MP-0/MACE-MH-1, MatterSim, ORB,
  SevenNet, CHGNet, eqV2/Fairchem**, near-DFT accuracy at MD speed and broad chemistry coverage (paired with
  large DFT corpora like the Materials Project / OMat24; **GNoME** for materials discovery). Weather/climate
  (**GraphCast, GenCast (diffusion ensembles), Aurora, FourCastNet**) now competitive with or beating numerical
  NWP at a fraction of the cost. **Physics-informed and operator learning** — **PINNs** (great for inverse
  problems/PDE-constrained fitting, but notoriously hard to optimize: stiff multi-objective loss balancing,
  spectral bias, use curriculum/causal weighting/PirateNets) and **neural operators** (**FNO**, DeepONet,
  and graph/transformer operators) that learn *mappings between function spaces* and generalize across
  resolutions/discretizations. Plus ML-accelerated simulation, surrogate modeling, and autonomous experiment
  loops.
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
| Speech/Audio | ASR: Whisper/Canary or WavLM-probe; TTS: F5-TTS or codec-LM | Speaker/session leakage (split by speaker); normalizer-defined WER; acoustic realism |
| Time series | Seasonal-naive + GBM on lags; foundation model for zero-shot | Temporal split + backtesting; look-ahead bias |
| Tabular | LightGBM/XGBoost/CatBoost (+ AutoGluon) | Leakage audit; categorical encoding in-fold |
| Recommenders | Two-tower retrieve + rank | Temporal split; A/B test; exposure bias |
| Multimodal/VLM | Strong open VLM + instruction tune | Grounding (uses the image?); hallucination |
| Scientific ML | Equivariant/operator model with physics priors | Scaffold/structural split; UQ; respect physics |

**Canonical references (entry points):** ImageNet/ConvNeXt/ViT/DINOv2/SAM (vision); the LLM canon in
[transformers-llms.md](transformers-llms.md) (NLP); Whisper, wav2vec 2.0/HuBERT/WavLM, EnCodec/DAC/Mimi,
VALL-E 2, F5-TTS, Open ASR Leaderboard, SUPERB (speech/audio); Makridakis M4/M5 competitions, Chronos/TimesFM/
Moirai & GIFT-Eval (forecasting); Grinsztajn et al. 2022 / TabPFN (tabular); two-tower/DLRM & off-policy
evaluation (recsys); Qwen-VL/SigLIP, MMMU (multimodal); AlphaFold 3/Boltz, NequIP/MACE & MACE-MP, GraphCast/
GenCast, Fourier Neural Operator/DeepONet (sci-ML).
