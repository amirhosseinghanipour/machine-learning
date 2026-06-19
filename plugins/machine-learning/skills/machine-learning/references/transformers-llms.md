# Transformers and Large Language Models

The dominant architecture and the full LLM lifecycle, current to 2026. Generative theory (diffusion/flow) is
in [generative-models.md](generative-models.md); RL details (PPO/GRPO) in
[reinforcement-learning.md](reinforcement-learning.md); scaling/distributed in
[engineering-scale.md](engineering-scale.md).

---

## 1. Attention and the transformer block

**Scaled dot-product attention:** $\text{Attn}(Q,K,V)=\text{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}\big)V$.
Each token forms a query, matches against all keys (content-based addressing), and reads a weighted sum of
values. The $\sqrt{d_k}$ keeps logits from saturating softmax. **Multi-head** attention runs $h$ attentions in
parallel subspaces and concatenates — different heads specialize. Cost is $O(L^2 d)$ in sequence length $L$ —
the central scaling pain point.

**The block** (modern pre-norm form): `x = x + Attn(RMSNorm(x))`; `x = x + MLP(RMSNorm(x))`. The MLP is
typically **SwiGLU** with hidden dim ~$\tfrac{8}{3}d$ (chosen to roughly match a $4d$ ReLU MLP's parameter
count after the gating doubles the projections). Decoder LMs use **causal masking** (a token attends only to
itself and earlier tokens). This residual+norm structure is why deep transformers train (see
[deep-learning.md](deep-learning.md) §2). Modern stabilizers that are now near-standard at scale: **QK-norm**
(RMSNorm on Q and K before the dot product — kills attention-logit blowups), **z-loss** (a small penalty on the
log-partition of the output softmax), **logit soft-capping** (Gemma 2), and occasionally **sandwich/peri-norm**
(norm before *and* after the sublayer) for very deep stacks. Pre-norm is the default; the residual stream is
the model's "working memory" and norms read/write to it.

**Attention-variant zoo:** Multi-Head (MHA) → **Multi-Query (MQA)** (one shared K/V head) and **Grouped-Query
(GQA)** (K/V shared within groups of query heads) shrink the KV cache (GQA is the modern default — most of
MHA's quality at a fraction of the cache; typical 4–8 KV groups). **Multi-head Latent Attention (MLA)**
(DeepSeek-V2/V3) compresses K/V into a low-rank latent that is cached instead of full K/V, cutting cache
~10–20× while *beating* GQA quality; it composes with a decoupled-RoPE trick so positions survive the
compression. The trade-offs: MQA is cheapest but can lose quality and is harder to tensor-parallelize cleanly;
GQA is the safe default; MLA is the frontier choice when KV memory is the binding constraint and you control
pretraining. **Attention sinks** — keeping the first few tokens (or a dedicated learned sink token) always in
the window — stabilize streaming/sliding-window attention and are now standard in long-context serving.

## 2. Positional information

Self-attention is permutation-equivariant, so position must be injected:
- **Absolute** (sinusoidal or learned) — original transformer; poor length extrapolation.
- **RoPE (Rotary)** — rotates Q/K by position-dependent angles ($\theta_i = b^{-2i/d}$, base $b$ usually
  10000); encodes *relative* position, the **current standard** in LLMs. Low-frequency dimensions carry
  coarse/long-range position, high-frequency dimensions carry local order — this split is exactly what the
  scaling methods in §7 exploit. The base $b$ is itself a lever: **ABF** (Adjusted Base Frequency, raising $b$
  to e.g. $5\times10^5$–$10^7$) during continued pretraining is the dominant way to natively extend context
  (used in Llama-3, Qwen2/3).
- **ALiBi** — linear distance bias on attention logits; cheap, extrapolates to longer sequences, but largely
  superseded by RoPE+scaling in frontier LLMs.
- **NoPE / partial-RoPE.** Decoder-only models can learn position implicitly from the causal mask, so dropping
  positional encodings ("NoPE") or applying RoPE to only a fraction of dimensions can *improve* length
  generalization — an active 2025 finding; hybrid SSM/attention stacks often use NoPE on the attention layers.
- For **long context**, RoPE scaling (position interpolation, YaRN, LongRoPE) plus training on long sequences
  is the mainstream path; see §7.

## 3. Tokenization

The model sees token IDs, not text — tokenization choices ripple everywhere.
- **BPE / byte-level BPE** (GPT family) and **Unigram/SentencePiece** (T5, many multilingual) are standard;
  WordPiece (BERT) is older. Byte-level fallback guarantees no OOV.
- **Gotchas:** number handling (left-to-right digit splitting, or per-digit tokenization as in Llama-3, helps
  arithmetic), whitespace/leading-space semantics, multilingual fairness (some languages cost 2–4× more tokens
  → higher cost and worse quality), code/indentation, and the fact that **perplexity/bits-per-byte is only
  comparable across identical tokenizers** (always report bits-per-byte to compare across vocabularies).
  Vocabularies have grown (32k → 128k–256k) because larger vocab amortizes the embedding cost and shortens
  sequences; tie or untie input/output embeddings deliberately.
- **Tokenizer-free / dynamic-patch models** are the most credible escape from BPE's brittleness (the 2025
  frontier). **Byte Latent Transformer (BLT, Meta)** segments raw bytes into *patches* whose boundaries are set
  by a small byte-level model's **next-byte entropy** (predictable spans → long patches; high-entropy onsets →
  short patches), so compute is allocated dynamically; a local encoder/decoder brackets a large global latent
  transformer. BLT matches BPE-tokenized Llama-3 quality at 8B/4T-byte scale with better robustness to noise
  and better character-level tasks. **H-Net** learns the chunking end-to-end and hierarchically. These remain
  research-grade; BPE/BBPE is still the production default.
- Always confirm the tokenizer matches the model checkpoint; mismatches silently destroy performance. Adding
  tokens (special/control tokens) requires resizing and re-initializing the embedding rows.

## 4. The LLM lifecycle (the big picture)

```
Pretraining (self-supervised next-token prediction on trillions of tokens)
   → a base model: broad knowledge + capabilities, but not aligned to instructions
Post-training:
   1. SFT (supervised fine-tuning on instruction/response data) → follows instructions
   2. Preference optimization (DPO / SimPO / KTO) → aligned to human/AI preferences
   3. RL with verifiable rewards (GRPO / DAPO / RLVR) → reasoning, correctness on checkable tasks
Inference-time: prompting, decoding, tool use, RAG, agents, test-time compute/reasoning
```

Each stage is its own discipline; the sections below go deep. Most *applied* work happens at SFT/preference/
inference; pretraining is for those with large compute.

## 5. Pretraining and scaling laws

- **Objective:** autoregressive next-token cross-entropy (decoder-only is dominant; encoder-only BERT-style
  masked LM for embeddings/understanding; encoder-decoder T5-style for some seq2seq).
- **Scaling laws (plan in FLOPs, not epochs):** loss falls as a power law in compute, parameters, and data,
  $L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$. **Chinchilla** (Hoffmann et al. 2022): for a dense model at compute
  budget $C\approx 6ND$ ($N$ params, $D$ tokens), compute-optimal is roughly **20 tokens per parameter** —
  many earlier models (GPT-3, Gopher) were badly undertrained. Caveats that matter in 2026:
  - **Inference-aware / overtraining.** If you will serve the model a lot, training *past* Chinchilla-optimal
    (a smaller model on far more tokens) minimizes total train+serve cost — most deployed models (Llama-3 8B at
    ~15T tokens ≈ 1800 tok/param) are overtrained by 50–100× the Chinchilla ratio on purpose. The crossover is
    set by expected inference volume.
  - **Data-constrained** (Muennighoff et al.): when unique tokens run out, repeating data helps but with sharp
    diminishing returns — up to ~4 epochs is "nearly free," value decays to near-zero by ~16 epochs; beyond
    that, add parameters or synthetic/augmented data instead. The effective dataset size saturates.
  - **MoE scaling** adds sparsity to the law: loss improves with total params and expert count even at fixed
    *active* FLOPs, but the compute-optimal token/active-param ratio shifts and granularity (experts per layer)
    is its own axis. Note also **scaling laws for precision** (Kumar et al. 2025): low-precision training/
    inference effectively *reduces* parameter count, so quantization interacts with the optimal $N$/$D$.
  - Caveat on the Chinchilla fit itself: 2025 reanalyses found systematic biases in the IsoFLOP parabola fits —
    treat "20×" as an order-of-magnitude guide, not a constant, and re-fit for your setup.
- **Emergence vs. smoothness.** Many "emergent" jumps are artifacts of discontinuous metrics (exact-match);
  under continuous metrics capability scales smoothly. Use this when forecasting — don't over-index on a single
  benchmark cliff.
- **Data is the lever:** quality, diversity, dedup, and decontamination dominate. Aggressive quality filtering
  (model-based classifiers, e.g. FineWeb-Edu), dedup (MinHash/exact-substring), and curriculum (e.g. anneal on
  high-quality/code/math data at the end) matter more than minor architecture tweaks. See [data.md](data.md).
- **Stability at scale:** bf16, careful init/LR/warmup, gradient clipping, z-loss/QK-norm, μP (maximal-update
  parameterization) for zero-shot HP transfer from small proxies, WSD/warmup-stable-decay or cosine schedules,
  and monitoring for loss spikes (skip-the-batch / rollback recipes). Training is engineering as much as
  science ([engineering-scale.md](engineering-scale.md)).

## 6. Mixture of Experts (MoE)

Replace (some) dense MLPs with $E$ experts; a router (a linear layer + softmax/sigmoid) sends each token to
top-$k$ experts. **Decouples total parameters from per-token compute** — e.g., DeepSeek-V3: 671B total / 37B
active; Qwen3-235B-A22B: 235B/22B; Llama-4, GPT-OSS, Mixtral, Kimi-K2 all sparse. MoE is now the default
*frontier* architecture above ~30B.
- **Why:** more capacity/knowledge at fixed inference FLOPs; memorization scales with total params, reasoning
  with active params.
- **Modern design (2025 consensus):** **fine-grained experts** (many small experts, e.g. 128–256, with high
  total $k$ ≈ 8) outperform few large ones — more combinatorial routing flexibility per FLOP. **Shared experts**
  (1–2 experts that *every* token uses, DeepSeek/Llama-4) absorb common computation so routed experts
  specialize — though several strong 2025 models (Qwen3, OLMoE, GPT-OSS) drop them, so treat them as optional.
- **Load balancing — the central difficulty.** Routers collapse toward a few experts. The old fix is an
  **auxiliary load-balance loss** (penalize uneven dispatch) which fights the LM objective. The 2025 default is
  DeepSeek's **auxiliary-loss-free balancing**: add a per-expert *bias* to the routing logits used only for the
  top-$k$ selection (not the gating weight), and nudge each bias up/down based on recent load. This balances
  with zero interference gradient — cleaner and better-performing. Often paired with a tiny sequence-level
  balance loss as a safety net.
- **Other challenges:** routing instability/expert collapse, all-to-all communication (expert parallelism is a
  comms problem — see [engineering-scale.md](engineering-scale.md)), memory (all experts resident even though
  few fire), and token-dropping under capacity limits. Inference uses **expert parallelism**; serving is
  bottlenecked by the all-to-all and by keeping experts hot.
- **When:** MoE shines at large scale where capacity is the bottleneck and you can afford the memory/comms; for
  small models, edge deployment, or simple tasks, dense is simpler and often better per-*total*-parameter.

## 7. Efficient attention, long context, and the KV cache

- **The bottleneck:** at inference, the **KV cache** (stored keys/values for all past tokens) grows linearly
  with context and dominates memory; attention is $O(L^2)$. Long context is mostly a *systems* problem.
- **FlashAttention** (and v2/v3, plus PyTorch's `varlen_attn`/SDPA) computes exact attention in a memory-
  efficient, IO-aware tiled way — use it always; it's exact, not approximate.
- **Sparse / linear / sliding-window attention** (Longformer, sliding window + global/sink tokens,
  linear-attention variants). Frontier models often **interleave** local sliding-window layers with sparse
  full-attention layers (Gemma 2/3, Llama-4, Cohere) so most layers are cheap while a few preserve global
  recall. **Native sparse attention** (DeepSeek NSA, trainable block-sparse) and **selective/top-k KV** are the
  2025 direction — make sparsity learned and hardware-aligned rather than fixed.
- **KV-cache reduction:** GQA/MQA/MLA (§1), **FP8 KV cache** (e4m3 — roughly halves cache memory with
  near-baseline accuracy, the production default in vLLM 2026), cross-layer KV sharing, and cache eviction
  (StreamingLLM-style: keep attention-sink tokens + a recent window; H2O/SnapKV keep high-attention tokens).
- **Length extension recipes (the practical path):** native extension via **ABF** (raise the RoPE base) during
  continued pretraining on long sequences; test-time extrapolation via **YaRN** (NTK-by-parts frequency-aware
  interpolation + attention-temperature scaling — the strongest training-free/short-finetune method) or
  **LongRoPE** (per-dimension search, reaches 2M+). The standard Qwen-style recipe is ABF during long-context
  continued-pretraining + YaRN at inference. Frontier context windows are now 128k–1M+ (Gemini 1M+, Grok-4-fast
  2M). **Verify with reasoning, not perplexity:** needle-in-a-haystack is necessary but easy; use RULER,
  multi-needle, variable-tracking, and long-context QA — many models pass NIAH yet collapse on tasks needing
  information buried mid-context (the "lost in the middle" effect).

## 8. State-space models and transformer alternatives

A live research frontier (2025–2026): linear-time sequence models that avoid the $O(L^2)$ cost.
- **SSMs / Mamba.** Structured state-space models (S4 → **Mamba/Mamba-2**, Gu & Dao) maintain a fixed-size
  recurrent state, scaling **linearly** in length with a parallel scan for training. Mamba-2 unifies SSMs and
  attention (state-space duality). Strong on long sequences, audio, genomics; slightly behind FlashAttention
  transformers on some tasks and in raw training efficiency.
- **RWKV, RetNet, gated linear attention (GLA), xLSTM** — RNN-like $O(1)$-per-step inference with
  parallelizable training; the broader "linear attention with a gate/decay" family that Mamba-2's SSD framing
  unifies. The core limitation is **finite state**: fixed-size recurrent memory cannot losslessly recall
  arbitrary past tokens (associative recall / exact copy), which is why pure SSMs lag on in-context retrieval.
- **Hybrids are the practical SOTA:** interleave a *small* fraction of full-attention layers (typically
  ~7–12% of depth — for exact recall/copying/long-context retrieval) with many Mamba-2/linear layers, often +
  MoE. Production families: **Jamba** (Mamba+attention+MoE, 52B/12B active), **Nemotron-H** and **Nemotron
  Nano 2** (NVIDIA — up to ~92% Mamba, several × faster long-context decode at matched accuracy), **Zamba2**,
  **Falcon-H1**, **MiniMax-01** (lightning attention). These match transformer quality with much smaller KV
  cache and faster long-context decode. Default to a standard transformer unless long-sequence throughput/
  memory is the binding constraint; then evaluate a hybrid — and keep enough attention layers to preserve
  recall.

## 9. Post-training in depth

(RL algorithm mechanics — PPO, GRPO — in [reinforcement-learning.md](reinforcement-learning.md) §9.)

- **SFT.** Supervised fine-tuning (next-token CE on the *response* tokens only — mask the prompt) on
  high-quality (instruction, response) pairs. Establishes instruction-following, format, and chat template.
  Data quality ≫ quantity; a few thousand excellent examples can beat hundreds of thousands of mediocre ones
  (the "less is more"/LIMA observation). **Sample packing** (concatenate to fill the context, with attention
  masks to prevent cross-contamination) and careful chat-template/EOS handling are the usual footguns. Often
  done with **LoRA/QLoRA** (see [representation-learning.md](representation-learning.md)) to save memory; full
  SFT for the biggest quality gains. Watch loss-masking and that you train on assistant turns only.
- **Preference optimization** (align to what humans/AI prefer):
  - **RLHF (PPO)** — train a reward model on preference pairs (Bradley–Terry), then optimize the policy with
    PPO against it plus a **KL penalty to the reference** model (prevents reward hacking / drift). Powerful,
    still used at frontier labs, but complex, memory-heavy (4 models in memory), and finicky.
  - **DPO** — *Direct* Preference Optimization reparameterizes the RLHF objective so you train directly on
    preference pairs with a simple binary-classification-style loss, **no reward model, no RL loop**. The
    pragmatic default for offline preference alignment. Knobs: $\beta$ (KL strength, typically 0.01–0.1), and
    *guard against the likelihood of chosen responses dropping* — DPO can decrease both chosen and rejected
    logprobs; mitigations are a small SFT/NLL term on chosen (DPO+NLL, used in Llama-3) or higher $\beta$.
  - **Variants and when to reach for them:** **IPO** (bounded loss — fixes DPO overfitting to near-deterministic
    preferences); **SimPO** (reference-free, length-normalized implicit reward + target margin — no reference
    model in memory, strong on AlpacaEval/Arena-Hard but sensitive to length/$\gamma$); **KTO** (uses *unpaired*
    thumbs-up/down via a prospect-theory value function — collect-cheap, robust to imbalance); **ORPO** (folds
    preference into SFT in one stage via an odds-ratio penalty — no reference model, no separate SFT). **Caveat
    from 2025/26 controlled studies:** under matched tuning these methods often differ less than headline
    numbers suggest, and *rankings invert with scale* — pick by data shape (paired vs. binary, online vs.
    offline) and memory budget, and always tune $\beta$/LR per method rather than trusting a leaderboard.
- **RL with Verifiable Rewards (RLVR)** — the 2025 shift for *reasoning*: when correctness is checkable (math
  answer via a parser/symbolic check, unit tests, a verifier, format/length constraints), use the programmatic
  reward instead of a learned reward model, eliminating reward-hacking of a proxy. **GRPO** (Group Relative
  Policy Optimization, DeepSeek) drops PPO's value network: sample a *group* of $G$ completions per prompt and
  use the group's mean/std-normalized reward as each sample's advantage — cheaper (no critic) and stable.
  DeepSeek-R1-Zero showed *pure* RLVR from a base model elicits emergent long-chain reasoning and "aha"
  self-correction with no SFT; R1 added a small cold-start SFT for readability.
  - **GRPO's known failure modes and fixes** (the active 2025–26 literature): **length bias** (response-level
    normalization rewards verbosity) → **Dr.GRPO** removes the length/std normalization; **DAPO** uses
    token-level loss, **decoupled/asymmetric clipping** ("clip-higher" to preserve exploration), **dynamic
    sampling** (drop prompts where all completions are right or all wrong — zero gradient), and often **no KL
    term**; **entropy collapse** (policy goes deterministic, exploration dies) is the recurring instability —
    monitor entropy and pass@k, not just pass@1. Many strong recipes now drop the reference-KL entirely for
    RLVR. RLVR mostly *sharpens* abilities already in the base model (raises pass@1 toward base pass@k) rather
    than adding wholly new ones — a useful mental model and a real ceiling.
- **A standard modern pipeline:** base → SFT (cold-start, incl. reasoning traces) → (DPO/preference for general
  alignment, safety, style) → (GRPO/RLVR for verifiable reasoning) → optional distillation of a big reasoner's
  traces into smaller models (often *more* effective than running RL on the small model directly). Don't
  over-engineer: for most applied alignment, SFT + DPO is enough; add RLVR only when you have verifiable rewards
  and reasoning is the goal. **Reasoning-model caveats:** long CoT costs latency/tokens; calibrate a thinking
  budget, and beware that visible CoT is not a faithful explanation of the computation.

## 10. Inference: decoding, reasoning, and test-time compute

- **Decoding:** greedy/beam (deterministic, for verifiable tasks), **temperature + top-p (nucleus) / top-k /
  min-p** sampling (open-ended). Lower temperature for reasoning/code, higher for creativity. Beam search hurts
  open-ended diversity.
- **Test-time compute / reasoning:** chain-of-thought, self-consistency (sample many, majority-vote),
  best-of-$n$ with a verifier/reward model, and trained "thinking" (long reasoning traces from RLVR). Spending
  more compute at inference can substitute for a larger model — a key 2025–2026 axis, but with **diminishing
  returns** and a compute-optimal allocation: for easy problems a small model + sampling wins; hard problems
  need the larger model. Process reward models (score steps) help verification-based scaling.
- **Inference is two regimes:** **prefill** (process the prompt — compute-bound, parallel) and **decode**
  (one token at a time — memory-bandwidth-bound, dominated by reading weights + KV cache). This split drives
  serving design: **prefill/decode disaggregation**, **chunked prefill**, and **prefix caching** (reuse KV for
  shared system prompts — huge win for agents/RAG).
- **Speed:** **speculative decoding** (a cheap drafter proposes $k$ tokens, the target verifies in one
  parallel pass — exact same output distribution; 2–3×). Modern drafters are **self-speculative**: **EAGLE-3**
  (feature-level autoregression on the target's own hidden states, with training-time test alignment) and
  **Medusa** (multiple decoding heads) are the strong defaults; **MTP** (multi-token prediction) heads trained
  in pretraining (DeepSeek-V3) double as drafters. **Continuous/in-flight batching** (vLLM/TGI/SGLang),
  **quantization** (FP8 weights+activations near-lossless and hardware-accelerated on Hopper/Blackwell; INT4
  weight-only via GPTQ/AWQ for memory; **FP8 KV cache** standard), and PagedAttention KV paging. See
  [engineering-scale.md](engineering-scale.md).
- **Structured output:** constrained decoding / grammars (JSON schema, regex via outlines/XGrammar) for
  reliable tool/agent I/O — masks invalid tokens at each step so output always parses. Cheap and high-leverage
  for agents; mind that over-constraining can degrade quality on free-form reasoning.

## 11. In-context learning, prompting, RAG, and agents

The application layer above the model — in-context learning, prompting discipline, Retrieval-Augmented
Generation, tool-use and multi-agent systems, context engineering, MCP, and end-to-end agent
evaluation/security — is covered in depth in **[agents.md](agents.md)**. Read it before building any
RAG pipeline or agent; the short version is: prefer the least-agentic thing that works, evaluate
retrieval and generation separately, bound every loop, treat external/tool text as adversarial, and
report agent success with variance over multiple rollouts.

## 12. Evaluating LLMs (see [evaluation-statistics.md](evaluation-statistics.md) for rigor)

- **Contamination is the central threat:** public static benchmarks leak into pretraining → recall masquerading
  as skill. Prefer **frequently-refreshed / held-out** benchmarks (LiveBench, LiveCodeBench, FrontierMath,
  MMLU-Pro) and report contamination checks (Min-K%, n-gram overlap, ConStat).
- **Capability evals:** task-specific with verifiable answers where possible (math/code with execution).
- **LLM-as-judge:** scalable but biased (position, verbosity, self-preference, style over substance) — calibrate
  against human labels, randomize order, use rubrics, and never let a model judge its own family unblinded.
  Prefer **pairwise** comparison over absolute scoring (more reliable), and report judge–human agreement.
  Arena-style human preference (Elo) is the gold standard for open-ended quality but is slow and gameable by
  style.
- **Safety/alignment evals:** refusal, jailbreak robustness, bias, honesty/calibration — see
  [interpretability-safety.md](interpretability-safety.md).
- Report variance and significance; LLM eval deltas are often within noise.

## 13. Practical defaults

| Decision | Default (2026) |
|---|---|
| Architecture | Pre-norm decoder transformer, RMSNorm + QK-norm, RoPE, SwiGLU, GQA (MLA if KV-bound) |
| Long-sequence efficiency critical | Evaluate attention+SSM hybrid (Mamba-2, ~10% attention layers) ± MoE |
| Want capacity at fixed inference FLOPs | MoE: fine-grained experts + aux-loss-free balancing (if memory/comms allow) |
| Token/param budget | Chinchilla ~20× as a floor; **overtrain** heavily if serving a lot |
| Extend context | ABF (raise RoPE base) + long-seq continued pretrain; YaRN at inference; verify with RULER not just NIAH |
| Align a model to instructions | SFT (LoRA/QLoRA, mask prompt) → DPO (tune $\beta$; guard chosen-logprob drop) |
| Improve verifiable reasoning | GRPO/RLVR (token-level loss, clip-higher, dynamic sampling; monitor entropy & pass@k) |
| Small reasoning model | Distill long-CoT traces from a big reasoner > RL on the small model |
| Ground on private/fresh data | Hybrid RAG (BM25+dense + reranker, prefix-cache the system prompt); evaluate retrieval separately |
| Serve efficiently | vLLM/SGLang, FP8 weights+KV, EAGLE-3/MTP speculative decoding, prefill/decode disaggregation, paged KV |
| Agents | Few well-described tools, constrained decoding, context engineering, MCP; eval end-to-end task success |
| Evaluate | Held-out/refreshed benchmarks + task-specific verifiable evals + variance |

**Canonical references:** Vaswani et al. 2017 (*Attention Is All You Need*); Brown et al. 2020 (GPT-3,
in-context learning); Hoffmann et al. 2022 (Chinchilla); Muennighoff et al. 2023 (data-constrained scaling);
Su et al. 2021 (RoPE); Peng et al. 2023 (YaRN); Dao et al. 2022 (FlashAttention); DeepSeek-AI 2024 (V3/MLA,
auxiliary-loss-free balancing); Pagnoni et al. 2024 (Byte Latent Transformer); Rafailov et al. 2023 (DPO);
Meng et al. 2024 (SimPO); Ethayarajh et al. 2024 (KTO); Hong et al. 2024 (ORPO); Shao et al. 2024 (GRPO) /
DeepSeek-R1 2025 (Nature 2025; RLVR); Yu et al. 2025 (DAPO); Liu et al. 2025 (Dr.GRPO); Gu & Dao 2023–24
(Mamba/Mamba-2 / SSD); Lieber et al. 2024 (Jamba); NVIDIA 2025 (Nemotron-H); Li et al. 2024 / EAGLE-3 2025
(speculative decoding); Kwon et al. 2023 (PagedAttention/vLLM); Lewis et al. 2020 (RAG); Wei et al. 2022
(chain-of-thought); Snell et al. 2024 (test-time compute scaling).
