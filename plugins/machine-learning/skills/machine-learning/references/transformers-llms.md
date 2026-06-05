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
typically **SwiGLU** with hidden dim ~$\tfrac{8}{3}d$. Decoder LMs use **causal masking** (a token attends
only to itself and earlier tokens). This residual+norm structure is why deep transformers train (see
[deep-learning.md](deep-learning.md) §2).

**Attention-variant zoo:** Multi-Head (MHA) → **Multi-Query (MQA)** and **Grouped-Query (GQA)** share K/V
across heads to shrink the KV cache (GQA is the modern default — most of MHA's quality at a fraction of the
cache). **Multi-head Latent Attention (MLA)** (DeepSeek) compresses KV into a latent for large cache savings.

## 2. Positional information

Self-attention is permutation-equivariant, so position must be injected:
- **Absolute** (sinusoidal or learned) — original transformer; poor length extrapolation.
- **RoPE (Rotary)** — rotates Q/K by position-dependent angles; encodes *relative* position, the **current
  standard** in LLMs, and extends to long context via interpolation/NTK-aware scaling/YaRN.
- **ALiBi** — linear distance bias on attention logits; cheap, extrapolates to longer sequences.
- For **long context**, RoPE scaling (position interpolation, YaRN) plus training on long sequences is the
  mainstream path; see §7.

## 3. Tokenization

The model sees token IDs, not text — tokenization choices ripple everywhere.
- **BPE / byte-level BPE** (GPT family) and **Unigram/SentencePiece** (T5, many multilingual) are standard;
  WordPiece (BERT) is older. Byte-level fallback guarantees no OOV.
- **Gotchas:** number handling (digit splitting affects arithmetic), whitespace/leading-space semantics,
  multilingual fairness (some languages cost far more tokens), code/indentation, and the fact that **perplexity
  is only comparable across identical tokenizers**. Tokenizer-free/byte-level models (e.g., byte-latent
  approaches) are an active research direction.
- Always confirm the tokenizer matches the model checkpoint; mismatches silently destroy performance.

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
- **Scaling laws (plan in FLOPs, not epochs):** loss falls as a power law in compute, parameters, and data.
  **Chinchilla** (Hoffmann et al. 2022): for a dense model at compute budget $C\approx 6ND$ ($N$ params, $D$
  tokens), compute-optimal is roughly **20 tokens per parameter** — many earlier models were badly
  undertrained. Caveats that matter in 2026: if you'll serve the model a lot, **overtrain** past
  Chinchilla-optimal (smaller model, more tokens → cheaper inference); **MoE** and **data-constrained**
  regimes shift the optimum (repeating data has diminishing returns; sparsity and expert count enter the law).
- **Data is the lever:** quality, diversity, dedup, and decontamination dominate. Filtering, dedup
  (MinHash/exact), and curriculum matter more than minor architecture tweaks. See [data.md](data.md).
- **Stability at scale:** bf16, careful init/LR/warmup, gradient clipping, z-loss/QK-norm, and monitoring for
  loss spikes. Training is engineering as much as science ([engineering-scale.md](engineering-scale.md)).

## 6. Mixture of Experts (MoE)

Replace (some) dense MLPs with $E$ experts; a router sends each token to top-$k$ experts (usually $k{=}1$–$2$).
**Decouples total parameters from per-token compute** — e.g., DeepSeek-V3: 671B total / 37B active per token.
- **Why:** more capacity/knowledge at fixed inference FLOPs.
- **Challenges:** load balancing (auxiliary loss or loss-free balancing), routing instability, expert
  collapse, all-to-all communication cost, and memory (all experts must be resident). Fine-grained + shared
  experts (DeepSeek) is a strong modern design.
- **When:** MoE shines at large scale where capacity is the bottleneck and you can afford the memory/comms; for
  small models or simple tasks, dense is simpler and often better per-parameter.

## 7. Efficient attention, long context, and the KV cache

- **The bottleneck:** at inference, the **KV cache** (stored keys/values for all past tokens) grows linearly
  with context and dominates memory; attention is $O(L^2)$. Long context is mostly a *systems* problem.
- **FlashAttention** (and v2/v3, plus PyTorch's `varlen_attn`/SDPA) computes exact attention in a memory-
  efficient, IO-aware tiled way — use it always; it's exact, not approximate.
- **Sparse / linear / sliding-window attention** (Longformer, sliding window + global tokens, linear-attention
  variants) trade exactness for sub-quadratic cost.
- **KV-cache reduction:** GQA/MQA/MLA (§1), quantized KV cache, paged attention (vLLM), and cache eviction.
- **Length extension:** RoPE interpolation/YaRN + continued training on long sequences; needle-in-a-haystack
  and long-context reasoning evals (not just perplexity) to verify it actually uses the context.

## 8. State-space models and transformer alternatives

A live research frontier (2025–2026): linear-time sequence models that avoid the $O(L^2)$ cost.
- **SSMs / Mamba.** Structured state-space models (S4 → **Mamba/Mamba-2**, Gu & Dao) maintain a fixed-size
  recurrent state, scaling **linearly** in length with a parallel scan for training. Mamba-2 unifies SSMs and
  attention (state-space duality). Strong on long sequences, audio, genomics; slightly behind FlashAttention
  transformers on some tasks and in raw training efficiency.
- **RWKV, RetNet** — RNN-like inference with parallelizable training.
- **Hybrids are the practical SOTA:** interleave a few attention layers (for exact recall/copying, which pure
  SSMs are weaker at) with many linear/SSM layers, often + MoE (e.g., Jamba). This is where much frontier
  efficiency work is going. Default to a standard transformer unless long-sequence efficiency is the binding
  constraint; then evaluate a hybrid.

## 9. Post-training in depth

(RL algorithm mechanics — PPO, GRPO — in [reinforcement-learning.md](reinforcement-learning.md) §9.)

- **SFT.** Supervised fine-tuning on high-quality (instruction, response) pairs. Establishes instruction-
  following and format. Data quality ≫ quantity; a few thousand excellent examples can beat hundreds of
  thousands of mediocre ones (the "less is more"/LIMA observation). Often done with **LoRA/QLoRA** (see
  [representation-learning.md](representation-learning.md)) to save memory.
- **Preference optimization** (align to what humans/AI prefer):
  - **RLHF (PPO)** — train a reward model on preference pairs, then optimize the policy against it with PPO and
    a **KL penalty to the reference** model (prevents reward hacking / drift). Powerful but complex and
    finicky.
  - **DPO** — *Direct* Preference Optimization reparameterizes the RLHF objective so you train directly on
    preference pairs with a simple classification-style loss, **no reward model, no RL loop**. The pragmatic
    default for preference alignment. Variants: **IPO** (fixes DPO overfitting to deterministic preferences),
    **SimPO** (reference-free), **KTO** (uses unpaired thumbs-up/down — cheaper to collect), **ORPO** (folds
    preference into SFT).
- **RL with Verifiable Rewards (RLVR)** — the 2025 shift for *reasoning*: when correctness is checkable
  (math answer, unit tests, a verifier), use the programmatic reward instead of a learned reward model,
  eliminating reward-hacking of a proxy. **GRPO** (Group Relative Policy Optimization, DeepSeek) drops PPO's
  value network and computes advantages from the **relative** scores of a *group* of sampled completions to the
  same prompt — cheaper and stable. **DAPO** and other refinements address length bias, entropy collapse, and
  clipping. DeepSeek-R1 showed pure RLVR can elicit emergent long-chain reasoning.
- **A standard modern pipeline:** base → SFT → (DPO for general alignment) → (GRPO/RLVR for verifiable
  reasoning). Don't over-engineer: for most applied alignment, SFT + DPO is enough; add RLVR only when you
  have verifiable rewards and reasoning is the goal.

## 10. Inference: decoding, reasoning, and test-time compute

- **Decoding:** greedy/beam (deterministic, for verifiable tasks), **temperature + top-p (nucleus) / top-k /
  min-p** sampling (open-ended). Lower temperature for reasoning/code, higher for creativity. Beam search hurts
  open-ended diversity.
- **Test-time compute / reasoning:** chain-of-thought, self-consistency (sample many, majority-vote),
  best-of-$n$ with a verifier/reward model, and trained "thinking" (long reasoning traces from RLVR). Spending
  more compute at inference can substitute for a larger model — a key 2025–2026 axis.
- **Speed:** speculative decoding (draft model proposes, target verifies), continuous batching (vLLM/TGI),
  quantization (INT8/INT4/FP8, GPTQ/AWQ), KV-cache paging. See [engineering-scale.md](engineering-scale.md).
- **Structured output:** constrained decoding / grammars (JSON schema, regex) for reliable tool/agent I/O.

## 11. In-context learning, prompting, RAG, and agents

- **In-context learning:** LLMs perform tasks from examples in the prompt without weight updates — few-shot
  demonstrations, instructions, and format steer behavior. Powerful but brittle to ordering/format; not a
  substitute for fine-tuning when you need reliability.
- **Prompting discipline:** clear instructions, explicit output format, relevant context first, decomposition
  (let the model reason step by step), and few-shot exemplars when format matters. Evaluate prompts like any
  other intervention — with a held-out set and the rigor in [evaluation-statistics.md](evaluation-statistics.md),
  not vibes.
- **Retrieval-Augmented Generation (RAG):** retrieve relevant chunks (embedding similarity via a vector
  index + often a reranker, sometimes hybrid BM25+dense) and condition generation on them. The default for
  grounding on private/fresh knowledge and reducing hallucination. Quality hinges on chunking, embedding model,
  retrieval recall, and reranking — evaluate retrieval and generation *separately*.
- **Agents / tool use:** the model plans, calls tools/functions, observes results, and iterates (ReAct-style).
  Reliability comes from constrained outputs, verification/guardrails, good tool design, and bounded loops.
  Evaluate end-to-end task success, not single-step plausibility.

## 12. Evaluating LLMs (see [evaluation-statistics.md](evaluation-statistics.md) for rigor)

- **Contamination is the central threat:** public static benchmarks leak into pretraining → recall masquerading
  as skill. Prefer **frequently-refreshed / held-out** benchmarks (LiveBench, LiveCodeBench, FrontierMath,
  MMLU-Pro) and report contamination checks (Min-K%, n-gram overlap, ConStat).
- **Capability evals:** task-specific with verifiable answers where possible (math/code with execution).
- **LLM-as-judge:** scalable but biased (position, verbosity, self-preference, style over substance) — calibrate
  against human labels, randomize order, use rubrics, and never let a model judge its own family unblinded.
- **Safety/alignment evals:** refusal, jailbreak robustness, bias, honesty/calibration — see
  [interpretability-safety.md](interpretability-safety.md).
- Report variance and significance; LLM eval deltas are often within noise.

## 13. Practical defaults

| Decision | Default (2026) |
|---|---|
| Architecture | Pre-norm decoder transformer, RMSNorm, RoPE, SwiGLU, GQA |
| Long-sequence efficiency critical | Evaluate attention+SSM hybrid (Mamba-2) ± MoE |
| Want capacity at fixed inference FLOPs | MoE (if memory/comms allow) |
| Align a model to instructions | SFT (LoRA/QLoRA) → DPO |
| Improve verifiable reasoning | GRPO / RLVR |
| Ground on private/fresh data | RAG (good chunking + reranker), evaluate retrieval separately |
| Serve efficiently | vLLM/TGI, quantization, speculative decoding, GQA/paged KV |
| Evaluate | Held-out/refreshed benchmarks + task-specific verifiable evals + variance |

**Canonical references:** Vaswani et al. 2017 (*Attention Is All You Need*); Brown et al. 2020 (GPT-3,
in-context learning); Hoffmann et al. 2022 (Chinchilla); Su et al. 2021 (RoPE); Dao et al. 2022
(FlashAttention); Rafailov et al. 2023 (DPO); Shao et al. 2024 / DeepSeek-R1 2025 (GRPO/RLVR); Gu & Dao 2023–24
(Mamba/Mamba-2); Lewis et al. 2020 (RAG); Wei et al. 2022 (chain-of-thought).
