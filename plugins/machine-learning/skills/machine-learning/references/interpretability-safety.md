# Interpretability, Robustness, Fairness, Privacy & Safety

Understanding what models do, and making them trustworthy. This file is framed for **defensive, evaluative, and
beneficial** use — auditing models, measuring robustness/fairness/privacy, and building safer systems. Dual-use
techniques (adversarial attacks, membership inference, model extraction) are covered as **evaluation and defense
tools**, which is how they should be applied (see SKILL.md §8).

---

## 1. Interpretability & explainability (XAI)

**First question: do you need an *interpretable model* or a *post-hoc explanation*?** When stakes are high,
prefer an inherently interpretable model (linear/GAM/shallow tree, EBM) over explaining a black box — post-hoc
explanations can be unfaithful (Rudin 2019). Use explanations to *debug and audit*, not to launder a model you
can't justify.

- **Feature attribution (which inputs mattered):**
  - **SHAP** — Shapley-value attributions, theoretically grounded (consistent, local-accurate), the de facto
    standard for tabular; TreeSHAP is exact and fast for tree ensembles. **LIME** — local linear surrogate;
    cheaper, less stable.
  - **Gradient-based** (deep nets): Integrated Gradients (axiomatic, needs a baseline), SmoothGrad, Grad-CAM
    (CNN spatial maps). **Saliency maps are fragile** — sensitive to the method and to adversarial manipulation;
    treat as hypotheses, not proof. Run sanity checks (Adebayo et al.: an explanation that doesn't change when
    the model is randomized is meaningless).
- **Global understanding:** partial dependence / ALE plots, permutation feature importance (model-agnostic,
  but misleading under correlated features), surrogate models, concept-based explanations (TCAV).
- **Probing** (what's encoded in representations): train a simple classifier on frozen activations to test for
  a property. Caveat: a probe can detect information the model *doesn't use* — pair with causal interventions.
- **Mechanistic interpretability** (reverse-engineering the computation — a fast-moving 2025–2026 frontier,
  named an MIT Tech Review 2026 breakthrough):
  - **Circuits:** identify subgraphs of neurons/heads implementing a behavior (induction heads, etc.) via
    activation patching / causal tracing (intervene on activations to establish *causal* role, not just
    correlation).
  - **Superposition & sparse dictionaries:** features are entangled across neurons; **sparse autoencoders
    (SAEs)** (and transcoders/crosscoders, CB-SAEs) decompose activations into sparser, more **monosemantic**
    features, enabling concept discovery and **steering** (add a feature direction to change behavior). Caveats:
    "feature" still lacks a rigorous definition, SAEs can underperform simple baselines on safety tasks, and
    interpretations need causal validation. Powerful and improving, not yet a solved tool.

## 2. Robustness & adversarial evaluation

- **Adversarial examples:** tiny, often imperceptible input perturbations flip predictions (Szegedy/Goodfellow).
  Generate with FGSM (one step) / **PGD** (iterative, the standard strong attack) to *measure* a model's
  worst-case robustness. **Always evaluate against an adaptive attack** — claims of robustness against weak/
  obfuscated-gradient attacks repeatedly fall (Athalye et al. 2018). Report robust accuracy under a clearly
  specified threat model (norm, budget).
- **Defenses:** **adversarial training** (train on PGD examples — the most reliable, but costs accuracy and
  compute), **certified robustness** (randomized smoothing, interval bound propagation — provable guarantees
  within a radius), input preprocessing (weak, often broken). There is a real **robustness–accuracy trade-off**.
- **For LLMs:** jailbreaks and prompt injection are the adversarial frontier — red-team with automated and
  manual attacks, evaluate refusal robustness, and defend in depth (input/output filtering, system-prompt
  hardening, constrained decoding). Treat prompt injection as an unsolved security problem, not a prompt-tuning
  task.

## 3. Distribution shift & OOD

Models fail when deployment data differs from training data — the dominant cause of real-world ML failure.
- **Types:** covariate shift ($p(x)$ changes), label shift ($p(y)$ changes), concept drift ($p(y\mid x)$
  changes over time). Diagnose which — the fix differs.
- **Detection:** monitor input/output distributions in production, OOD detection (max-softmax/energy scores,
  Mahalanobis distance, deep-ensemble disagreement), and drift tests. Calibration degrades under shift (see
  [evaluation-statistics.md](evaluation-statistics.md)).
- **Robustifying:** domain adaptation/generalization, **distributionally-robust optimization (DRO) / Group-DRO**
  (optimize worst-group loss — directly targets the failing slice), test-time adaptation, and the simplest
  fix — **collect representative data and retrain**. Spurious correlations (shortcut learning: the model keys on
  the background/watermark, not the object) are a core failure — find them via slicing and counterfactual tests.
- **Evaluate on shifted benchmarks** (WILDS, ImageNet-C/-R/-A, OOD splits), not just i.i.d. test, before
  claiming robustness.

## 4. Fairness & bias

- **Bias enters through data** (historical bias, under-representation, biased labels) and gets amplified.
  Mitigation starts with the data and the problem framing, not a post-hoc patch.
- **Fairness metrics conflict — you cannot satisfy all simultaneously** (impossibility results):
  demographic parity (equal positive rates), equalized odds (equal TPR/FPR across groups), equal opportunity
  (equal TPR), calibration within groups, individual fairness. **Choosing the metric is a value judgment** tied
  to the context and harm — make it explicit with stakeholders; don't default silently.
- **Always report sliced/worst-group performance** across protected groups (and intersections), not just the
  aggregate (see [evaluation-statistics.md](evaluation-statistics.md) §9).
- **Mitigations:** pre-processing (reweighing, representation debiasing), in-processing (fairness constraints,
  adversarial debiasing, Group-DRO), post-processing (group-specific thresholds). Each trades off against
  accuracy and against other fairness notions. Audit with tools (Fairlearn, AIF360) and document with model
  cards (Mitchell et al.).

## 5. Privacy

- **Threats (use these to *audit* your model's leakage):** **membership inference** (was this example in
  training?), **training-data extraction** (LLMs and generative models can memorize and regurgitate verbatim
  data — a real privacy/IP risk), model inversion, and attribute inference. Run these attacks against your own
  model to *quantify* leakage before release.
- **Defenses:** **Differential Privacy (DP)** — the rigorous standard; **DP-SGD** (clip per-example gradients +
  add calibrated noise) bounds any single example's influence, with a privacy budget ε (smaller = more private)
  and a real utility cost. Federated learning (keep data on device; combine with DP and secure aggregation for
  real guarantees — FL alone is *not* private). Deduplication reduces memorization (see [data.md](data.md)).
- **The honest framing:** privacy is quantitative — state the threat model and the guarantee (ε), and verify
  empirically with the attacks above.

## 6. Safety & alignment (esp. for LLMs/agents)

- **Alignment goal:** models that are helpful, honest, and harmless, and that do what's intended rather than
  what's literally specified. **Specification gaming / reward hacking** (the model optimizes the proxy, not the
  intent) is the core technical failure — the same lesson as RL reward hacking (see
  [reinforcement-learning.md](reinforcement-learning.md) §10) and the generalization of "don't fool yourself."
- **Techniques:** RLHF / preference optimization / RLVR and Constitutional-AI-style methods for steering
  behavior (see [transformers-llms.md](transformers-llms.md) §9); refusal training; **scalable oversight**
  (debate, recursive reward modeling) for tasks humans can't directly supervise; interpretability/steering (§1)
  as a monitoring tool.
- **Evaluation (the safety case):** capability *and* propensity evals — refusal/jailbreak robustness, honesty/
  calibration (does it know what it doesn't know?), bias/toxicity, deception, autonomy/dangerous-capability
  evals, and agentic-task safety. **Red-team** (manual + automated). Report safety evals with the same rigor and
  variance as capability evals; many safety deltas are within noise.
- **Hallucination/factuality:** measure with grounded benchmarks; mitigate with retrieval (RAG), calibration,
  abstention ("I don't know"), and verification. Don't conflate fluency with correctness.

## 7. Cross-cutting principles

- **Interpretation needs causal validation.** Correlational evidence (attributions, probes) can mislead;
  confirm with interventions (patching, ablation, counterfactuals) before believing a mechanistic claim.
- **Robustness, fairness, privacy, and accuracy trade off.** There is rarely a free lunch — state which you're
  optimizing and the cost to the others.
- **Evaluate the property you claim**, on data that stresses it (shifted/adversarial/per-group), with variance
  and an adaptive/strong test — not the i.i.d. average.
- **Document** with datasheets (data), model cards (models), and clear statements of intended use, limitations,
  and known failure modes (see [research-workflow.md](research-workflow.md)).

**Canonical references:** Rudin 2019 (stop explaining black boxes — use interpretable models); Lundberg & Lee
2017 (SHAP); Sundararajan et al. 2017 (Integrated Gradients); Adebayo et al. 2018 (saliency sanity checks);
Madry et al. 2018 (PGD adversarial training); Athalye et al. 2018 (obfuscated gradients); Sagawa et al. 2020
(Group-DRO); Dwork & Roth (DP foundations); Abadi et al. 2016 (DP-SGD); Carlini et al. 2021 (extracting training
data from LLMs); Mitchell et al. 2019 (model cards); Anthropic/OpenAI interpretability work on SAEs (2024–2026).
