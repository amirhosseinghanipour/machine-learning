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
  - **SHAP** — Shapley-value attributions, theoretically grounded (consistent, local-accurate, missingness), the
    de facto standard for tabular; TreeSHAP is exact and fast for tree ensembles. Caveat: the default
    (interventional vs. conditional/observational) **changes attributions under correlated features** — pick
    deliberately and report which. KernelSHAP is slow and approximate; don't over-read tiny differences.
    **LIME** — local linear surrogate; cheaper, **unstable** (re-runs give different explanations; sensitive to
    the perturbation kernel and neighborhood) — prefer SHAP when you can afford it.
  - **Gradient-based** (deep nets): **Integrated Gradients** (axiomatic — completeness/sensitivity — but
    **baseline choice is load-bearing**: black image ≠ blurred ≠ random ≠ expectation over a reference set; report
    it and ideally average over baselines), SmoothGrad, Grad-CAM (CNN spatial maps; for ViTs use attention
    rollout/attention-flow or Grad-CAM on the last attention block, not raw attention). **Raw attention is not
    explanation** (Jain & Wallace 2019; Wiegreffe & Pinter 2019) — attention weights are neither necessary nor
    sufficient for the prediction. **Saliency maps are fragile** — sensitive to the method and to adversarial
    manipulation; treat as hypotheses, not proof. **Run the sanity checks** (Adebayo et al. 2018): a
    *model-parameter-randomization test* and a *label-randomization test* — an explanation that barely changes
    when you randomize weights or labels is measuring the input/edges, not the model, and is meaningless.
- **Global understanding:** partial dependence / **ALE** plots (ALE is correct under correlated features where
  PDP is not), permutation feature importance (model-agnostic but **double-counts and misleads under correlated
  features** — prefer conditional/leave-one-covariate-out or grouped importance), surrogate models, concept-based
  explanations (TCAV; concept-bottleneck models for built-in concepts).
- **Probing & representation analysis** (what's encoded): train a simple classifier on frozen activations to
  test for a property. **Two essential caveats:** (1) a high-capacity probe can fit a property the model
  *doesn't use* — use **control tasks / selectivity** (Hewitt & Liang 2019) and minimum-description-length
  probing (Voita & Titov) to separate "encoded" from "memorized by the probe"; (2) "encoded" ≠ "used" — confirm
  with **causal interventions** (amnesic probing / INLP, activation patching). Representational comparison: CKA,
  but it is sensitive to outliers and can disagree with other similarity measures — triangulate.
- **Mechanistic interpretability** (reverse-engineering the computation into human-understandable algorithms — a
  fast-moving 2024–2026 frontier; Anthropic's *Transformer Circuits Thread* and the "biology of an LLM" /
  attribution-graph work are the reference points):
  - **Circuits & causal methods:** identify subgraphs of neurons/heads implementing a behavior (induction heads,
    IOI circuit, successor heads). The toolkit is **causal, not correlational**: **activation patching /
    interchange interventions** (swap a clean activation into a corrupted run, or vice versa, and measure the
    effect on the logit/metric), **path patching** (which *edges* carry the effect), **attribution patching**
    (a cheap linear/gradient approximation of patching that scales to many sites), and **causal scrubbing /
    distributed alignment search (DAS)** to test a hypothesized circuit. Always specify the metric, the
    clean/corrupt pair, and what a null (random direction) does.
  - **Superposition & sparse dictionaries.** Features are in **superposition** — more concepts than neurons,
    spread across directions. **Sparse autoencoders (SAEs)** decompose activations into a sparse overcomplete
    dictionary of more **monosemantic** latents, enabling concept discovery and **steering** (add/clamp a latent
    to change behavior). Defaults & current practice (2024–2026): expect **>10⁴–10⁶ latents**; modern
    architectures **TopK / BatchTopK / JumpReLU / gated SAEs** beat vanilla L1 on the
    sparsity–reconstruction–interpretability frontier and avoid L1 shrinkage; evaluate with
    reconstruction loss, **explained loss recovered when you patch the SAE back in**, dead-latent fraction, and
    automated interpretability scores — not reconstruction MSE alone.
  - **Transcoders & crosscoders (the 2025 shift).** **Transcoders** learn a sparse map from a module's *input*
    to its *output* (e.g. replacing an MLP), giving sparser, more interpretable, *input-independent* weight
    circuits and outperforming same-width SAEs on circuit analysis; **skip/cross-layer transcoders** power
    Anthropic's **attribution graphs** (replacement model → prune → human-readable circuit). **Crosscoders**
    learn shared dictionaries across layers or across models, enabling **model diffing** (what changed in
    fine-tuning). **Caveats that matter for a researcher:** "feature" still lacks a rigorous definition;
    **feature splitting/absorption** and **manifold** structure complicate the "one latent = one concept" story;
    SAEs/probes can **underperform simple linear baselines on downstream safety tasks** (the
    interpretability-illusion critique) and have **partially negative results** for steering controllability;
    reconstruction can miss causally-important low-norm directions. Treat SAEs as a hypothesis-generation and
    monitoring tool that **requires causal validation and a baseline**, not a finished instrument.
  - **Parameter-space & other frontiers:** attribution-based parameter decomposition / weight-space methods,
    and emerging **"interpretability for oversight"** framings (use features as a probe/monitor for deception or
    sandbagging) — promising but immature; validate against behavioral evals.

## 2. Robustness & adversarial evaluation

- **Adversarial examples:** tiny, often imperceptible input perturbations flip predictions (Szegedy/Goodfellow).
  Generate with FGSM (one step) / **PGD** (iterative, the standard strong attack) to *measure* a model's
  worst-case robustness. **Always evaluate against an adaptive attack and a strong ensemble** — claims of
  robustness against weak/obfuscated-gradient attacks repeatedly fall (Athalye et al. 2018, Tramèr et al. 2020
  "On adaptive attacks"). The community standard battery is **AutoAttack** (parameter-free ensemble: APGD-CE,
  APGD-DLR, FAB, Square) reported on **RobustBench**; a single hand-tuned PGD is not enough. Report robust
  accuracy under a clearly specified threat model (norm ℓ∞/ℓ2, budget ε, query limit). **Sanity gates:** robust
  accuracy should drop monotonically as ε grows and hit ~0 at large ε; if it plateaus, you have gradient
  masking. Beyond ℓp: **patch/physical attacks**, semantic/unrestricted perturbations, and (for text) discrete
  token attacks (GCG) — pick the threat model that matches deployment.
- **Defenses:** **adversarial training** (PGD-AT, TRADES that explicitly trades robust vs. clean accuracy,
  MART; the most reliable, but costs accuracy and compute and can overfit — use early stopping on robust val,
  Rice et al. 2020). Recent gains come mostly from **more/better data** (generative-model-augmented training)
  and scale, not new losses. **Certified robustness** (randomized smoothing → probabilistic ℓ2 certificate at
  inference cost of many forward passes; interval-bound propagation / CROWN/α,β-CROWN for deterministic
  verification of small nets) gives *provable* guarantees within a radius but lags empirical accuracy. Input
  preprocessing / purification is weak and often broken by adaptive attacks. There is a real
  **robustness–accuracy and robustness–fairness trade-off** (robustness gains can be uneven across classes).
- **For LLMs — the live frontier (2026):** jailbreaks and **prompt injection** are the dominant attack
  surface; **OWASP ranks prompt injection #1 on the LLM Top-10 two years running**, and indirect
  (data-borne) prompt injection in tool-using agents is largely **unsolved**. Know the attack families:
  multi-turn **Crescendo**, **many-shot jailbreaking** (long context full of faux-compliant exemplars —
  scales with context length), **best-of-N** (cheap stochastic resampling across modalities), **GCG**-style
  optimized suffixes, and **automated/agentic red-teamers**. Evaluate with jailbreak suites (HarmBench,
  JailbreakBench, **StrongREJECT** to avoid over-counting low-quality "jailbreaks"). Treat prompt injection
  as a **security** problem (assume the input is adversarial), not a prompt-tuning task. Regulatory note: the
  EU AI Act expects adversarial-robustness testing for GPAI/systemic-risk models. **The agent-specific attack
  surface, defense-in-depth (least-privilege tools, human-in-the-loop, dual-LLM/quarantine), and agentic
  safety benchmarks (AgentDojo, AgentHarm, SHADE-Arena) are covered in [agents.md](agents.md) §5.**

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
- **Fairness metrics conflict — you cannot satisfy all simultaneously** (formal impossibility, Kleinberg et al.
  2016 / Chouldechova 2017): **calibration within groups**, **equal FPR**, and **equal FNR** cannot all hold at
  once unless base rates are equal or prediction is perfect. The metric menu: demographic/statistical parity
  (equal positive rates), equalized odds (equal TPR *and* FPR), equal opportunity (equal TPR), predictive
  parity, calibration-by-group, individual fairness, counterfactual fairness (causal). **Choosing the metric is
  a value judgment** tied to the context and the harm (allocative vs. representational; who bears a false
  positive vs. false negative) — make it explicit with stakeholders; don't default silently. Beware **fairness
  gerrymandering** (parity on each marginal group but not on intersections) and that enforcing parity can
  *lower* outcomes for everyone — interrogate whether the disparity is in the model or the world.
- **Always report sliced/worst-group performance** across protected groups (and intersections), not just the
  aggregate (see [evaluation-statistics.md](evaluation-statistics.md) §9).
- **Mitigations:** pre-processing (reweighing, representation debiasing), in-processing (fairness constraints,
  adversarial debiasing, Group-DRO), post-processing (group-specific thresholds). Each trades off against
  accuracy and against other fairness notions. Audit with tools (Fairlearn, AIF360) and document with model
  cards (Mitchell et al.).

## 5. Privacy

- **Threats (use these to *audit* your model's leakage):** **membership inference** (was this example in
  training?), **training-data extraction** (LLMs and generative models can memorize and regurgitate verbatim
  data — a real privacy/IP risk; extraction scales with model size and duplication), model inversion, and
  attribute inference. Run these attacks against your own model to *quantify* leakage before release.
  **Measure MIA correctly:** average-case **accuracy/AUC is misleading** — report **TPR at low FPR**
  (e.g. 0.1%/1%) on a log-ROC, because privacy is about the *worst-case* example. Use a **calibrated
  likelihood-ratio attack (LiRA**, Carlini et al. 2022) with shadow/reference models, not loss thresholds;
  account for **example difficulty** (easy non-members masquerade as members). Per-example vulnerability rises
  with how often an example appears and falls with dataset size (roughly power-law). For LLM-specific auditing,
  insert **canaries** and measure their extraction/MIA exposure; preference/alignment data has its own MIAs.
- **Defenses:** **Differential Privacy (DP)** — the rigorous standard; **DP-SGD** (clip per-example gradients +
  add calibrated Gaussian noise) bounds any single example's influence, accounted with the **Rényi/PRV/Gaussian
  DP accountant** (tighter than naive composition), giving a budget ε (smaller = more private; for ML, ε≈1–10
  is the common range, but the meaning depends on δ and the unit of privacy). Practical recipe: large batches
  (DP loves big batch sizes), tuned clipping norm, often **fine-tune only** under DP or use parameter-efficient
  DP; expect a real utility cost that **shrinks with scale and pretraining**. **Empirically audit ε**
  (one-run DP auditing via inserted canaries) — implementations frequently violate the claimed bound.
  Federated learning keeps data on device but is **not private by itself**; combine with **DP + secure
  aggregation** (and beware reconstruction-from-gradients attacks). Deduplication reduces memorization (see
  [data.md](data.md)). **Machine unlearning** (removing a training example's influence post hoc) is requested
  for compliance but **hard to verify** — approximate unlearning often leaves measurable traces; treat claims
  skeptically and audit with MIA.
- **The honest framing:** privacy is quantitative — state the threat model, the **unit** (example vs. user
  vs. group), and the guarantee (ε, δ), and verify empirically with the attacks above. "Anonymized" and
  "we don't store data" are not guarantees.

## 6. Safety & alignment (esp. for LLMs/agents)

- **Alignment goal:** models that are helpful, honest, and harmless, and that do what's intended rather than
  what's literally specified. **Specification gaming / reward hacking** (the model optimizes the proxy, not the
  intent) is the core technical failure — the same lesson as RL reward hacking (see
  [reinforcement-learning.md](reinforcement-learning.md) §10) and the generalization of "don't fool yourself."
- **Techniques:** RLHF / preference optimization / RLVR and Constitutional-AI-style methods (RLAIF) for steering
  behavior (see [transformers-llms.md](transformers-llms.md) §9); refusal/safety training and **deliberative
  alignment** (train the model to reason over a safety spec before answering); **scalable oversight** (debate,
  recursive reward modeling, weak-to-strong generalization) for tasks humans can't directly supervise;
  interpretability/steering (§1) as a monitoring tool. Watch the failure modes: **sycophancy** (telling the
  user what they want), **reward hacking**, and **over-refusal** (a safety-tax that breaks helpfulness) — eval
  for all three.
- **Emerging-risk evals to know:** **alignment faking / deceptive alignment** (a model behaves well under
  observation but not otherwise), **sandbagging** (strategically underperforming on capability evals),
  **scheming/situational awareness**, **sabotage and self-exfiltration**. These are propensity questions and
  motivate **control** approaches (assume the model may be misaligned; design monitoring/containment that is
  robust to it — Greenblatt et al. "AI control") alongside alignment. Frontier labs gate releases on
  **Responsible Scaling Policies / Preparedness frameworks** with capability thresholds (CBRN, cyber, autonomy)
  and **dangerous-capability evals** (e.g. METR autonomy/uplift studies, WMDP as a proxy with the hazardous
  content removed).
- **Evaluation (the safety case):** capability *and* propensity evals — refusal/jailbreak robustness (HarmBench,
  StrongREJECT), honesty/calibration (does it know what it doesn't know? — TruthfulQA is dated; prefer
  calibration + abstention metrics), bias/toxicity, deception, dangerous-capability and **agentic-task safety**
  (AgentHarm, AgentDojo, SHADE-Arena). **Red-team** with manual + automated/agentic methods. Report safety evals
  with the **same rigor and variance** as capability evals; many safety deltas are within noise, evals saturate
  and **contaminate**, and a passed eval is **evidence of a capability floor, not a safety ceiling** — absence
  of a demonstrated harm is weak evidence of safety. Prefer **elicitation-maximizing** evals (fine-tune/scaffold
  to surface the true capability) for safety-relevant claims.
- **Hallucination/factuality:** measure with grounded benchmarks; mitigate with retrieval (RAG), calibration,
  **abstention** ("I don't know"), and verification. Note the trained-in pressure toward confident guessing:
  standard scoring **rewards bluffing over abstaining**, so penalize confident errors more than honest
  abstentions in your eval. Don't conflate fluency with correctness.

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
Jain & Wallace 2019 / Wiegreffe & Pinter 2019 (attention is/ isn't explanation); Hewitt & Liang 2019 (probing
control tasks); Olah et al. *Transformer Circuits Thread* and Anthropic "Towards Monosemanticity" / "Scaling
Monosemanticity" / "On the Biology of an LLM" + attribution graphs (2023–2025); Templeton/Lieberum et al.
(JumpReLU/gated SAEs); Dunefsky et al. & Ameisen et al. 2025 (transcoders, circuit tracing); Madry et al. 2018
(PGD adversarial training); Zhang et al. 2019 (TRADES); Croce & Hein 2020 (AutoAttack/RobustBench); Cohen et al.
2019 (randomized smoothing); Athalye et al. 2018 / Tramèr et al. 2020 (obfuscated/adaptive attacks); Zou et al.
2023 (GCG); Anil et al. 2024 (many-shot jailbreaking); Sagawa et al. 2020 (Group-DRO); Dwork & Roth (DP
foundations); Abadi et al. 2016 (DP-SGD); Carlini et al. 2021 (extracting training data from LLMs); Carlini et
al. 2022 (LiRA membership inference); Mitchell et al. 2019 (model cards); Greenblatt et al. 2023–2024 (AI
control; alignment faking). Tooling: SHAP, Captum, `transformer-lens`/`nnsight` + `SAELens`, Fairlearn/AIF360,
Opacus (DP-SGD), RobustBench, HarmBench/Inspect for safety evals.
