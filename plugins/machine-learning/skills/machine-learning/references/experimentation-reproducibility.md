# Experimentation & Reproducibility

Running ML experiments so that (a) you can trust the conclusions, (b) you can reproduce any run, and (c) others
can too. Reproducibility is now a first-class research activity — the Machine Learning Reproducibility Challenge
is an official NeurIPS track (2026). Build for it from run one; retrofitting is painful and usually fails.

---

## 1. Experiment design — ask a question, isolate the cause

- **One change at a time.** To attribute an effect to a cause, vary one thing and hold the rest fixed. Multi-
  change "it's better now" experiments teach you nothing about *why*. Keep a running ablation log.
- **State the hypothesis and the falsifier** before running: "X improves Y because Z; if X doesn't beat baseline
  by >δ across seeds, I'm wrong." Pre-deciding the success criterion prevents post-hoc rationalization.
- **Controls.** Every experiment needs a baseline run under identical conditions (same data, compute, tuning
  effort — see SKILL.md). A label-shuffled / random-feature control catches leakage and harness bugs.
- **Ablations** are how you prove the claim: remove/replace each component of your method and show the metric
  drops. A method whose ablations don't move the metric isn't doing what you claim. Ablate the actual mechanism,
  not a strawman.
- **Power/budget:** decide seeds and test-set size to detect the effect size you care about *before* running
  (see [evaluation-statistics.md](evaluation-statistics.md)). Don't run 1 seed and hope.
- **Cheap before expensive.** Validate the idea at small scale / short runs / a data subset before committing
  big compute. Use scaling-law trends to extrapolate (see [transformers-llms.md](transformers-llms.md)).

## 2. Configuration management (the backbone of reproducibility)

Every run is fully determined by a **config** — never hardcode hyperparameters or pass long CLI strings.
- **Hydra + OmegaConf** (the standard): composable YAML configs, command-line overrides, config groups
  (swap optimizer/model/dataset), multirun sweeps, and **the resolved config is logged with every run**.
  Structured configs (dataclasses) add type safety.
- **The rule:** the saved config + the code commit + the data version must **fully reproduce** a run. If
  reproduction requires undocumented manual steps, it's not reproducible.
- **Log the *resolved* config** (after all overrides/interpolation) as an artifact of the run, not just the base
  file.

```yaml
# config.yaml (Hydra) — composable, overridable, logged with the run
defaults: [_self_, {model: transformer}, {optimizer: adamw}, {data: imagenet}]
seed: 0
train: {epochs: 100, batch_size: 256, precision: bf16, grad_clip: 1.0}
# override at the CLI: python train.py optimizer.lr=3e-4 model=resnet50 seed=1,2,3
```

## 3. Seeds & determinism

- **Seed everything:** Python `random`, NumPy, the framework RNG, and the DataLoader workers (and `PYTHONHASHSEED`).
  `scripts/repro.py` does this and dumps the environment. CUDA adds nondeterminism (atomic ops, algorithm
  selection); `torch.use_deterministic_algorithms(True)` + `cudnn.deterministic=True` makes runs bit-reproducible
  at a speed cost.
- **The deeper point:** report results that are **robust across seeds**, not a single lucky seed. Determinism
  reproduces *one* run; **multiple seeds** establish that the *finding* is real. You need both: determinism for
  debugging/auditing, multi-seed for claims (see [evaluation-statistics.md](evaluation-statistics.md) §5).
- **Don't seed-hack:** trying seeds until one "works" and reporting it is fabrication. Fix seeds in advance
  (e.g., 0–4) and report all.

## 4. Experiment tracking

Log every run automatically — metrics, config, system stats, artifacts, and code version.
- **Tools:** **Weights & Biases** (best interactive viz/collaboration, SaaS), **MLflow** (3.x; open-source,
  self-hostable, model registry + lifecycle, now also LLM tracing/eval), **TensorBoard** (local, lightweight).
  (Neptune's SaaS shut down March 2026.) Many teams use **Hydra → MLflow/W&B** (e.g., HydraFlow) so the config
  is saved as a run artifact automatically.
- **Log:** primary + secondary metrics over steps, the resolved config, git commit hash + dirty-state, hardware/
  library versions, learning curves, gradient/param norms, sample predictions, and final artifacts
  (checkpoints). Future-you and reviewers need all of it.
- **Organize:** consistent run names/tags, group by experiment, and record *why* you ran it. A tracker full of
  un-annotated runs is write-only memory.

## 5. Hyperparameter optimization (HPO)

- **Methods (worst→best per dollar):** grid search (only for ≤2–3 params), **random search** (better than grid —
  it explores important dimensions more — Bergstra & Bengio 2012), **Bayesian optimization** (Optuna,
  Ax/BoTorch; GP/TPE surrogate, sample-efficient for expensive evals — see
  [probabilistic-ml.md](probabilistic-ml.md)), **Hyperband / ASHA** (early-stop bad configs, great with
  parallelism), **PBT** (Population-Based Training, evolves schedules during training).
- **Search smart:** log-scale the learning rate and regularization; start wide then narrow; identify the few
  params that matter (LR almost always dominates) and fix the rest. Use multi-fidelity (short runs / subsets) to
  triage.
- **Tools:** **Optuna** (flexible, define-by-run, pruning), **Ray Tune** (distributed, integrates ASHA/PBT),
  W&B Sweeps.
- **The integrity rule:** **tune on validation, report on test, once.** HPO is a multiple-comparison machine —
  the best-of-many validation score is biased high; re-estimate the chosen config (see
  [evaluation-statistics.md](evaluation-statistics.md) §1). Report the search space and budget; "we tuned ours,
  used defaults for the baseline" is the cardinal sin.

## 6. Versioning code, data, models, and environment

Reproducibility needs all four pinned together:
- **Code:** git; record the exact commit (and whether the tree was dirty) in every run. Tag releases used for
  papers.
- **Data:** version datasets (DVC, git-LFS, or a hashed/dated snapshot in object storage). Record the data
  version/hash per run — "trained on the dataset" is not reproducible; "trained on dataset@a1b2c3" is. Pin
  preprocessing too.
- **Models:** checkpoint with metadata (config, data version, metrics, code commit); a model registry (MLflow)
  for lifecycle/staging.
- **Environment:** pin dependencies (`uv`/`pip-tools` lockfile, conda env export, or `requirements.txt` with
  hashes) and capture CUDA/driver/hardware. A **container (Docker)** is the gold standard for "runs the same
  everywhere." Record framework + CUDA versions — silent version drift breaks reproduction.

## 7. Compute hygiene & efficiency

- **Checkpoint frequently and resume** — long runs will be preempted; save optimizer + scheduler + RNG state,
  not just weights.
- **Fail fast:** validate the full pipeline on a tiny config (1 step, 1 batch) before launching a long/expensive
  run. Smoke-test the eval harness too.
- **Budget in FLOPs/GPU-hours**, log compute cost per run, and prefer the cheapest experiment that answers the
  question. Report total compute for the project (now expected in many venues — see
  [research-workflow.md](research-workflow.md)).
- **Sweep efficiently:** early-stopping (ASHA), shared data caching, and mixed precision (see
  [engineering-scale.md](engineering-scale.md)) multiply throughput.

## 8. Reproducibility levels (aim for the top)

1. **Repeatable** — *you* can rerun and get the same result (seeds + config logged).
2. **Reproducible** — *someone else* with your code + data + environment matches your numbers within CI.
3. **Replicable** — an independent reimplementation reaches the same *conclusion* (the gold standard; the goal
   of the MLRC).
Most published ML fails level 2. The fixes are cheap and listed below.

## 9. The reproducibility checklist (adapted from NeurIPS/ML community standards)

- [ ] Code released, runnable, with exact commands to reproduce each result.
- [ ] All hyperparameters and the search space/budget reported; configs included.
- [ ] Data: source, version/hash, splits, and preprocessing fully specified (and shared if possible).
- [ ] Environment pinned (lockfile/container); framework/CUDA versions recorded.
- [ ] Seeds fixed and reported; results are **mean ± variability over multiple seeds**, not single-run.
- [ ] Compute reported (hardware + total GPU-hours/FLOPs).
- [ ] Evaluation protocol (metric, significance test, splits) fully described and matches the claims.
- [ ] Known limitations, negative results, and failure modes stated honestly.
- [ ] A second person (or CI) reran a key result and matched it within CI before you call it done.

**Canonical references:** NeurIPS Reproducibility Program & Checklist; Pineau et al. 2021 ("Improving
Reproducibility in ML Research"); Bergstra & Bengio 2012 (random search); Hydra & Optuna docs; Kapoor &
Narayanan 2023 (leakage/reproducibility in ML-based science); the ML Reproducibility Challenge (MLRC).
