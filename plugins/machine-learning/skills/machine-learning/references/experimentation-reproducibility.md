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
  not a strawman. Discipline: (1) **hold compute/params/data fixed** across ablation arms — removing a component
  that also shrinks the model conflates the mechanism with capacity; re-tune or compute-match the ablated arm.
  (2) Run **both directions** where feasible — *remove from full* and *add to baseline* — since interactions mean
  the two can disagree. (3) Ablate over **≥3 seeds with CIs**; a single-seed ablation table is decorative.
  (4) Beware ablation **multiple comparisons** — many small "each component helps by 0.3%" rows are often noise;
  test the ones the claim rests on.
- **Power/budget:** decide seeds and test-set size to detect the effect size you care about *before* running
  (see [evaluation-statistics.md](evaluation-statistics.md)). Don't run 1 seed and hope.
- **Cheap before expensive.** Validate the idea at small scale / short runs / a data subset before committing
  big compute. Use scaling-law trends to extrapolate (see [transformers-llms.md](transformers-llms.md)).

## 2. Configuration management (the backbone of reproducibility)

Every run is fully determined by a **config** — never hardcode hyperparameters or pass long CLI strings.
- **Hydra + OmegaConf** (the standard): composable YAML configs, command-line overrides, config groups
  (swap optimizer/model/dataset), multirun sweeps, and **the resolved config is logged with every run**.
  Define **structured configs** as dataclasses registered in the `ConfigStore` (or pydantic) for type-checking,
  required-field validation, and IDE autocomplete — catches typos like `lr` vs `learning_rate` before a run
  burns GPU-hours. Use interpolation (`${...}`) and `oc.env`/custom resolvers to keep a single source of truth,
  but resolve them at log time.
- **The rule:** the saved config + the code commit + the data version must **fully reproduce** a run. If
  reproduction requires undocumented manual steps, it's not reproducible.
- **Log the *resolved* config** (after all overrides/interpolation, `OmegaConf.to_container(cfg, resolve=True)`)
  as an artifact of the run, not just the base file. Note OmegaConf `DictConfig` is not a plain dict — convert
  before handing it to `wandb.config` or it serializes wrong on the dashboard.
- **Avoid config drift:** don't read hyperparameters from two places (config *and* argparse) or mutate the
  config mid-run without logging the change. One config object, threaded read-only through the code.

```yaml
# config.yaml (Hydra) — composable, overridable, logged with the run
defaults: [_self_, {model: transformer}, {optimizer: adamw}, {data: imagenet}]
seed: 0
train: {epochs: 100, batch_size: 256, precision: bf16, grad_clip: 1.0}
# override at the CLI: python train.py optimizer.lr=3e-4 model=resnet50 seed=1,2,3
```

## 3. Seeds & determinism

- **Seed everything:** Python `random`, NumPy, the framework RNG (`torch.manual_seed` / `torch.cuda.manual_seed_all`),
  **the DataLoader workers** (set `worker_init_fn` and a `generator`, or PyTorch re-seeds each worker from the
  base seed every epoch — silently changing augmentation order otherwise), and `PYTHONHASHSEED` (set in the
  environment *before* the interpreter starts — setting it inside the process is too late). `scripts/repro.py`
  does this and dumps the environment. JAX is different and cleaner: RNG state is an explicit `PRNGKey` you
  `split` and thread through — no global state, so reproducibility is by construction (but you must actually
  thread the keys).
- **GPU nondeterminism is real and has specific causes:** atomic-add accumulation (scatter/`index_add`, some
  pooling and `bincount`), nondeterministic cuDNN convolution algorithms, autotuned algorithm selection, and the
  fact that **floating-point addition is non-associative** so any change in reduction order changes the bits.
  The toggles:
  - `torch.use_deterministic_algorithms(True)` — errors out (helpfully) on any op lacking a deterministic
    kernel, forcing you to confront it.
  - `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` — stop cuDNN from
    autotuning/selecting variable algorithms.
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) — **required** for deterministic cuBLAS GEMMs on CUDA ≥10.2,
    set in the environment before launch.
  - Cost: expect meaningful slowdowns (deterministic kernels and disabled autotuning); some attention kernels
    (FlashAttention paths, fused ops) and `torch.compile` modes may have no deterministic variant — you may have
    to disable them to get bit-exactness.
  - **Bit-exactness does not survive a hardware/driver/library change.** Determinism is "same code + same
    hardware + same library versions." Reductions reorder across GPU counts, so even a correct distributed run is
    not bit-identical across world sizes (see [engineering-scale.md](engineering-scale.md)). Pin the container.
- **The deeper point:** report results that are **robust across seeds**, not a single lucky seed. Determinism
  reproduces *one* run for debugging/auditing; **multiple seeds** establish that the *finding* is real (and the
  seed-to-seed std is often larger than the effect you are claiming — see
  [evaluation-statistics.md](evaluation-statistics.md) §5). You need both. Don't pay the determinism speed tax
  for production training runs — pay it for debugging and for the audit trail (logged seeds + versions), and get
  trustworthiness from multi-seed instead.
- **Don't seed-hack:** trying seeds until one "works" and reporting it is fabrication, and is a form of the
  multiple-comparisons problem. Fix seeds in advance (e.g., 0–4), report all, and never tune *on* the seed.

## 4. Experiment tracking

Log every run automatically — metrics, config, system stats, artifacts, and code version.
- **Tools (2026):** **Weights & Biases** (best interactive viz/collaboration, SaaS; **W&B Weave** for LLM/agent
  trace + eval logging), **MLflow 3.x** (open-source, self-hostable; revamped model registry with **Logged
  Models** as first-class entities, plus GenAI tracing and `mlflow.evaluate`/LLM-judge eval baked in),
  **TensorBoard** (local, lightweight), **Aim**/**ClearML** (open-source alternatives). (Neptune's SaaS shut
  down March 2026.) Many teams wire **Hydra → MLflow/W&B** (e.g., HydraFlow) so the resolved config is saved as a
  run artifact automatically.
- **Log:** primary + secondary metrics over steps, the resolved config, **git commit hash + dirty-state diff**,
  hardware/library/CUDA versions, learning curves, **gradient/param/update norms and LR** (the cheapest training-
  health telemetry — a NaN or norm spike is invisible in the loss until it isn't), throughput/util, sample
  predictions, and final artifacts (checkpoints with metadata). Future-you and reviewers need all of it.
- **Log eval the same way you log training** — version the eval harness, log the exact metric definition and the
  per-slice numbers, not just the headline. An unversioned eval that silently changes makes runs incomparable.
- **Organize:** consistent run names/tags, group by experiment, and record *why* you ran it (hypothesis +
  result, one line). A tracker full of un-annotated runs is write-only memory. Tag the runs that back each
  figure/table so the paper is traceable to runs.

## 5. Hyperparameter optimization (HPO)

- **Methods (worst→best per dollar):** grid search (only for ≤2–3 params — cost is exponential in dimensions and
  it wastes budget on unimportant axes); **random search** (better than grid — with k of d params mattering,
  random samples k effective dimensions densely while grid samples them sparsely — Bergstra & Bengio 2012);
  **Bayesian optimization** (TPE in Optuna, GP/BoTorch in Ax; sample-efficient surrogate, best when each eval is
  expensive and total budget is small — see [probabilistic-ml.md](probabilistic-ml.md)); **multi-fidelity /
  bandit** methods that early-stop weak configs — **Successive Halving → Hyperband** (runs SH across several
  brackets to hedge the budget-vs-#configs trade-off) → **ASHA** (asynchronous SH — no synchronization barrier,
  the right default at scale/with parallelism) → **BOHB / DEHB** (replace Hyperband's random sampling with a
  TPE model / differential evolution — combine multi-fidelity speed with model-based sample efficiency);
  **PBT** (Population-Based Training, evolves a *schedule* for params like LR online — good for non-stationary
  hyperparameters).
- **Search smart:** **log-scale** the learning rate, weight decay, and other multiplicative params; start wide
  then narrow; identify the few params that matter (LR almost always dominates; for transformers LR, warmup,
  weight decay, and batch size interact) and fix the rest. Use multi-fidelity (short runs / data subsets / fewer
  steps) to triage cheaply, but beware **fidelity–rank disagreement**: the best config at small scale is not
  always best at full scale (LR especially shifts with width/batch — see μP/μTransfer in
  [transformers-llms.md](transformers-llms.md) for transferring HPs across scale).
- **Tools (2026):** **Optuna** (flexible define-by-run, TPE/CMA-ES, multi-objective, ASHA/Hyperband/median
  pruners), **Ray Tune** (distributed, integrates ASHA/PBT/BOHB), **SMAC3** and **Syne Tune** (research-grade
  multi-fidelity BO), W&B Sweeps. Integrate with the tracker so every trial's config + curve is logged.
- **The integrity rule:** **tune on validation, report on test, once.** HPO is a multiple-comparison machine —
  the best-of-many validation score is biased high (the max of noisy estimates), so the winner's val score is an
  optimistic estimate; re-estimate the chosen config and reserve test for the single final number (see
  [evaluation-statistics.md](evaluation-statistics.md) §1). **Report the search space, the budget (# trials /
  GPU-hours), and the method** — without them a result is irreproducible. "We tuned ours, used defaults for the
  baseline" is the cardinal sin; tune baselines with **equal budget** and report it.

## 6. Versioning code, data, models, and environment

Reproducibility needs all four pinned together:
- **Code:** git; record the exact commit (and whether the tree was dirty) in every run. Tag releases used for
  papers.
- **Data:** version datasets (**DVC** — git-tracked `.dvc` pointer files hashing content in remote object
  storage, so `git checkout` + `dvc pull` restores the exact data for a commit; or git-LFS, lakeFS, or a
  hashed/dated immutable snapshot). DVC also captures **pipelines** (`dvc.yaml` stages with dependencies +
  outputs) so the data→features→model DAG is reproducible and only stale stages rerun. Record the data
  version/hash per run — "trained on the dataset" is not reproducible; "trained on dataset@a1b2c3" is. Pin
  preprocessing too, and version the train/val/test **split** (store IDs or a seeded deterministic procedure).
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

**Canonical references:** NeurIPS Paper Checklist & Reproducibility Program; Pineau et al. 2021 ("Improving
Reproducibility in ML Research" — the ML reproducibility checklist); Gundersen & Kjensmo 2018 (state of
reproducibility in AI); Bergstra & Bengio 2012 (random search); Li et al. 2018 (Hyperband); Falkner et al. 2018
(BOHB); Li et al. 2020 (ASHA); Awad et al. 2021 (DEHB); Jaderberg et al. 2017 (PBT); Akiba et al. 2019 (Optuna);
PyTorch Reproducibility / `use_deterministic_algorithms` docs and CUBLAS_WORKSPACE_CONFIG note; Hydra & DVC
docs; Kapoor & Narayanan 2023 (leakage/reproducibility in ML-based science); the ML Reproducibility Challenge
(MLRC, a NeurIPS track).
