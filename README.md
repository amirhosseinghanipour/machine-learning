# Machine Learning — a research-grade skill for Claude Code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-plugin-7C3AED.svg)](https://code.claude.com/docs/en/plugins)
![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue.svg)

A comprehensive, **research-grade machine learning skill** that teaches Claude Code not just *which*
model to build, but how to build it so the result is **correct, fairly measured, reproducible, and
defensible**. Most ML mistakes are not modeling mistakes — they are measurement and methodology
mistakes that make a wrong result look right. This skill is built to prevent that first, and to
provide deep, current (2026) domain knowledge second.

It spans the whole field — foundations and learning theory, classical ML, deep learning,
transformers and LLMs, generative models (diffusion / flow matching), reinforcement learning,
probabilistic and Bayesian ML, graph/geometric and representation learning — together with the rigor
that makes results trustworthy: experimental design, evaluation and statistics, reproducibility,
distributed training and scaling, data curation, interpretability and safety, and the research
workflow of reading, reproducing, and writing papers.

---

## Install

In Claude Code, add this marketplace and install the plugin:

```text
/plugin marketplace add amirhosseinghanipour/machine-learning
/plugin install machine-learning@machine-learning
```

> The `@machine-learning` after the plugin name is the **marketplace** name (it happens to match the
> repo name). Run `/plugin` any time to browse, enable/disable, or update installed plugins.

To update later, after new releases are pushed:

```text
/plugin marketplace update machine-learning
```

---

## How it works

Once installed, the skill activates automatically when you work on machine learning — designing,
training, evaluating, debugging, or scaling models; choosing architectures/losses/optimizers;
running experiments or ablations; analyzing results or significance; or reproducing and writing up
research. You can also nudge it explicitly ("use the machine-learning skill…").

It uses **progressive disclosure**: a compact `SKILL.md` carries the methodology and a router table,
and points to deep reference documents that load **on demand** for the task at hand. So the model
gets the rigor every time, and the right domain depth exactly when it's relevant — without bloating
the context.

The spine is a single principle — **don't fool yourself**: protect the test set, use strong and
fairly-tuned baselines, report variance and significance across seeds, audit for data leakage, and
make every run reproducible by construction.

---

## What's inside

**`SKILL.md`** — the always-loaded anchor: the rigor methodology, a 7-phase research workflow, a
non-negotiable rigor checklist, an anti-pattern catalog, the 2026 stack, and the router below.

**`references/` — 17 deep, on-demand documents**

| Area | File |
|---|---|
| Math, optimization theory, learning theory, information theory | `foundations.md` |
| Linear/SVM/trees & GBMs, clustering, dimensionality reduction, tabular, calibration | `classical-ml.md` |
| Architectures, normalization/init/optimizers, regularization, training & debugging | `deep-learning.md` |
| Attention/transformers, LLM lifecycle, scaling laws, post-training (SFT/DPO/GRPO/RLVR), MoE, RAG, SSMs | `transformers-llms.md` |
| VAEs, GANs, flows, diffusion, flow matching, consistency models, generative evaluation | `generative-models.md` |
| MDPs, PPO/SAC, model-based & offline RL, exploration, multi-agent, RLHF/RLVR | `reinforcement-learning.md` |
| Bayesian inference, graphical models, VI, MCMC, Gaussian processes, uncertainty | `probabilistic-ml.md` |
| GNNs, message passing, graph transformers, equivariance, geometric deep learning | `graph-geometric.md` |
| Self-supervised/contrastive, CLIP, transfer, fine-tuning & PEFT/LoRA, embeddings | `representation-learning.md` |
| Meta/few-shot, multi-task, continual, active, semi-/weakly-supervised, curriculum, causal ML | `learning-paradigms.md` |
| Metrics, cross-validation, significance testing, calibration, benchmarking, contamination | `evaluation-statistics.md` |
| Experiment design, ablations, configs (Hydra), seeds, tracking, HPO, versioning | `experimentation-reproducibility.md` |
| PyTorch/JAX, mixed precision, FSDP/tensor/pipeline/3D parallelism, profiling, numerics | `engineering-scale.md` |
| Dataset construction, leakage, splitting, imbalance, augmentation, synthetic data | `data.md` |
| Interpretability/SAEs, robustness/adversarial, OOD, fairness, privacy, alignment | `interpretability-safety.md` |
| Literature, reproducing papers, ideation, paper writing, figures, reviewing, ethics | `research-workflow.md` |
| CV, NLP, speech/audio, time series, tabular, recommenders, multimodal/VLM, scientific ML | `domains.md` |

**`scripts/` — runnable helpers**

- `repro.py` — seed every RNG (Python/NumPy/PyTorch) and dump a reproducible environment + git header.
- `compare_models.py` — paired bootstrap confidence intervals and a permutation significance test for
  honestly comparing two models' predictions.

---

## Repository structure

```text
.
├── .claude-plugin/
│   └── marketplace.json          # marketplace catalog (lists the plugin)
├── plugins/
│   └── machine-learning/
│       ├── .claude-plugin/
│       │   └── plugin.json        # plugin manifest (name, version, author…)
│       ├── skills/
│       │   └── machine-learning/
│       │       ├── SKILL.md        # the skill entry point
│       │       ├── references/     # 17 deep reference docs
│       │       └── scripts/        # repro.py, compare_models.py
│       ├── README.md
│       └── LICENSE
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

This repo is **both a marketplace and a plugin host**: `marketplace.json` lists the local plugin at
`./plugins/machine-learning`. To grow it into a multi-skill library, add more folders under
`plugins/` and list them in `marketplace.json`.

---

## Develop & test locally

Before pushing, validate the structure and try it from a local path:

```bash
# Validate the plugin manifest and structure
claude plugin validate ./plugins/machine-learning
```

```text
# In Claude Code: add this repo as a local marketplace and install from it
/plugin marketplace add ./
/plugin install machine-learning@machine-learning
```

The two scripts are self-contained and runnable:

```bash
python plugins/machine-learning/skills/machine-learning/scripts/repro.py --seed 0 --deterministic
python plugins/machine-learning/skills/machine-learning/scripts/compare_models.py --help
```

---

## Releasing & updating

Versioning is driven by `version` in `plugins/machine-learning/.claude-plugin/plugin.json`
(intentionally **not** duplicated in the marketplace entry — when both are set, the `plugin.json`
value silently wins, and a mismatch is the most common reason a marketplace listing is rejected).

To ship an update: make changes → bump `version` (semver) → update `CHANGELOG.md` → commit and push.
Users pick it up with `/plugin marketplace update machine-learning`.

---

## Listing on claudemarketplaces.com

[claudemarketplaces.com](https://claudemarketplaces.com) is a community directory that indexes public
GitHub repositories containing a valid `.claude-plugin/marketplace.json`. After this repo is public:

1. Confirm `claude plugin validate ./plugins/machine-learning` passes and the repo is public on GitHub.
2. Submit the repository URL via the directory's submission form (look for "Submit" / "Add marketplace").
3. The directory validates the manifest, checks that referenced files exist, and indexes it.

The same repo is installable directly by anyone via `/plugin marketplace add amirhosseinghanipour/machine-learning`
whether or not it's listed in a directory.

---

## Contributing

Issues and pull requests are welcome — corrections, new references, additional domains, or better
scripts. Please keep the **rigor-first** spirit: claims should be accurate and current, comparisons
fair, and any code runnable. Bump the version and update the changelog for releases.

## License

[MIT](LICENSE) © 2026 Amirhossein Ghanipour.

This is an independent, community project and is not affiliated with or endorsed by Anthropic. It is
intended for research, education, and beneficial/defensive ML practice.
