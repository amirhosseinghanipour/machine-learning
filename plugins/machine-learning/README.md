# machine-learning (Claude Code plugin)

A research-grade machine learning skill for Claude Code. It teaches Claude to do ML the way a strong
research group does, not just which model to build, but how to build it so the result is correct,
fairly measured, reproducible, and defensible.

## What this plugin provides

A single skill, `machine-learning`, using progressive disclosure:

- **`skills/machine-learning/SKILL.md`** — the always-loaded anchor: a rigor-first methodology (the
  "don't fool yourself" core), a 7-phase research workflow, a rigor checklist, an anti-pattern
  catalog, the current (2026) stack, and a router to the references.
- **`skills/machine-learning/references/`** — 17 deep documents loaded on demand: foundations,
  classical ML, deep learning, transformers/LLMs, generative models, reinforcement learning,
  probabilistic ML, graph/geometric learning, representation learning, learning paradigms,
  evaluation & statistics, experimentation & reproducibility, engineering & scale, data,
  interpretability & safety, the research workflow, and per-domain guidance.
- **`skills/machine-learning/scripts/`** — `repro.py` (seed all RNGs + dump a reproducible
  environment/git header) and `compare_models.py` (paired bootstrap CIs + permutation significance
  test for model comparison).

## When it activates

Automatically, when you design, implement, train, evaluate, debug, or scale ML models; choose
methods/architectures/losses/optimizers; run experiments or ablations; analyze results or
significance; reproduce or critique papers; or write up ML research. Triggers on ML/DL/RL/LLM/
diffusion work, PyTorch/JAX, training runs, model evaluation, and academic ML tasks.

## Install

```text
/plugin marketplace add amirhosseinghanipour/machine-learning
/plugin install machine-learning@machine-learning
```

See the [repository README](../../README.md) for full documentation, local testing, and contribution
notes.

## License

[MIT](LICENSE) © 2026 Amirhossein Ghanipour. Independent community project; not affiliated with Anthropic.
