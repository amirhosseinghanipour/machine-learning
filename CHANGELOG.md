# Changelog

All notable changes to the `machine-learning` plugin are documented here. This project follows
[Semantic Versioning](https://semver.org). The plugin version lives in
`plugins/machine-learning/.claude-plugin/plugin.json` — bump it on every release so users receive updates.

## [1.0.0] — 2026-06-06

### Added
- Initial release of the **machine-learning** skill.
- `SKILL.md`: rigor-first methodology (the "don't fool yourself" core), a 7-phase research workflow,
  a rigor checklist, an anti-pattern catalog, and a router to 17 reference documents.
- 17 `references/` documents covering foundations, classical ML, deep learning, transformers/LLMs,
  generative models, reinforcement learning, probabilistic ML, graph/geometric learning,
  representation learning, learning paradigms, evaluation & statistics, experimentation &
  reproducibility, engineering & scale, data, interpretability & safety, the research workflow, and
  per-domain guidance.
- 2 runnable `scripts/`: `repro.py` (seed all RNGs + dump a reproducible environment/git header) and
  `compare_models.py` (paired bootstrap CIs + permutation significance test for model comparison).
- Marketplace packaging for installation via Claude Code (`/plugin marketplace add` + `/plugin install`).
