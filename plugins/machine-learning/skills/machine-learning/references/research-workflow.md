# The ML Research Workflow

Doing and communicating research: finding and reading the literature, generating and positioning ideas,
reproducing prior work, writing papers, making figures, reviewing, and the ethics/open-science obligations.
This complements the rigor in [evaluation-statistics.md](evaluation-statistics.md) and
[experimentation-reproducibility.md](experimentation-reproducibility.md).

---

## 1. Literature search & staying current

- **Where:** arXiv (cs.LG, cs.CL, cs.CV, stat.ML — but **arXiv is unrefereed**, treat claims with skepticism),
  the venues (NeurIPS, ICML, ICLR, ACL/EMNLP, CVPR/ICCV/ECCV, AISTATS, COLT, JMLR/TMLR), OpenReview (reviews +
  rebuttals are a goldmine for understanding a paper's weaknesses), Semantic Scholar / Google Scholar /
  Connected Papers (citation graphs), and Papers with Code (code + leaderboards, but verify the leaderboard's
  protocol).
- **How to find the frontier:** start from a strong recent survey, then trace **forward citations** (who built
  on this) and **backward** (what it builds on). Follow a few groups/authors, not hype threads. Find the
  canonical paper for each method, not a blog summary of it.
- **Healthy skepticism:** extraordinary claims need extraordinary evidence and reproduction. Single-paper SOTA,
  no code, no variance, cherry-picked qualitatives → discount heavily. The reproduction track record of a result
  matters more than its venue.

## 2. Reading a paper efficiently (three passes)

1. **Skim (5–10 min):** title, abstract, figures, conclusion. What's the claim, the setup, the headline result?
   Decide if it's worth more.
2. **Understand (30–60 min):** method and experiments. What *exactly* is new? What's the key equation/mechanism?
   What baselines, datasets, and metrics — and are they fair (the [evaluation-statistics.md](evaluation-statistics.md)
   checklist)?
3. **Critique / reproduce (hours):** Could you reimplement it from the paper? What are the hidden assumptions and
   failure modes? **Read it adversarially:** is the comparison fair, the baseline tuned, the gain significant
   across seeds, the test set clean? Check the appendix and the OpenReview reviews for the real weaknesses.
- **Capture:** a one-paragraph note per paper (problem, idea, why it works, limitations, relation to your work)
  in a reference manager (Zotero) beats re-reading. Maintain a living literature map.

## 3. Reproducing prior work (a research skill in itself)

The MLRC (now an official NeurIPS track) exists because reproduction is hard and valuable.
- **Start from authors' code** if released; pin its environment; reproduce the headline number *first* before
  changing anything. If it doesn't reproduce, that itself is a finding — debug systematically (data version,
  preprocessing, eval protocol, seeds).
- **From scratch:** implement, then climb the debugging ladder (overfit one batch, check loss-at-init — see
  [deep-learning.md](deep-learning.md) §4). Match the paper's exact protocol (splits, metric, preprocessing) or
  comparisons are meaningless.
- **Common reproduction gaps:** undocumented hyperparameters, different data versions/splits, eval-protocol
  differences, and unreported tricks. Document every deviation. Many "SOTA" gaps shrink or vanish under matched,
  honestly-tuned reproduction.

## 4. Ideation & positioning

- **Sources of good problems:** a gap a survey names; a method's stated limitation; a surprising negative
  result; a reproduction that fails; a tool/dataset newly available; transferring an idea across domains; a real
  practitioner pain point. The best problems are *important* and *tractable with your resources*.
- **Position precisely:** what is the **delta** vs. the closest prior work, in one sentence? If you can't state
  it crisply, the contribution isn't clear yet. Know the 3–5 most-related papers cold and how you differ.
- **Strong contribution types:** a new method that wins fairly; a new understanding (why something works); a new
  problem/benchmark/dataset; a rigorous negative result or reproduction; a unifying framework. Not every paper
  needs SOTA — a clear, correct, well-scoped insight is a contribution.
- **De-risk early:** run the cheapest experiment that could *kill* the idea first (see
  [experimentation-reproducibility.md](experimentation-reproducibility.md) §1). Fail fast and cheap.

## 5. Writing the paper

Structure (standard ML paper):
- **Abstract:** problem → gap → what you did → key result (with a number) → why it matters. Write it last; it's
  the most-read part.
- **Introduction:** motivate the problem, state the gap, list contributions as crisp bullets, preview the
  result. The "why should I care" must land in the first page.
- **Related work:** position, don't enumerate — group by approach and say how you differ. Be fair to prior work
  (reviewers wrote it).
- **Method:** precise enough to reimplement. Notation consistent; equations earn their place; a clear method
  figure. State assumptions.
- **Experiments:** the questions each experiment answers; datasets/metrics/baselines/protocol; **ablations** that
  attribute the gain to your mechanism; results with **variance and significance** (see
  [evaluation-statistics.md](evaluation-statistics.md)). Tables and figures should be readable standalone.
- **Limitations:** honest and specific (now expected/required at major venues). Stating real limitations builds
  credibility; hiding them invites desk-reject reviews.
- **Conclusion:** what we learned, what's next.
- **Reproducibility statement + appendix:** full hyperparameters, compute, data details, proofs.

Writing principles: **claims must match evidence** — every claim in the intro/abstract is backed by a specific
result; no overclaiming ("we show X" only if you show X with adequate evidence). Lead with the contribution,
cut throat-clearing, define notation once, and make it skimmable (a reviewer reads under time pressure). Tense
and notation consistent throughout.

## 6. Figures & tables (where reviewers form opinions)

- **One message per figure.** A reader should get the point from the figure + caption alone. Self-contained
  captions.
- **Plot with uncertainty:** error bars / CI bands over seeds; learning curves with variance; never a single
  bar without spread. Show distributions (box/violin) over bar-of-means when variance matters.
- **Honest axes:** don't truncate y-axes to exaggerate gains; consistent scales across compared plots; label
  units. Truncated/misleading axes are a credibility killer.
- **Tables:** bold the best, mark significance, include the strong baseline and the variance, state what's
  compared. Don't copy numbers from other papers run on a different protocol — re-run baselines (see SKILL.md).
- **Accessibility:** colorblind-safe palettes, legible fonts at print size, vector formats.

## 7. Peer review (giving and receiving)

- **Reviewing well:** judge correctness, novelty, significance, clarity, and reproducibility. Be specific and
  constructive; separate "this is wrong" from "I'd have done it differently." Check the [evaluation-statistics.md](evaluation-statistics.md)
  rigor items (fair baselines, variance, leakage, significance). Acknowledge strengths. Calibrate confidence;
  don't reject for missing a citation if the core is sound.
- **Rebuttals:** address the actual concern, run the requested experiment if feasible, concede what's fair,
  and be concise and professional. Reviewers are time-pressed humans; make it easy to raise the score.
- **The norms:** confidentiality, no dual submission, conflict-of-interest disclosure, and (increasingly)
  scrutiny of LLM-assisted reviewing — review with integrity.

## 8. Ethics, broader impact & open science

- **Broader-impact / ethics statements** are required at major venues — engage genuinely: foreseeable misuse,
  dual-use, bias/fairness harms, environmental cost, and affected stakeholders. See
  [interpretability-safety.md](interpretability-safety.md).
- **Responsible release:** datasheets for datasets and model cards for models (intended use, limitations, biases,
  evaluation). Consider release strategy for dual-use capabilities (staged/gated release where warranted).
- **Open science:** release code, configs, seeds, and (where possible) data and checkpoints — it's both an
  ethical norm and the thing that makes your work cited and trusted. Reproducibility is a contribution.
- **Integrity:** report negative results and failures honestly; never p-hack, seed-hack, cherry-pick, or
  tune-on-test (see the anti-patterns in SKILL.md §6). Properly attribute prior work and disclose AI assistance
  per venue policy. The reputational and scientific cost of a non-replicable result far exceeds the short-term
  win.

## 9. Quick research checklist

- [ ] Clear, single-sentence contribution and delta vs. the closest prior work.
- [ ] Reproduced/▢understood the baselines I compare against; comparison is fair and matched.
- [ ] Claims in abstract/intro each backed by a specific result with variance + significance.
- [ ] Ablations isolate the mechanism I credit for the gain.
- [ ] Figures self-contained, with uncertainty and honest axes.
- [ ] Limitations stated honestly; ethics/broader impact engaged.
- [ ] Code/configs/seeds released; results reproduce within CI (see experimentation-reproducibility.md).

**Canonical references:** Keshav "How to Read a Paper"; Whitesides "How to Write a Paper"; the NeurIPS/ICML/ICLR
author guides & checklists; the ML Reproducibility Challenge; Zinkevich "Rules of ML" (applied); venue ethics
and reproducibility guidelines.
