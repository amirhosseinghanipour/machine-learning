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
  canonical paper for each method, not a blog summary of it. Tools: Semantic Scholar / **Connected Papers** /
  **Litmaps** for citation-graph exploration, **alphaXiv / arXiv-sanity** for discussion, Zotero (with the
  Better BibTeX plugin) as the source of truth.
- **AI-assisted search, used carefully (2026):** LLM "deep research" agents and tools (Elicit, Undermind,
  SciSpace, etc.) are excellent for *recall* — casting a wide net and surfacing papers you'd miss — but they
  **hallucinate citations and misstate results**. Use them to find candidates, then **verify every claim and
  bibentry against the actual PDF**. Never cite a paper you haven't opened. Disclose substantive LLM assistance
  per venue policy.
- **Healthy skepticism (more necessary than ever):** arXiv volume has exploded with LLM-assisted writing, so the
  noise floor is higher — extraordinary claims need extraordinary evidence and reproduction. Single-paper SOTA,
  no code, no variance, cherry-picked qualitatives → discount heavily. The **reproduction track record** of a
  result matters more than its venue or citation count. Prefer results that survived independent replication,
  open code/checkpoints, and ablations over headline-grabbing first claims.

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

The **ML Reproducibility Challenge (MLRC)** is now an official NeurIPS 2026 track (reproductions are published
via TMLR, then admitted) — reproduction is recognized first-class science, not busywork.
- **Start from authors' code** if released; pin its environment; reproduce the headline number *first* before
  changing anything. If it doesn't reproduce, that itself is a finding — debug systematically (data version,
  preprocessing, eval protocol, seeds, library/CUDA versions, and **non-determinism** from GPU ops, see
  [experimentation-reproducibility.md](experimentation-reproducibility.md)).
- **Reproduce the eval, not just the train.** A large share of irreproducibility is **evaluation-protocol
  drift**: different decontamination, prompt templates, metric implementations, or scoring scripts. Run the
  *authors'* eval harness on a known checkpoint and match their number before trusting your own pipeline.
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
- **Accessibility & craft:** colorblind-safe palettes (avoid red/green pairing; viridis/cividis for sequential,
  a perceptually-uniform diverging map otherwise), legible fonts **at print size** (match the paper's font, ~7–9pt
  in-figure; check at the column width you'll actually use), and **vector formats** (PDF/SVG, not rasterized
  PNG of a plot). Embed fonts. Keep one visual language (colors=methods, linestyles=variants) consistent across
  every figure so the reader learns it once.
- **The hero figure earns acceptance.** Page-1 method/teaser figures are where a rushed reviewer forms their
  prior — invest disproportionately. It should convey the core idea without the text. Avoid 3D bar charts,
  dual y-axes, and chartjunk. For learning curves use steps/tokens/FLOPs (not just epochs) on the x-axis so
  comparisons are compute-matched (see SKILL.md §2).

## 7. Peer review (giving and receiving)

- **Reviewing well:** judge correctness, novelty, significance, clarity, and reproducibility. Be specific and
  constructive; separate "this is wrong" from "I'd have done it differently." Check the [evaluation-statistics.md](evaluation-statistics.md)
  rigor items (fair baselines, variance, leakage, significance). Acknowledge strengths. Calibrate confidence;
  don't reject for missing a citation if the core is sound.
- **Rebuttals:** address the actual concern, run the requested experiment if feasible, concede what's fair,
  and be concise and professional. Open with a short summary of changes; map each response to the specific
  reviewer point; put new results in a clearly-marked table. Reviewers are time-pressed humans; make it easy to
  raise the score. Don't relitigate the reviewer's taste — answer factual/technical objections with evidence.
- **The norms (2026):** confidentiality (don't upload submissions to external LLM services — it breaks
  confidentiality and most venues now **prohibit pasting papers into third-party LLMs**), no dual submission,
  conflict-of-interest disclosure. **LLM use is policy-governed at NeurIPS/ICML/ICLR 2026:** authors may use
  LLMs but **remain fully responsible** for all content (AI-generated plagiarism or fabricated citations =
  misconduct; "AI slop" submissions are explicitly discouraged/desk-rejectable), and **methodologically
  significant LLM/agent use must be disclosed**. Reviewers must follow their console's stated LLM policy;
  fabricated or clearly LLM-generated reviews are sanctionable. Review with integrity — a hallucinated weakness
  wastes everyone's time.

## 8. Ethics, broader impact & open science

- **Broader-impact / ethics statements** are required at major venues — engage genuinely: foreseeable misuse,
  dual-use, bias/fairness harms, environmental cost, and affected stakeholders. See
  [interpretability-safety.md](interpretability-safety.md).
- **Responsible release:** datasheets for datasets and model cards for models (intended use, limitations, biases,
  evaluation). Consider release strategy for dual-use capabilities (staged/gated release where warranted).
- **Open science:** release code, configs, seeds, and (where possible) data and checkpoints — it's both an
  ethical norm and the thing that makes your work cited and trusted. Practical: a **README that reproduces the
  headline table with one command**, pinned environment (lockfile/Docker), the exact eval harness, and a
  permissive license; archive an immutable snapshot (Zenodo DOI / `git tag`); release **model + data cards** and
  a `requirements`/`environment` lock. Anonymize for double-blind review (anonymous repo). For models, state the
  license and any use restrictions; for data, the provenance and consent basis.
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

**Canonical references:** Keshav "How to Read a Paper"; Whitesides "How to Write a Paper"; Simon Peyton Jones
"How to write a great research paper" (talk); the **NeurIPS/ICML/ICLR 2026 author & reviewer guides, checklists,
and LLM-use policies**; the **ML Reproducibility Challenge** (NeurIPS 2026 track via TMLR); the NeurIPS paper
checklist & reproducibility checklist; Zinkevich "Rules of ML" (applied); Sculley et al. "Winner's Curse?"
(empirical rigor) and "Hidden Technical Debt in ML Systems"; venue ethics and broader-impact guidelines.
