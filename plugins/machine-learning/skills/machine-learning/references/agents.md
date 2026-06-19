# LLM Applications: Prompting, RAG, Agents & Tool Use

How to *build, ground, orchestrate, evaluate, and secure* systems on top of LLMs — the layer above
the model itself. Model internals (attention, post-training, inference, decoding) live in
[transformers-llms.md](transformers-llms.md); this file is about turning a capable model into a
reliable application. The discipline is young and moves monthly; the durable parts are the
**failure modes** and the **evaluation/security mindset**, not any one framework.

**Prime directive carries over:** an agent demo that works once is not a result. Agent success
rates have high run-to-run variance, and the dominant risk is that you are measuring a contaminated
or single-trajectory number. Evaluate end-to-end, with seeds and variance, exactly as in
[evaluation-statistics.md](evaluation-statistics.md).

---

## 1. In-context learning and prompting

- **In-context learning (ICL).** LLMs perform tasks from examples/instructions in the prompt with no
  weight updates — few-shot demonstrations, instructions, and output format steer behavior. Powerful
  but brittle to example ordering, format, and selection; it is not a substitute for fine-tuning when
  you need reliability or when the behavior must hold across many inputs. Mechanistically, ICL behaves
  like implicit gradient descent / Bayesian inference over the demonstrations (see
  [learning-paradigms.md](learning-paradigms.md)) — useful as a mental model, not a guarantee.
- **Prompting discipline.** Clear instructions, explicit output format/schema, the most relevant
  context first, task decomposition (let the model reason step by step), and few-shot exemplars when
  format matters more than knowledge. Put long, static context (system prompt, docs) at the front so
  it can be **prefix-cached** (see [transformers-llms.md](transformers-llms.md) §10).
- **Evaluate prompts like any other intervention** — with a held-out set and the rigor in
  [evaluation-statistics.md](evaluation-statistics.md), not vibes. A prompt change that "looks better"
  on three hand-picked examples is the qualitative-cherry-pick anti-pattern. Version prompts, diff
  them, and A/B with variance.
- **Reasoning models change the prompting contract.** With RLVR-trained "thinking" models, heavy
  chain-of-thought scaffolding in the prompt is often redundant or harmful — the model reasons
  internally. Keep prompts terse and let the model think; spend the budget on tools and verification.
  Visible CoT is **not** a faithful trace of the computation — never use it as an explanation or a
  guardrail signal.

---

## 2. Retrieval-Augmented Generation (RAG)

Retrieve relevant chunks and condition generation on them — the default for grounding on private or
fresh knowledge and for reducing (not eliminating) hallucination.

- **Retrieval stack.** Embed and index a corpus; at query time fetch top-$k$ by similarity, usually
  with a **reranker** on top. **Hybrid retrieval** (sparse BM25 + dense embeddings, fused by
  **reciprocal-rank fusion**) reliably beats either alone, especially on rare terms/entities and
  out-of-distribution jargon that dense embeddings smear. Use a cross-encoder or late-interaction
  reranker (ColBERT/ColPali; see [representation-learning.md](representation-learning.md)) to fix the
  recall/precision tradeoff cheaply.
- **What actually moves quality:** chunking strategy, the embedding model, retrieval recall, and
  reranking — roughly in that order. 2025–26 practice: **contextual retrieval / late chunking** (embed
  each chunk with surrounding document context so it is not stranded), **semantic chunking**, query
  rewriting / **HyDE** (generate a hypothetical answer, embed *that*), and metadata filtering.
- **Evaluate retrieval and generation separately.** Retrieval: recall@k, nDCG, MRR — if the right
  chunk is not retrieved, no prompt fixes it. Generation: **faithfulness/groundedness** (is every
  claim supported by retrieved context?) and answer correctness (RAGAS-style decomposition, or an
  LLM-judge calibrated against humans — §6). Reporting only end-to-end answer accuracy hides whether
  the bottleneck is retrieval or synthesis.
- **Agentic RAG.** The model iterates retrieve → reason → retrieve, issuing its own follow-up queries,
  rather than a single fixed retrieval. Stronger on multi-hop questions; costs more calls and adds the
  agent failure modes in §3.
- **Long context is not a RAG replacement.** Long-context models reduce but do not remove the need for
  retrieval: retrieval is cheaper, fresher, attributable to a source, and avoids **lost-in-the-middle**
  degradation (models under-use the middle of a long window — verify with RULER/needle tests, see
  [transformers-llms.md](transformers-llms.md) §7). Use long context for *reasoning over* retrieved
  material, not as a dumping ground for the whole corpus.

---

## 3. Agents and tool use

An agent plans, calls tools/functions, observes results, and iterates — the **ReAct** loop (reason →
act → observe) is the base pattern; **plan-and-execute** (plan once, then run) and **reflexion**
(self-critique on failure) are common elaborations. Reliability, not cleverness, is the hard part.

- **Tool design dominates.** Few, well-described, composable, **orthogonal** tools beat many
  overlapping ones. Each tool needs an unambiguous name, a typed schema, and a description written for
  the model (state preconditions, side effects, and what the return looks like). Make tools hard to
  misuse: validate inputs, return structured errors the model can act on, and prefer idempotent
  operations. Most "the agent is dumb" bugs are actually tool-surface bugs.
- **Constrain the I/O.** Use constrained decoding / grammars (JSON schema, regex via Outlines/XGrammar;
  see [transformers-llms.md](transformers-llms.md) §10) so tool calls always parse. Cheap and
  high-leverage; just don't over-constrain free-form reasoning steps.
- **Bound the loop.** Cap iterations, wall-clock, token spend, and tool-call count. Detect and break
  oscillation (the agent calling the same tool with the same args). Always have a graceful "I could not
  complete this" terminal state — runaway loops are the most common production incident.
- **Context engineering is the dominant lever** for long-horizon agents. Verbose tool outputs cause
  context overflow, stale state, cost blow-ups, and "context rot" (quality degrades as the window
  fills with low-signal history). Tactics: curate/compress/summarize the running context, **offload to
  external memory or files** and re-read on demand, prune or truncate large tool results, and keep a
  compact running "scratchpad" of decisions rather than the full transcript. Treat the context window
  as a managed cache, not an append-only log.
- **Memory.** Short-term (the window, managed as above) vs. long-term (a vector store / database the
  agent reads and writes across sessions). Long-term memory reintroduces retrieval-quality and staleness
  problems — version it and evaluate it like RAG (§2).
- **Model Context Protocol (MCP).** The de-facto open standard for connecting models to tools and data
  sources (adopted across providers and IDEs), so a tool/server is written once and reused across hosts.
  Treat MCP servers as part of your **trust boundary** — see §5; an MCP server that returns attacker-
  controlled text is an indirect-prompt-injection vector.

### Multi-agent systems

- **Use multiple agents when roles genuinely differ** (e.g. a planner + specialized workers, or a
  generator + an independent critic/verifier). The verifier-as-separate-agent pattern is reliably
  useful because it decorrelates errors.
- **Be skeptical of large agent swarms.** Coordination overhead, compounding per-step error
  (0.95^n collapses fast over long horizons), context duplication, and cost often make a single
  well-scaffolded agent with good tools beat a committee. Add agents to *reduce correlated error or
  parallelize independent subtasks*, not for their own sake. Orchestration patterns: supervisor/worker,
  blackboard/shared-state, and pipeline hand-off — pick the simplest that fits.

---

## 4. When to reach for what

| Situation | Reach for |
|---|---|
| Answer grounded in a private/fresh corpus | RAG (hybrid retrieve + rerank) before any agent |
| One-shot transformation with a known format | A single constrained-decoding call, not an agent |
| Multi-hop question over documents | Agentic RAG |
| Task needs external actions (code, search, APIs) | Tool-use agent (ReAct), bounded loop |
| Correctness is checkable | Generator + independent verifier; consider RLVR (see [reinforcement-learning.md](reinforcement-learning.md)) |
| Behavior must hold reliably across many inputs | Fine-tune (SFT/DPO, see [transformers-llms.md](transformers-llms.md)), don't prompt-engineer forever |
| Long-horizon, many tools | Single agent + aggressive context engineering before multi-agent |

The default is the **least agentic thing that works**: a plain call < a constrained call < RAG < a
single tool-use agent < multi-agent. Every step up adds latency, cost, variance, and attack surface.

---

## 5. Security: agents are an adversarial system

Treat any text that enters the context from outside (retrieved docs, tool outputs, web pages, MCP
servers, user uploads) as **adversarial input**, not data. This is a security problem, not a
prompt-tuning task. (Adversarial robustness for models generally lives in
[interpretability-safety.md](interpretability-safety.md) §2; agent-specific surface is here.)

- **Prompt injection is the #1 threat.** OWASP ranks it #1 on the LLM Top-10 two years running.
  **Indirect (data-borne) prompt injection** — malicious instructions embedded in a retrieved document,
  a tool result, or a web page that the agent then obeys — is largely **unsolved** for tool-using
  agents. The more capable the tools, the higher the blast radius.
- **Attack families to know:** multi-turn **Crescendo** (escalate gradually), **many-shot jailbreaking**
  (fill long context with faux-compliant exemplars — scales with context length), **best-of-N** (cheap
  stochastic resampling across modalities until one slips through), **GCG**-style optimized adversarial
  suffixes, and **automated/agentic red-teamers** (RL- or LLM-driven dialogue search that beats
  single-turn fuzzing).
- **Defend in depth — no single layer holds:**
  - **Least-privilege tool scoping** and **human-in-the-loop for irreversible/high-impact actions**
    (sending money, deleting data, posting externally). This is the highest-leverage control.
  - **Dual-LLM / quarantine patterns:** a privileged planner never sees untrusted content directly; a
    sandboxed, tool-less LLM processes untrusted text and returns only structured, validated data.
  - Input/output classifiers and **constitutional/safety classifiers**; system-prompt hardening;
    constrained decoding to limit what actions are even expressible.
  - Sandbox tool execution; validate and bound every tool's effects; log and make actions auditable.
- **Evaluate safety on agentic benchmarks**, not just chat-jailbreak suites: **AgentDojo** and
  **AgentHarm** (tool-use harm/robustness), **SHADE-Arena** (sabotage/monitoring), plus jailbreak suites
  (HarmBench, JailbreakBench, and **StrongREJECT** to avoid over-counting low-quality "jailbreaks").
  **Red-team with manual + automated/agentic methods** and report results with the same rigor as
  capability evals. Regulatory note: the EU AI Act expects adversarial-robustness testing for
  GPAI/systemic-risk models. For propensity/deception/dangerous-capability evals and the broader safety
  case, see [interpretability-safety.md](interpretability-safety.md) §6.

---

## 6. Evaluating agents

- **End-to-end task success on realistic, multi-step benchmarks — not single-step plausibility.** A
  step that "looks reasonable" tells you nothing about whether the task completed. Standard harnesses:
  **τ-bench / τ²-bench** (tool-agents in customer-service-style domains), **SWE-bench(-Verified)**
  (resolve real GitHub issues), **GAIA** (general assistant), **WebArena / VisualWebArena / OSWorld**
  (web and computer use), **MCP-Universe** (tool breadth). Pick the harness that matches your
  deployment, or build a held-out task suite that does.
- **Report variance.** Agent success rates have high run-to-run noise from sampling, tool flakiness, and
  environment state — report mean ± CI over multiple seeds/rollouts, never a single trajectory. A "70%"
  from one run is not a number.
- **Contamination applies here too.** Public agent benchmarks leak into training data; prefer
  held-out/refreshed task sets and check for contamination (see [data.md](data.md) and
  [evaluation-statistics.md](evaluation-statistics.md)). SWE-bench in particular has documented leakage
  concerns — trust Verified/held-out variants and report the split.
- **Cost and latency are first-class metrics.** Success at 10× the tokens or 30s of latency may be a
  worse product than a cheaper, faster, slightly-less-accurate baseline. Report success *per dollar* and
  *per second*, and always include a non-agentic baseline (single call / RAG) so the agent has to earn
  its complexity.
- **Trajectory and failure analysis.** Log full traces; categorize failures (wrong tool, bad args, loop,
  hallucinated result, gave up, injection) — the distribution of failure types tells you what to fix far
  better than the aggregate score. This is the agent version of slice-wise error analysis.

---

## 7. Anti-patterns and red flags

- **Agentifying what a single constrained call does.** Latency, cost, and variance for no benefit.
- **A multi-agent swarm where one agent + good tools would do.** Coordination overhead and compounding
  error usually make it worse, not better.
- **Demo-driven evaluation.** One successful trajectory, hand-picked, no seeds, no variance.
- **Trusting tool output as data.** The classic indirect-prompt-injection hole; assume it's adversarial.
- **Unbounded loops / no spend cap.** The most common production incident.
- **Reporting only end-to-end answer accuracy for RAG.** Hides whether retrieval or generation is the
  bottleneck; measure them separately.
- **No human-in-the-loop on irreversible actions.** One injection away from real damage.
- **Treating visible chain-of-thought as a faithful explanation or a safety signal.** It is neither.

---

## Canonical references

ReAct (Yao et al. 2022); Reflexion (Shinn et al. 2023); Toolformer (Schick et al. 2023); RAG (Lewis
et al. 2020) and RAGAS; contextual retrieval and reciprocal-rank fusion (current practice); Model
Context Protocol spec (Anthropic, 2024–26); τ-bench / τ²-bench (Sierra); SWE-bench / SWE-bench Verified;
GAIA; WebArena / OSWorld; MCP-Universe; AgentDojo (Debenedetti et al. 2024); AgentHarm; SHADE-Arena;
StrongREJECT; OWASP LLM Top-10; Greshake et al. 2023 (indirect prompt injection); Anil et al. 2024
(many-shot jailbreaking); the dual-LLM / quarantine pattern (Willison). Frameworks move fast — confirm
APIs against current docs before writing code.
