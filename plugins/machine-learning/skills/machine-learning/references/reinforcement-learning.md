# Reinforcement Learning

Learning to act from reward. RL is powerful but **notoriously finicky** — high variance, sensitive to
seeds/hyperparameters, and easy to fool yourself with. Treat RL results with extra skepticism: many seeds,
proper baselines, and the reproducibility discipline in [experimentation-reproducibility.md](experimentation-reproducibility.md).
The RLHF/GRPO connection to LLMs is in §9 and [transformers-llms.md](transformers-llms.md).

---

## 1. The formalism (MDPs)

An agent in state $s_t$ takes action $a_t\sim\pi(a\mid s)$, gets reward $r_t$, transitions to $s_{t+1}\sim
P(\cdot\mid s_t,a_t)$. Goal: maximize expected discounted return $\mathbb{E}[\sum_t \gamma^t r_t]$.
- **Value functions:** $V^\pi(s)$ = expected return from $s$; $Q^\pi(s,a)$ = from taking $a$ in $s$. The
  **Bellman equation** $Q^\pi(s,a)=\mathbb{E}[r+\gamma Q^\pi(s',a')]$ is the recursive backbone of value-based
  methods.
- **Advantage** $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ — how much better an action is than average; central to policy
  gradients (lower variance than raw return).
- **Key tensions:** exploration vs. exploitation; credit assignment over long horizons; sample efficiency vs.
  stability; on-policy (data from current policy) vs. off-policy (reuse old data).

## 2. Tabular foundations (the intuitions that carry over)

- **Dynamic programming** (value/policy iteration) when the model $P$ is known.
- **Monte Carlo** (learn from complete episode returns) and **Temporal Difference** (bootstrap from
  next-state estimates). **TD vs. MC** is a bias/variance trade: TD is lower-variance, biased, online; MC is
  unbiased, higher-variance. **TD(λ)/eligibility traces** interpolate.
- **Q-learning** (off-policy, learns $Q^*$) vs. **SARSA** (on-policy). Q-learning + function approximation is
  the seed of DQN.

## 3. Deep value-based methods

- **DQN** (Mnih et al. 2015): a neural net approximates $Q$, trained on the Bellman error with two crucial
  tricks — a **replay buffer** (decorrelate samples, reuse data) and a **target network** (stabilize the
  bootstrap target). Works for discrete actions.
- **Improvements (Rainbow bundles them):** Double DQN (reduce overestimation), Dueling (separate V and A),
  Prioritized Experience Replay, distributional RL (C51/QR-DQN — model the return *distribution*, not just its
  mean), n-step returns, noisy nets for exploration.
- **Limitations:** discrete actions only; sensitive to the "deadly triad" (function approximation +
  bootstrapping + off-policy) which can diverge. **Overestimation** (max over noisy Q) is the practical culprit
  behind most divergence — Double DQN / clipped-double-Q (the trick TD3 and SAC also use) is the first fix to
  reach for.
- **Scaling Q-learning is hard but back in fashion.** Naïvely widening the net hurt for years; recent work
  (**BBF**, plus normalization/regularization recipes like layer-norm on the critic, **CrossQ**, high UTD with
  resets) makes value-based methods scale and reach high sample efficiency again — relevant when off-policy reuse
  matters more than the simplicity of PPO.

## 4. Policy gradient & actor-critic (the default for continuous control)

Directly optimize the policy: $\nabla_\theta J = \mathbb{E}[\nabla_\theta\log\pi_\theta(a\mid s)\,A^\pi(s,a)]$
(REINFORCE with an advantage baseline to cut variance).
- **Actor-critic:** an *actor* (policy) and a *critic* (value function estimating the baseline/advantage,
  usually via **GAE** — Generalized Advantage Estimation, which trades bias/variance with a $\lambda$ knob). A2C/A3C are the basic forms.
- **PPO** (Proximal Policy Optimization, Schulman et al. 2017): the **workhorse** on-policy algorithm. Clips
  the probability ratio $r=\pi_\theta/\pi_{\theta_\text{old}}$ to keep updates in a trust region — stable, simple,
  robust. The default for most policy-gradient work and the basis of classic RLHF. TRPO is the theoretical
  ancestor (hard KL constraint via a conjugate-gradient natural-gradient step). **Defaults that usually work**
  (continuous control): clip $\epsilon{=}0.2$, GAE $\lambda{=}0.95$, $\gamma{=}0.99$, Adam $3\times10^{-4}$,
  3–10 epochs over each rollout, 2048–4096 steps/update, minibatch 64; **normalize advantages per-batch** and
  **clip the value loss**. PPO is on-policy, so reusing data for too many epochs silently makes the ratio stale
  — fewer epochs if KL spikes; many implementations add a target-KL early-stop.
- **Off-policy actor-critic for continuous control:** **SAC** (Soft Actor-Critic — maximum-entropy, very
  sample-efficient, robust, a top default for robotics/control; **auto-tune the temperature** $\alpha$ to a
  target entropy rather than fixing it, and use twin critics), **TD3** (twin critics + delayed actor updates +
  target-policy smoothing, addresses overestimation), **DDPG** (the deterministic ancestor; brittle, superseded
  by TD3/SAC). **REDQ / DroQ / CrossQ** push sample efficiency further with high **update-to-data (UTD)** ratios
  + critic ensembles/normalization — the modern picks when each env step is precious.
- **Max-entropy framing** (SAC): augment reward with policy entropy, $J=\mathbb E[\sum_t r_t+\alpha\mathcal
  H(\pi(\cdot|s_t))]$. The entropy bonus is principled exploration *and* the reason SAC is robust to
  hyperparameters; it is the control-task analog of the KL/entropy terms that stabilize RLHF (§9).

**On-policy (PPO) vs. off-policy (SAC):** PPO is stable and parallelizes across many cheap environments
(simulators) — with massively parallel sims (Isaac Gym, Brax, hundreds–thousands of envs) it trains locomotion
policies in minutes; SAC is far more **sample-efficient** (reuses a replay buffer) — prefer it when environment
steps are expensive (real robots). Rule of thumb: **abundant cheap steps → PPO; scarce expensive steps → SAC/
off-policy with high UTD**.

## 5. Model-based RL

Learn (or use) a model of dynamics $P$ and plan or generate synthetic experience.
- **Why:** dramatically better **sample efficiency**; planning (MCTS, MPC, CEM) leverages the model.
- **Examples:** **MuZero/AlphaZero** (learned model + MCTS, superhuman games; MuZero learns a *value-equivalent*
  latent model — it predicts reward/value/policy, not pixels — so planning targets only what matters for the
  decision), **Dreamer** (world models + latent imagination, SOTA sample efficiency on control/Atari, runs on
  modest compute), PETS/MBPO. **EfficientZero** brought MuZero to human-level Atari in ~2 hours of data;
  **Sampled MuZero** extends it to continuous/large action spaces.
- **DreamerV3 (Hafner et al. 2023, *Nature* 2025)** is the headline result: **one fixed hyperparameter set**
  masters 150+ tasks across domains (and was first to mine diamonds in Minecraft from scratch, no human data).
  Its robustness comes from a bag of normalization tricks worth copying generally: **symlog** prediction to
  handle unknown reward/value magnitudes, **two-hot** discrete regression for value/reward (more stable than MSE
  on heavy-tailed returns), **percentile return normalization**, free-bits on the KL, and fixed-entropy
  balancing. Scales monotonically with model size — bigger world model = better *and* more sample-efficient.
- **Cost:** model bias compounds over rollouts; harder to implement. Use when samples are precious — but
  DreamerV3 has largely removed the "harder to tune" objection for the world-model approach.

## 6. Offline RL (batch RL)

Learn a policy from a **fixed dataset**, no environment interaction — crucial when exploration is unsafe/
expensive (healthcare, robotics from logs). The core problem is **distribution shift / extrapolation error**:
the policy queries actions absent from the data and the value function extrapolates over-optimistically (the
Q-function has no data to correct it). Methods constrain the policy to the data support: **CQL** (conservative
Q-learning — push down Q on OOD actions), **IQL** (implicit Q-learning — avoid querying OOD actions at all via
expectile regression + advantage-weighted extraction; the robust default), **BCQ/TD3+BC** (behavior-cloning
regularization), and **Decision Transformer** (recast RL as return-conditioned sequence modeling — simple,
strong on dense-reward data, but **weak at stitching/sparse rewards** and only as good as the returns in the
data; not a universal replacement for value-based offline RL).
- **Defaults:** start with **IQL** (few knobs, rarely diverges); reach for CQL when you need more conservatism
  and can afford to tune $\alpha$. Always normalize observations and reward-scale to the dataset.
- **Offline-to-online (the practical sweet spot):** pretrain offline, then fine-tune online. Naïve fine-tuning
  causes a **distribution-shift dip** as the over-conservative policy meets real data. **Cal-QL** (calibrated
  CQL) fixes the value scale so online fine-tuning improves monotonically; **RLPD** instead does online RL from
  scratch with the offline data mixed into the replay buffer at a high UTD ratio — often the strongest and
  simplest. **PA-RL** generalizes offline+online fine-tuning to any policy class (diffusion/transformer).
- **Evaluation is the trap.** You cannot reliably score an offline policy offline — **off-policy evaluation
  (OPE)** is itself an open problem (importance sampling has crippling variance; fitted-Q evaluation is biased).
  Report normalized scores on a standard suite (**D4RL/Minari**) and treat any number obtained by tuning against
  the env as leakage. Do not select hyperparameters using online rollouts you wouldn't have in deployment.

## 7. Exploration

Beyond ε-greedy/entropy bonuses: **count-based / pseudo-counts**, **curiosity / intrinsic motivation** (ICM,
RND — reward prediction error / novelty), **Thompson sampling / bootstrapped ensembles**, and
**optimism** (UCB). Hard-exploration sparse-reward tasks (Montezuma's Revenge) drove much of this work
(Go-Explore). For most applied tasks, reward shaping and good entropy regularization suffice — but **reward
shaping is where bugs and reward hacking live** (see §10). Prefer **potential-based shaping**
($F=\gamma\Phi(s')-\Phi(s)$), the one form proven not to change the optimal policy. In the LLM-RL setting the
"exploration" problem reappears as **entropy collapse** (§9): keep the policy from prematurely committing.

## 8. Multi-agent RL (MARL)

Multiple learning agents → non-stationarity (everyone's policy changes), credit assignment across agents, and
cooperation/competition. Paradigms: independent learners, **centralized training with decentralized execution**
(CTDE: QMIX, MADDPG), and self-play (the engine behind superhuman Go/poker/Dota/StarCraft). Game-theoretic
equilibria (Nash, correlated) frame the solution concepts.

## 9. RL for LLMs (RLHF / RLVR) — the highest-impact application now

(Conceptual placement in the LLM lifecycle is in [transformers-llms.md](transformers-llms.md) §9.)
- **RLHF with PPO:** treat the LLM as a policy generating a sequence; reward = a learned **reward model** (RM)
  trained on human preference pairs (Bradley–Terry on chosen/rejected); optimize with PPO plus a **per-token KL
  penalty to the reference policy** (prevents the policy from drifting into RM blind spots / gibberish that hacks
  the reward — and the KL coefficient is the single most important knob). Effective but complex (4 models in
  memory: policy, ref, reward, critic) and unstable. The bandit framing (one reward at the end of the sequence)
  + a per-token value head is the standard recipe; **reward over-optimization** (Goodhart on a finite RM) is the
  characteristic failure — RM score keeps rising while true quality falls.
- **GRPO** (Group Relative Policy Optimization, DeepSeek): **drops the critic** — sample a *group* of $G$
  completions per prompt (typ. 8–16), use their **mean reward as the baseline**, and set each completion's
  advantage to the **group-standardized reward** $(r_i-\mu)/\sigma$. Cheaper (no value network), stable, the
  basis of DeepSeek-R1-style reasoning training. **Known biases to correct** (this is where 2025 work lives):
  - **Dr.GRPO** removes GRPO's **length** and **std normalization** biases — dividing by response length and by
    group std injects gradients that make wrong answers grow longer and over-weight low-variance prompts. Drop
    both for cleaner, less length-inflating updates.
  - **DAPO** adds four fixes that matter in practice: **clip-higher** (raise only the *upper* PPO clip so
    low-prob "exploratory" tokens aren't suppressed — directly fights entropy collapse), **dynamic sampling**
    (discard prompts where all $G$ samples are all-correct or all-wrong → zero advantage → no gradient; resample
    to keep batches informative), **token-level loss** (average over tokens, not sequences, so long correct
    reasoning isn't down-weighted), and **overlong reward shaping**.
  - **Entropy collapse is the dominant failure mode.** Policy entropy crashes early, the model stops exploring,
    pass@k *degrades* even as pass@1 rises. Mitigate with clip-higher, a (carefully tuned) entropy bonus, or
    KL/temperature control — over-large entropy coefficients cause entropy *explosion*, so it is finicky.
- **RLVR** (verifiable rewards): replace the learned RM with a **programmatic verifier** (math answer checker,
  code unit tests, format/regex) — eliminates RM hacking and powers emergent reasoning. The key 2025 shift, and
  why GRPO+RLVR became the reasoning-training default.
  - **Outcome (ORM) vs process (PRM) rewards.** ORM scores only the final answer — cheap and hard to hack, but
    **rewards spurious successes** (right answer via wrong reasoning) and gives coarse credit. PRM scores each
    step — better credit assignment, but data is scarce/expensive and PRMs are themselves **reward-hackable**.
    2025 practice leans ORM/RLVR for scale, with PRM/hybrid for faithfulness.
  - **The elicitation debate (know this before claiming "RL taught new reasoning").** Strong evidence that RLVR
    largely **sharpens/reweights reasoning already latent in the base model** rather than creating new
    capability: pass@1 improves while **pass@k (large k) often matches or trails the base model**, i.e. the
    reachable solution set doesn't grow. Implication for rigor: report **pass@k across k**, not just pass@1/greedy;
    a method that wins at k=1 but loses at k=256 has narrowed the distribution, not expanded ability.
- **DPO and friends** sidestep online RL entirely for preference alignment — DPO derives a closed-form
  KL-regularized objective directly on preference pairs (no RM, no sampling loop), with SimPO/KTO/ORPO/IPO
  refining the reference-model and length-bias handling. Often the pragmatic choice for *preferences*; RLVR/GRPO
  is the choice for *verifiable reasoning*. See [transformers-llms.md](transformers-llms.md) §9.

## 10. RL-specific rigor (read before reporting any RL result)

RL has a **reproducibility crisis** (Henderson et al. 2018) — it is exceptionally easy to publish noise.
- **Many seeds.** Report **≥5–10 seeds** with confidence intervals / IQM, not max-over-seeds and not 3 seeds.
  Use the **rliable** methodology (interquartile mean, stratified bootstrap CIs, performance profiles) — mean ±
  std hides bimodal seed outcomes.
- **Fair baselines & tuning:** equal environment steps, equal tuning effort; report sample efficiency curves
  (return vs. environment steps), not just final return.
- **Evaluation protocol:** separate train/eval episodes, report evaluation with a *deterministic* or fixed-seed
  eval policy, average over many eval episodes (returns are high-variance).
- **Reward hacking / specification gaming:** agents optimize the literal reward, not your intent — watch for
  degenerate solutions, and design/verify rewards adversarially. This is the RL-flavored version of "don't fool
  yourself."
- **Implementation details dominate:** PPO's performance is largely in the "code-level optimizations" (advantage
  normalization, value clipping, orthogonal init, observation/reward normalization, learning-rate annealing,
  GAE). Engstrom et al. 2020 / Andrychowicz et al. 2021 showed these matter *more* than the algorithmic choice
  between PPO and TRPO. Start from a **trusted implementation** (CleanRL, Stable-Baselines3, TorchRL) — do not
  reimplement from the paper and expect it to work, and when comparing algorithms, equalize the code-level tricks
  or you are benchmarking implementations, not ideas.
- **RL-for-LLM rigor (the new front):** report **pass@k across multiple k** (not just greedy/pass@1) so you can
  tell capability gain from distribution narrowing; track **policy entropy** and **KL-to-reference** as
  first-class curves (collapse/blow-up is invisible in the reward curve alone); watch for **reward hacking** of
  the verifier (format exploits, degenerate "answers"); and beware **benchmark contamination** of the eval math/
  code sets. Seeds and prompt-set variance matter here too — single-run RLVR deltas are as noisy as classic RL.

## 11. Tooling & reach-for table

- **Libraries:** Gymnasium (envs/API), Stable-Baselines3 (reliable baselines), CleanRL (single-file, readable,
  reproducible), TorchRL, Ray RLlib (distributed/scale), EnvPool (fast vectorized envs).
- **Benchmarks:** Atari/ALE, MuJoCo/DMControl (continuous control), ProcGen (generalization), MetaWorld
  (multi-task), and LLM reasoning suites for RLVR.

| Setting | Reach for |
|---|---|
| Discrete actions, a simulator | PPO (or DQN/Rainbow for sample reuse) |
| Continuous control, cheap sim | PPO |
| Continuous control, expensive samples | SAC (or high-UTD off-policy: REDQ/CrossQ; model-based: Dreamer) |
| Very sample-constrained | Model-based (DreamerV3 — fixed hypers / MuZero family) |
| Fixed dataset, no interaction | Offline RL (**IQL** default, CQL for more conservatism) or Decision Transformer |
| Offline pretrain → online fine-tune | Cal-QL or RLPD (offline data in the buffer) |
| Teach verifiable reasoning to an LLM | GRPO+RLVR (use Dr.GRPO/DAPO fixes) |
| Align an LLM to preferences | DPO/SimPO (or PPO-RLHF if you need an explicit reward model) |
| Implementation you can trust | CleanRL / SB3 / TorchRL — not from scratch |

**Canonical references:** Sutton & Barto *Reinforcement Learning: An Introduction* (2nd ed. — the bible); Mnih
et al. 2015 (DQN), Hessel et al. 2018 (Rainbow); Schulman et al. 2015/2017 (TRPO/PPO, GAE); Haarnoja et al.
2018 (SAC); Fujimoto et al. 2018 (TD3); Hafner et al. 2023 (DreamerV3, *Nature* 2025); Schrittwieser et al. 2020
(MuZero); Kumar et al. 2020 (CQL), Kostrikov et al. 2022 (IQL), Nakamoto et al. 2023 (Cal-QL), Ball et al. 2023
(RLPD); Chen et al. 2021 (Decision Transformer); Henderson et al. 2018 + Agarwal et al. 2021 (rliable — RL
evaluation rigor); Engstrom et al. 2020 / Andrychowicz et al. 2021 (implementation details dominate).
**RL-for-LLMs:** Ouyang et al. 2022 (InstructGPT/RLHF-PPO); Rafailov et al. 2023 (DPO); Shao et al. 2024 (GRPO);
DeepSeek-AI 2025 (R1 — RLVR reasoning); Yu et al. 2025 (DAPO); Liu et al. 2025 (Dr.GRPO). **Texts:** Sutton &
Barto (above) for foundations.
