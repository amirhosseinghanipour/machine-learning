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
  bootstrapping + off-policy) which can diverge.

## 4. Policy gradient & actor-critic (the default for continuous control)

Directly optimize the policy: $\nabla_\theta J = \mathbb{E}[\nabla_\theta\log\pi_\theta(a\mid s)\,A^\pi(s,a)]$
(REINFORCE with an advantage baseline to cut variance).
- **Actor-critic:** an *actor* (policy) and a *critic* (value function estimating the baseline/advantage,
  usually via **GAE** — Generalized Advantage Estimation, which trades bias/variance with a $\lambda$ knob). A2C/A3C are the basic forms.
- **PPO** (Proximal Policy Optimization, Schulman et al. 2017): the **workhorse** on-policy algorithm. Clips
  the probability ratio $\pi_\theta/\pi_{\theta_\text{old}}$ to keep updates in a trust region — stable, simple,
  robust. The default for most policy-gradient work and the basis of classic RLHF. TRPO is the theoretical
  ancestor (hard KL constraint).
- **Off-policy actor-critic for continuous control:** **SAC** (Soft Actor-Critic — maximum-entropy, very
  sample-efficient, robust, a top default for robotics/control), **TD3** (twin critics + delayed updates,
  addresses overestimation), **DDPG** (the deterministic ancestor; brittle).

**On-policy (PPO) vs. off-policy (SAC):** PPO is stable and parallelizes across many cheap environments
(simulators); SAC is far more **sample-efficient** (reuses a replay buffer) — prefer it when environment steps
are expensive (real robots). 

## 5. Model-based RL

Learn (or use) a model of dynamics $P$ and plan or generate synthetic experience.
- **Why:** dramatically better **sample efficiency**; planning (MCTS, MPC, CEM) leverages the model.
- **Examples:** **MuZero/AlphaZero** (learned model + MCTS, superhuman games), **Dreamer** (world models +
  latent imagination, SOTA sample efficiency on control/Atari, runs on modest compute), PETS/MBPO.
- **Cost:** model bias compounds over rollouts; harder to implement. Use when samples are precious.

## 6. Offline RL (batch RL)

Learn a policy from a **fixed dataset**, no environment interaction — crucial when exploration is unsafe/
expensive (healthcare, robotics from logs). The core problem is **distribution shift / extrapolation error**:
the policy queries actions absent from the data and the value function extrapolates wildly. Methods constrain
the policy to the data support: **CQL** (conservative Q-learning), **IQL** (implicit Q-learning), **BCQ/TD3+BC**
(behavior cloning regularization), and **Decision Transformer** (recast RL as conditional sequence modeling —
predict actions conditioned on desired return). Evaluate offline RL extremely carefully; off-policy evaluation
is itself hard and a research area.

## 7. Exploration

Beyond ε-greedy/entropy bonuses: **count-based / pseudo-counts**, **curiosity / intrinsic motivation** (ICM,
RND — reward prediction error / novelty), **Thompson sampling / bootstrapped ensembles**, and
**optimism** (UCB). Hard-exploration sparse-reward tasks (Montezuma's Revenge) drove much of this work
(Go-Explore). For most applied tasks, reward shaping and good entropy regularization suffice — but **reward
shaping is where bugs and reward hacking live** (see §10).

## 8. Multi-agent RL (MARL)

Multiple learning agents → non-stationarity (everyone's policy changes), credit assignment across agents, and
cooperation/competition. Paradigms: independent learners, **centralized training with decentralized execution**
(CTDE: QMIX, MADDPG), and self-play (the engine behind superhuman Go/poker/Dota/StarCraft). Game-theoretic
equilibria (Nash, correlated) frame the solution concepts.

## 9. RL for LLMs (RLHF / RLVR) — the highest-impact application now

(Conceptual placement in the LLM lifecycle is in [transformers-llms.md](transformers-llms.md) §9.)
- **RLHF with PPO:** treat the LLM as a policy generating a sequence; reward = a learned **reward model**
  trained on human preference pairs; optimize with PPO plus a **KL penalty to the reference policy** (prevents
  the policy from drifting into reward-model blind spots / gibberish that hacks the reward). Effective but
  complex (4 models in memory: policy, ref, reward, critic) and unstable.
- **GRPO** (Group Relative Policy Optimization): **drops the critic** — sample a *group* of completions per
  prompt, use their **mean reward as the baseline**, and compute advantages from relative scores within the
  group. Cheaper (no value network), stable, and the basis of DeepSeek-R1-style reasoning training. **DAPO** and
  others refine length/entropy/clipping pathologies.
- **RLVR** (verifiable rewards): replace the learned reward model with a **programmatic verifier** (math answer
  checker, code unit tests) — eliminates reward-model hacking and powers emergent reasoning. The key 2025 shift.
- **DPO and friends** sidestep RL entirely for preference alignment — often the pragmatic choice (see
  [transformers-llms.md](transformers-llms.md) §9).

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
  normalization, value clipping, orthogonal init, observation normalization, learning-rate annealing). Start
  from a **trusted implementation** (CleanRL, Stable-Baselines3, TorchRL) — do not reimplement from the paper
  and expect it to work.

## 11. Tooling & reach-for table

- **Libraries:** Gymnasium (envs/API), Stable-Baselines3 (reliable baselines), CleanRL (single-file, readable,
  reproducible), TorchRL, Ray RLlib (distributed/scale), EnvPool (fast vectorized envs).
- **Benchmarks:** Atari/ALE, MuJoCo/DMControl (continuous control), ProcGen (generalization), MetaWorld
  (multi-task), and LLM reasoning suites for RLVR.

| Setting | Reach for |
|---|---|
| Discrete actions, a simulator | PPO (or DQN/Rainbow for sample reuse) |
| Continuous control, cheap sim | PPO |
| Continuous control, expensive samples | SAC (or model-based: Dreamer) |
| Very sample-constrained | Model-based (Dreamer/MuZero) |
| Fixed dataset, no interaction | Offline RL (IQL/CQL) or Decision Transformer |
| Align/teach reasoning to an LLM | GRPO/RLVR (or DPO for preferences) |
| Implementation you can trust | CleanRL / SB3 / TorchRL — not from scratch |

**Canonical references:** Sutton & Barto *Reinforcement Learning: An Introduction* (the bible); Mnih et al.
2015 (DQN); Schulman et al. 2015/2017 (TRPO/PPO, GAE); Haarnoja et al. 2018 (SAC); Hafner et al. (Dreamer);
Schrittwieser et al. 2020 (MuZero); Henderson et al. 2018 + Agarwal et al. 2021 (rliable — RL evaluation rigor).
