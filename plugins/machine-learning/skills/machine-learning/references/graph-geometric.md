# Graph & Geometric Deep Learning

Learning on non-Euclidean and structured data: graphs, sets, point clouds, meshes, manifolds, and anything with
**symmetry**. The unifying idea (Bronstein et al.'s "Geometric Deep Learning" program): **build the data's
symmetry group into the architecture** so the model is equivariant/invariant to transformations that shouldn't
change the answer. This is the inductive-bias principle from [deep-learning.md](deep-learning.md) §1 taken
seriously — CNNs (translation), GNNs (permutation), and transformers (permutation) are all instances.

---

## 1. Why symmetry/equivariance matters

- **Invariance:** output unchanged by a transformation (a molecule's energy is invariant to rotating it).
- **Equivariance:** output transforms *consistently* with the input (rotate the input → rotated output, e.g.,
  predicting force vectors). Encoding the right equivariance is often worth orders of magnitude in data
  efficiency — it's a hard constraint, not a soft hint, and the model can't waste capacity relearning the
  symmetry. Choosing the symmetry group is the key modeling decision.

## 2. Graph Neural Networks (GNNs) — the message-passing core

Most GNNs are **Message Passing Neural Networks (MPNN)**: each node iteratively (1) **aggregates** messages from
its neighbors with a **permutation-invariant** function (sum/mean/max/attention), then (2) **updates** its
representation. After $k$ layers a node sees its $k$-hop neighborhood. Permutation equivariance over nodes is
built in.
- **GCN** (Kipf & Welling 2017) — spectral-motivated, symmetric-normalized neighbor averaging
  ($\hat D^{-1/2}\hat A\hat D^{-1/2}$); simple, strong baseline; prone to over-smoothing past ~2–3 layers.
- **GraphSAGE** — sampled neighbor aggregation with a separable self vs. neighbor transform; **scales to large
  graphs** via mini-batching neighborhoods, and is **inductive** (generalizes to unseen nodes).
- **GAT / GATv2** — attention-weighted neighbor aggregation (learn which neighbors matter). Use **GATv2**:
  the original GAT's attention is "static" (the ranking of neighbors is input-independent); GATv2 fixes this with
  a simple reordering of the nonlinearity and is a strict generalization — prefer it by default.
- **GIN** (Graph Isomorphism Network) — sum aggregation + MLP; provably as expressive as the **1-Weisfeiler-
  Lehman** test (the standard expressiveness yardstick — see §4). Sum > mean/max for *counting* multiplicities;
  mean/max lose cardinality information.
- **The aggregator is a design choice with teeth.** Sum preserves structure/degree, mean is degree-invariant
  (good when degree is a nuisance), max is selective. **PNA** combines multiple aggregators + degree scalers and
  is a strong default when you don't know which to pick. Add **edge features** via the message function (the full
  MPNN form $m_{ij}=\phi(h_i,h_j,e_{ij})$) whenever edges carry information (bond type, distance, relation).
- **Tasks:** node classification (semi-supervised, transductive), link prediction (recommenders, knowledge
  graphs — score node-pair embeddings), graph classification/regression (molecules — need a **readout/pooling**:
  sum/mean for permutation invariance, or learned hierarchical pooling like DiffPool/SAGPool), and node/edge
  regression.

## 3. The hard parts of GNNs (know these before you build one)

- **Over-smoothing:** stacking many layers makes all node representations converge (Dirichlet energy → 0) → deep
  GNNs often *underperform* shallow ones. Mitigate with residual/initial-residual + identity (**GCNII**),
  **jumping-knowledge** connections, normalization (PairNorm, **GraphNorm**), or just staying shallow (2–4 layers
  is common). It is fundamentally a *low-pass-filter* effect; adding a high-pass channel (e.g., GPR-GNN, FAGCN)
  helps, and also fixes heterophily.
- **Over-squashing:** information from exponentially-growing receptive fields gets compressed through bottleneck
  edges (negatively-curved / high **effective-resistance** regions) → long-range dependencies are lost. The
  modern understanding: over-squashing is governed by the **Jacobian sensitivity** $\partial h_v^{(k)}/\partial
  x_u$ decaying with commute time / effective resistance (Topping et al.; Di Giovanni et al.; Black et al.).
  Fixes: **graph rewiring** — curvature-based (SDRF), spectral-gap (**FoSR**), or Delaunay/expander rewiring — and
  **graph transformers** (global attention bypasses the bottleneck). Caveat backed by recent ablations
  (Tortorella et al. 2024): curvature-rewiring gains are **highly hyperparameter-sensitive** and often vanish
  against a properly tuned baseline — treat rewiring as a tunable, not a free win, and report the un-rewired
  baseline. Note over-smoothing and over-squashing pull in opposite directions (depth vs. connectivity) — you are
  balancing, not eliminating.
- **Scalability:** full-batch training needs the whole graph + activations in memory. Use **neighbor sampling**
  (GraphSAGE — beware exponential fan-out; bound the per-layer samples), **subgraph sampling** (Cluster-GCN,
  GraphSAINT), or **historical embeddings / GAS** (GNNAutoScale) for web-scale graphs. **SIGN/SGC**-style
  precompute-the-propagation tricks turn the graph op into feature engineering and scale trivially.
- **Expressiveness ceiling:** standard message-passing GNNs are bounded by **1-WL** — they cannot distinguish
  certain non-isomorphic graphs, can't count triangles/cycles, and can't detect some substructures. Ways up the
  ladder (see §4): higher-order/$k$-WL GNNs (expensive), **subgraph GNNs** (ESAN, run a GNN over node-deleted/
  ego subgraphs — between 1-WL and 3-WL), and the cheapest practical lever: **positional/structural encodings**
  (Laplacian eigenvectors, random-walk RWPE). Don't over-index on WL — expressivity ≠ generalization; a more
  expressive model can overfit, and PE/SE often help more than raw WL power.
- **Heterophily:** classic GNNs assume connected nodes are similar (homophily); they degrade on heterophilous
  graphs (where neighbors differ). Fixes: separate self/neighbor channels, signed/high-pass messages
  (FAGCN, GPR-GNN), or H2GCN-style ego/neighbor separation + higher-order neighborhoods. A tuned MLP is a
  surprisingly strong baseline on heterophilous benchmarks — always include it.

## 4. Expressivity — the Weisfeiler-Leman ladder (the yardstick)

The standard lens for "what can a GNN tell apart?" is the **WL graph-isomorphism test**. Key facts to carry:
- Message-passing GNNs are **at most as powerful as 1-WL** (Xu et al.; Morris et al. 2019): two graphs 1-WL
  cannot separate, no MPNN can either. GIN with sum aggregation hits this ceiling exactly.
- **What 1-WL misses:** cycle/triangle counting, distinguishing regular graphs, some symmetric substructures —
  exactly the features that matter for molecules (rings) and social structure.
- **Climbing the ladder:** $k$-**WL / $k$-GNNs** (operate on $k$-tuples — $3$-WL ≈ counts triangles, but
  $O(n^k)$ memory, rarely practical past $k\!=\!3$); **subgraph GNNs** (ESAN, GNN-AK, SUN — apply an MPNN to a
  bag of subgraphs; provably between 1- and 3-WL at much lower cost); **positional/structural encodings** as
  cheap node features. There is now a *complete* subgraph-WL hierarchy mapping these methods onto the WL scale.
- **Positional (PE) vs. structural (SE) encodings** — the workhorse upgrade: **Laplacian eigenvectors** (PE;
  fix the sign/basis ambiguity with SignNet/BasisNet, else you inject noise), **random-walk landing
  probabilities** (RWPE/RRWP — robust, no sign ambiguity, used in GraphGPS/GRIT), shortest-path/heat-kernel
  distances. These let even a plain transformer or weak MPNN exceed 1-WL.
- **Caveat:** expressivity is *necessary not sufficient*. More WL power ≠ better test accuracy — it can mean more
  overfitting and worse optimization. Reach for higher expressivity only when the *task* provably needs the
  substructure (e.g., counting), and benchmark against the cheap PE/SE route first.

## 5. Graph Transformers

Apply transformer attention over nodes (global receptive field, one hop reaches every node → sidesteps
over-smoothing/over-squashing) with **graph structure injected via positional/structural encodings** (Laplacian
eigenvectors, random-walk encodings, shortest-path/spatial biases — e.g., **Graphormer**'s centrality + spatial
+ edge-encoding biases). **GraphGPS** (Rampášek et al. 2022) is the de-facto backbone: a hybrid
**local message passing ‖ global attention** block + modular PE/SE — the strong default for graph-level tasks.
- **Scaling past small graphs** (dense attention is $O(n^2)$): **Exphormer** (expander-graph sparse attention —
  scales GraphGPS to 100K+ node graphs like ogbn-arxiv), and linear-attention families for very large graphs —
  **NodeFormer, SGFormer, NAGphormer, Polynormer**. On node-level tasks at scale, these now rival or beat
  sampling-based MPNNs.
- **When they win:** graph-level tasks (especially molecular property prediction) with long-range interactions
  and graphs small enough for full attention; GraphGPS-class models top many OGB/LRGB/molecular leaderboards as
  of 2026. **When a plain GNN is enough:** large homophilous node-classification graphs where locality dominates
  — don't pay for attention you don't need. **GRIT** (fully attention-based with RRWP, no message passing) shows
  the PE can carry the structure alone.

## 6. Equivariant networks for physical/geometric data

For 3D data (molecules, proteins, point clouds, physics) the symmetry is the **Euclidean group E(3)/SE(3)**
(rotation, translation, reflection) plus permutation; force/energy models must be **energy-invariant,
force-equivariant** ($F = -\nabla_x E$, so predict a scalar energy and differentiate — guarantees a conservative
field, unlike directly regressing forces).
- **Invariant approaches** use only invariant geometric features (distances, then angles, then dihedrals) —
  **SchNet** (distances), **DimeNet/DimeNet++** (+ angles), **GemNet** (+ dihedrals, two-hop). Simple and robust;
  the ceiling is that pure invariant scalars lose directional information (a known incompleteness for some
  geometries).
- **Equivariant approaches** carry geometric vectors/higher-order **spherical-tensor** features that transform
  with rotations via the irreps of SO(3)/O(3) (Clebsch–Gordan tensor products) — **Tensor Field Networks,
  SE(3)-Transformer, e3nn** (the reference library), the cheaper vector-only **EGNN/PaiNN**, and the
  interatomic-potential line **NequIP → Allegro** (strictly local, scalable) **→ MACE** (higher body-order
  messages via the Atomic Cluster Expansion — fewer layers, SOTA accuracy/efficiency, the current default for
  MLIPs). The same equivariant machinery is the geometric backbone behind **AlphaFold2/3** and modern protein/
  small-molecule structure and **diffusion-based docking/generation** (DiffDock, equivariant diffusion). These
  are dramatically more **data-efficient** than non-equivariant nets for force/structure prediction.
- **The 2024–2026 shift — universal/foundation MLIPs.** Models pretrained on huge DFT corpora (Materials Project,
  Alexandria, OMat24) now act as **general-purpose force fields** out of the box: **MACE-MP-0**, **MatterSim**,
  **Orb**, **SevenNet**, **GNoME**-derived models; **MACE-OFF** for organic molecules. Benchmark on
  **Matbench-Discovery** before trusting one. Practical caveats: enforce smoothness/energy conservation, validate
  on *your* chemistry (foundation MLIPs extrapolate unevenly), and note an active debate that strict equivariance
  can be traded for scale + data (some large *unconstrained*/local-frame models now compete) — equivariance is a
  strong default, not dogma.
- **Sets & point clouds:** Deep Sets (permutation-invariant via sum-pooling — the canonical theorem), PointNet/
  PointNet++ (point clouds), Set Transformer (attention-based, handles interactions). For large-scale 3D scenes,
  sparse-voxel/transformer hybrids (MinkowskiNet, Point Transformer v3) dominate. The defining constraint is
  permutation invariance over set elements.

## 7. Other structured/geometric settings

- **Knowledge graphs:** embedding methods (TransE, RotatE, ComplEx) for link prediction / relational reasoning;
  R-GCN for multi-relational message passing.
- **Temporal / dynamic graphs:** graphs that evolve (social networks, traffic) — spatio-temporal GNNs combine
  message passing with sequence models.
- **Hyperbolic embeddings:** for hierarchical/tree-like data, hyperbolic space embeds hierarchies with far less
  distortion than Euclidean.
- **Manifold/topological:** learning on meshes (spectral/spatial), and topological data analysis (persistent
  homology) for shape-aware features.

## 8. Tooling, evaluation, and pitfalls

- **Libraries:** **PyTorch Geometric (PyG)** (de-facto default; ships GraphGPS, transformers, samplers) and
  **DGL** (general GNNs), **e3nn** + **MACE/NequIP** (equivariant MLIPs), **Jraph** (JAX), **OGB** + **Long-Range
  Graph Benchmark (LRGB)** for long-range tasks, **Matbench-Discovery** for universal MLIPs.
- **Evaluation pitfalls (graph-specific, easy to get wrong):**
  - **Transductive vs. inductive:** is the test graph seen during training (transductive, semi-supervised node
    classification) or entirely new (inductive)? They measure different things — state which and don't compare
    across them.
  - **Splitting leaks easily:** random node splits in a single connected graph leak neighborhood structure; use
    the standardized OGB splits (often scaffold/temporal/structural splits) and respect them.
  - **Molecular splits:** random splits massively overestimate generalization — use **scaffold splits** to test
    on novel chemical scaffolds (deployment-realistic). This is the [data.md](data.md) leakage lesson in a
    chemistry costume.
  - **Tiny-dataset variance:** many graph benchmarks are small → report many seeds with CIs; "SOTA by 0.3%" is
    usually noise (see [evaluation-statistics.md](evaluation-statistics.md)).
- **Baselines:** a tuned MLP on node features (ignoring the graph) or a simple GCN often matches fancy GNNs —
  always include it. If the graph structure doesn't beat a structure-free baseline, the graph may not help.

## 9. Reach-for table

| Data / goal | Reach for |
|---|---|
| Node/link tasks on a large graph | GraphSAGE / Cluster-GCN / GAS (sampled); SIGN/SGC if precompute fits; GCN/GATv2 + tuned-MLP baseline |
| Very large node-classification graph, want global context | Linear graph transformer (Exphormer / SGFormer / Polynormer) |
| Graph-level (molecular property) | GraphGPS (local ‖ global + RWPE) or GIN; **scaffold split** |
| Long-range graph task | Graph transformer or rewiring; verify on LRGB |
| 3D molecules/materials, forces/energy | Equivariant MLIP — MACE (from-scratch) or a universal MLIP (MACE-MP-0/MatterSim/Orb), validated on your chemistry |
| Protein/ligand structure or docking | Equivariant backbone / equivariant diffusion (AlphaFold-class, DiffDock) |
| Point clouds / sets | PointNet++ / Point Transformer v3 / Deep Sets / Set Transformer |
| Knowledge-graph link prediction | RotatE/ComplEx or R-GCN |
| Hierarchical/tree-like data | Hyperbolic embeddings |
| Need to count substructures / >1-WL | Subgraph GNN (ESAN/GNN-AK) or PE/SE first; $k$-WL only if forced |
| Heterophilous graph | High-pass / signed message GNN (FAGCN, GPR-GNN, H2GCN) + MLP baseline |
| Strong, honest baseline | Tuned MLP on node features (no graph) + plain GCN/GIN |

**Canonical references:** Bronstein, Bruna, Cohen & Veličković 2021 (*Geometric Deep Learning* proto-book — the
symmetry program); Kipf & Welling 2017 (GCN); Hamilton et al. 2017 (GraphSAGE); Veličković et al. 2018 (GAT) and
Brody et al. 2022 (GATv2); Xu et al. 2019 + Morris et al. 2019 (GIN / 1-WL expressiveness); Gilmer et al. 2017
(MPNN); Rampášek et al. 2022 (GraphGPS); Shirzad et al. 2023 (Exphormer); Topping et al. 2022 & Di Giovanni et
al. 2023 (over-squashing / curvature / effective resistance); Satorras et al. 2021 (EGNN); Batzner et al. 2022
(NequIP); Batatia et al. 2022 (MACE) and Batatia et al. 2023 (MACE-MP-0 foundation MLIP); Hu et al. 2020 (OGB);
Dwivedi et al. 2022 (LRGB).
