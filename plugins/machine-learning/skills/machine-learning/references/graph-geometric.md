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
- **GCN** (Kipf & Welling 2017) — spectral-motivated, normalized neighbor averaging; simple, strong baseline.
- **GraphSAGE** — sampled neighbor aggregation; **scales to large graphs** via mini-batching neighborhoods.
- **GAT** — attention-weighted neighbor aggregation (learn which neighbors matter).
- **GIN** (Graph Isomorphism Network) — sum aggregation + MLP; provably as expressive as the **1-Weisfeiler-
  Lehman** test (the standard expressiveness yardstick — see §4).
- **Tasks:** node classification (semi-supervised, transductive), link prediction (recommenders, knowledge
  graphs), graph classification/regression (molecules), and node/edge regression.

## 3. The hard parts of GNNs (know these before you build one)

- **Over-smoothing:** stacking many layers makes all node representations converge to the same value → deep GNNs
  often *underperform* shallow ones. Mitigate with residual/jumping-knowledge connections, normalization,
  PairNorm, or just staying shallow (2–4 layers is common).
- **Over-squashing:** information from exponentially-growing receptive fields gets compressed through
  bottleneck edges → long-range dependencies are lost. Graph rewiring and **graph transformers** address this.
- **Scalability:** full-batch training needs the whole graph in memory. Use neighbor sampling (GraphSAGE),
  subgraph sampling (Cluster-GCN, GraphSAINT), or historical embeddings for web-scale graphs.
- **Expressiveness ceiling:** standard message-passing GNNs cannot distinguish certain non-isomorphic graphs
  (bounded by 1-WL) — they can't count triangles or detect some substructures. Higher-order GNNs, subgraph
  GNNs, and positional/structural encodings raise the ceiling.
- **Heterophily:** classic GNNs assume connected nodes are similar (homophily); they degrade on heterophilous
  graphs (where neighbors differ) — use specialized architectures.

## 4. Graph Transformers

Apply transformer attention over nodes (global receptive field, sidesteps over-smoothing/over-squashing) with
**graph structure injected via positional/structural encodings** (Laplacian eigenvectors, random-walk
encodings, shortest-path biases — e.g., Graphormer, GraphGPS). Increasingly SOTA on graph-level tasks
(especially molecular property prediction) where long-range interactions matter and graphs are small enough for
$O(n^2)$ attention. The hybrid (local message passing + global attention) is a strong default.

## 5. Equivariant networks for physical/geometric data

For 3D data (molecules, proteins, point clouds, physics) the symmetry is the **Euclidean group E(3)/SE(3)**
(rotation, translation, reflection):
- **Invariant approaches** use only invariant features (distances, angles) — **SchNet, DimeNet** (molecular
  property prediction).
- **Equivariant approaches** operate on geometric vectors/tensors that rotate with the input — **E(3)/SE(3)-
  equivariant nets, Tensor Field Networks, e3nn, EGNN, Allegro/NequIP** (interatomic potentials), and the
  geometric backbones behind **AlphaFold** and modern protein/molecule structure models. These are dramatically
  more data-efficient for force/structure prediction.
- **Sets & point clouds:** Deep Sets (permutation-invariant via sum-pooling), PointNet/PointNet++ (point
  clouds), Set Transformer. The defining constraint is permutation invariance over set elements.

## 6. Other structured/geometric settings

- **Knowledge graphs:** embedding methods (TransE, RotatE, ComplEx) for link prediction / relational reasoning;
  R-GCN for multi-relational message passing.
- **Temporal / dynamic graphs:** graphs that evolve (social networks, traffic) — spatio-temporal GNNs combine
  message passing with sequence models.
- **Hyperbolic embeddings:** for hierarchical/tree-like data, hyperbolic space embeds hierarchies with far less
  distortion than Euclidean.
- **Manifold/topological:** learning on meshes (spectral/spatial), and topological data analysis (persistent
  homology) for shape-aware features.

## 7. Tooling, evaluation, and pitfalls

- **Libraries:** **PyTorch Geometric (PyG)** and **DGL** (general GNNs), **e3nn**/NequIP (equivariant),
  **Jraph** (JAX), **OGB** (Open Graph Benchmark — standardized datasets/splits/leaderboards).
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

## 8. Reach-for table

| Data / goal | Reach for |
|---|---|
| Node/link tasks on a large graph | GraphSAGE / Cluster-GCN (sampled), GCN/GAT baseline |
| Graph-level (molecular property) | Graph transformer (GraphGPS) or GIN; scaffold split |
| 3D molecules/proteins, forces/energy | E(3)-equivariant net (NequIP/EGNN/e3nn) |
| Point clouds / sets | PointNet++ / Deep Sets / Set Transformer |
| Knowledge-graph link prediction | RotatE/ComplEx or R-GCN |
| Hierarchical data | Hyperbolic embeddings |
| Strong, honest baseline | MLP on node features (no graph) + plain GCN |

**Canonical references:** Bronstein et al. 2021 (*Geometric Deep Learning* proto-book); Kipf & Welling 2017
(GCN); Hamilton et al. 2017 (GraphSAGE); Veličković et al. 2018 (GAT); Xu et al. 2019 (GIN/expressiveness);
Gilmer et al. 2017 (MPNN); Satorras et al. 2021 (EGNN); Batzner et al. 2022 (NequIP); Hu et al. 2020 (OGB).
