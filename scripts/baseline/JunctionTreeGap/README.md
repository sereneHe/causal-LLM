# JunctionTreeGap

This baseline turns the junction-tree discussion into a runnable causal discovery prototype.

## Core idea

- Treat each clique as a local probabilistic unit.
- Let separators carry shared-marginal messages between adjacent cliques.
- Measure the disagreement between upward and downward messages on each separator.
- Penalize separator disagreement together with sparsity and acyclicity.
- Orient the final graph by a score order so the output stays acyclic by construction.
- Compute a weighted adjacency matrix first, then find the minimal DAG threshold and convert it into a binary adjacency matrix.

## Modeling view

- Clique potentials can be represented by:
  - a traditional BN table
  - a neural network
  - an MLP or Transformer encoder
- In the current prototype, `nn` uses a lightweight MLP, `mlp` uses a deeper residual MLP, and `transformer` uses a token-level encoder with a CLS token over clique features, while `gaussian`/`bn` stay as closed-form score backends.
- A separator can also be parameterized explicitly, for example:
  - `q(S_ij) = softmax(f_theta(S_ij))`
- In this prototype, separator messages are estimated from the two adjacent cliques and stored as Gaussian summaries.

## Separator gap

For every clique edge `(i, j)` with separator `S_ij`, the model stores:

- `m_{i->j}(S_ij)` as the upward-style message
- `m_{j->i}(S_ij)` as the downward-style message

Their disagreement can be measured in three common ways:

```text
gap_ij = KL(normalize(m_{i->j}) || normalize(m_{j->i}))
gap_ij = || normalize(m_{i->j}) - normalize(m_{j->i}) ||_2^2
gap_ij = KL(p || q) + KL(q || p)
```

where `p = normalize(m_{i->j})` and `q = normalize(m_{j->i})`.

## Objective

```text
Loss =
  NLL(data | clique potentials, A)
  + λ1 * Σ_gap_ij
  + λ2 * sparsity(A)
  + λ3 * acyclicity(A)
```

## Current prototype

- builds a correlation-based skeleton
- extracts maximal cliques and a clique tree
- fits Gaussian clique potentials
- estimates separator messages from the two adjacent cliques
- supports `sym_kl`, `kl`, and `l2` as gap metrics
- orients edges by a gap-regularized node score
- thresholds the weighted matrix with the minimal DAG threshold before saving the final adjacency matrix
- outputs an adjacency matrix plus a structured artifact with clique and separator summaries

## Run

```bash
python main.py --data-npy /path/to/data.npy --output-dir outputs/JunctionTreeGap/asia
```
