# Model and training design

## 1. Task

**Input:** a peak list `[[m/z, intensity], ...]`, precursor m/z, normalized collision energy (NCE) and
precursor type (`[M+H]+` or `[M-H]-`).

**Output:** for each non-precursor peak, the top-k fragment structures from a fixed training library,
with mass error and base/joint scores. Peaks with no mass-compatible candidate are reported as such.

The query's DAG, teacher labels, correct precursor and atom mapping are never inference inputs.

## 2. Supervision

Labels come from AIMe fragmentation DAGs.

- For predicted spectra, each peak has a soft distribution over the DAG nodes that contribute to it.
- For experimental spectra, each peak has a set of mass-compatible nodes from the DAG of the known
  precursor. Any of them may be correct.
- A library label is defined by the non-stereo exported graph, the reported ion formula and the ion
  mode. Graph-derived and reported formulas are kept separately and are not silently reconciled.
- Peaks without a label still provide context; they are not treated as negatives.

## 3. Spectrum encoder

Each peak has five base features (scaled m/z, m/z relative to the precursor, precursor mass difference,
square-root and log intensity) and sine/cosine Fourier features of its mass and precursor difference.
A conditioning MLP adds NCE, precursor m/z and ion mode.

Four pre-norm Transformer blocks (hidden size 256, 8 heads) follow. Attention carries an explicit bias
from the signed pairwise mass difference:

```text
A_ij = q_i · k_j / sqrt(d) + b(m_i - m_j)
```

All valid peaks can attend to each other. `peak_only=True` and `use_pair_bias=False` provide matched
ablations. A mass difference is treated as evidence, not as a hard-coded neutral loss.

## 4. Fragment encoder and retrieval

Atoms and bonds are encoded by a three-layer message-passing network with a multi-layer mean/sum
readout. Element counts of the reported ion formula, its mass and its mode are encoded separately and
fused with the graph embedding. Spectrum and fragment vectors are L2-normalized:

```text
u_i(f) = (q_i · z_f) / temperature
```

Candidates must match the ion mode and lie within `max(0.007 Da, 5 ppm)` of the peak before ranking.

## 5. DAG relation learning

Four relations are supervised between fragments in the training DAGs: direct edge, multi-step
reachability, shared parent and shared precursor origin. Each has a directed low-rank bilinear score
(rank 48) plus a condition MLP over masses, mass differences, NCE and mode. Unobserved relations and
substituted endpoints are weak negatives with reduced weight, because a missing edge is not evidence
that a relation is impossible. Test DAGs are not written to the relation cache.

## 6. Joint reranking

For each peak, the base retriever supplies a top-32 shortlist. Up to six supporting peaks send
compatibility messages, weighted by the entropy of their own candidates. With mean support `m_i(f)`:

```text
l_i(t+1) = (1 - rho) * l_i(t) + rho * [ u_i + lambda * (m_i(t) - mean(m_i(t))) ]
```

The default is two iterations with `rho = 0.5`; `lambda` is selected on validation data only.
Messages average both relation directions, so the joint score is symmetric.

## 7. Training

| Stage | Data | Objective |
| --- | --- | --- |
| Pretraining | Predicted spectra | Soft-label cross-entropy + relation loss |
| Fine-tuning | Experimental spectra | Joint set loss + base set loss + relation loss |

The set loss is `logsumexp(all) - logsumexp(positives)`. Positives injected to compute the loss are not
automatically allowed to send messages; message candidates are sampled independently of the labels.
Training uses AdamW, gradient clipping, warmup with cosine decay, and distributed data parallelism.
Checkpoints store optimizer, scheduler, RNG, data signatures and protocol identity.

## 8. Evaluation protocol

`query_scaffold_intersection_v1` keeps a record only if its scaffold split agrees with the split of its
query. Queries and candidate scaffolds are then disjoint across train, validation and test; rejected
rows are quarantined with checksums instead of being relabelled.

Reports should separate all annotated peaks from peaks whose reference is in the library, compare
against a frequency baseline under the same mass gate, and state library coverage.

## 9. Limits

- Retrieval is restricted to the training fragment library; it cannot generate unseen structures.
- Labels are model-derived, so agreement measures consistency with the teacher, not chemical truth.
- Joint scores do not recover a directed fragmentation pathway, and fragments are not constrained to
  come from one complete precursor.
- Stereochemistry is not distinguished. Spectra at different collision energies are encoded separately.
