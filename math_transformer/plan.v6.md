# Math-Routed Sparse Transformer Plan v6

## Status

Project path:

```text
/workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer/
```

Current confirmed baseline:

```text
tests: 154 passed
attention_mode switch: full / dense_masked / neighbor_sparse
topology cache: in-process, device-aware (GPU path on CUDA, CPU numpy path otherwise)
GPU-native topology build: torch ops for all 7 relation matrices
torch.compile sparse kernel: neighbor_attention_compiled (CPU Inductor works; CUDA requires cc≥7.0)
Triton fused kernel: _nbr_sparse_attn_kernel (one program per batch×head×token)
topology_build_ms: isolated in benchmark, confirmed 50–286× more expensive than attention
runs/ persistence structure: benchmarks, topology_cache, eval_reports, model_checkpoints, failure_sets
```

Current architecture stage:

```text
GPU-resident topology + Triton kernel — kernel-level speedup proven, routing still quadratic
```

Target architecture stage:

```text
Fixed-K algebraic routing — O(nK) complexity where K is a compile-time constant, not proportional to n
```

---

## Variables

$$n = \text{nodes/tokens}$$

$$F = n^2 = \text{full attention edges}$$

$$A = \text{allowed routed edges}$$

$$\rho = \frac{A}{F}$$

$$R = 1 - \rho$$

$$\bar{k} = \frac{A}{n}$$

$$S_{\text{comp}} = \frac{t_{\text{dense}}}{t_{\text{sparse\_compiled}}}$$

$$D = \frac{t_{\text{topology}}}{t_{\text{sparse\_compiled}}}$$

---

## Equations

### Edge density

$$\rho_{\text{roots}} \approx 0.224$$

$$\rho_{\text{trees}} \approx 0.520$$

**Meaning:** `roots` keeps about **22%** of full attention; `trees` keeps about **52%**.

---

### Reduction

$$R_{\text{roots}} \approx 0.776$$

$$R_{\text{trees}} \approx 0.480$$

**Meaning:** `roots` removes about **77%** of attention edges; `trees` removes about **48%**.

---

### Complexity check

For `roots`:

$$\bar{k}_{256} = 58.8 \qquad \bar{k}_{512} = 115.9 \qquad \bar{k}_{1024} = 229.4$$

For `trees`:

$$\bar{k}_{256} = 133.8 \qquad \bar{k}_{512} = 266.9 \qquad \bar{k}_{1024} = 532.6$$

Since:

$$\bar{k} \propto n$$

Then:

$$A = n\bar{k} \approx n(cn) = cn^2$$

**Meaning:** this is still **quadratic attention with a smaller constant**, not true sparse/subquadratic attention yet.

---

## 1. Benchmark Reference (CPU, Python 3.14.3, v5 run)

```text
device=cpu  threads=4  python=3.14.3  max_neighbors=32

n     mode   allowed    full    avg_k  max_k  rel_red  topo_ms   d_attn  s_trnc  s_comp
256   roots   15046   65536     58.8     87   0.7704    76.8     0.307   1.283   1.075
512   roots   59334  262144    115.9    173   0.7737   275.9     5.406   3.156   1.368
1024  roots  234950 1048576    229.4    343   0.7759  1014.4    12.688   6.028   3.544
256   trees   34246   65536    133.8    195   0.4774    65.1     0.408   1.370   1.167
512   trees  136648  262144    266.9    389   0.4787   235.0     1.997   3.151   2.262
1024  trees  545422 1048576    532.6    778   0.4798   809.0    13.253   7.333   4.243
```

---

## Main Interpretation

### 1. The architecture works as a routing filter

The mask/routing system is successfully reducing attention.

| Mode | Density | Reduction | Meaning |
|---|---|---|---|
| `roots` | ~22% | ~77% removed | Aggressive sparse routing |
| `trees` | ~52% | ~48% removed | More complete, less sparse |

$$\text{Full Attention} \rightarrow \text{Math-Routed Attention Subgraph}$$

---

### 2. But it is not sparse enough yet

The key problem:

$$\bar{k} \text{ doubles when } n \text{ doubles}$$

That means every node is still attending to a fixed percentage of all other nodes.

Current form:

$$O(n^2)$$

Desired form:

$$O(nK), \quad K = \text{constant or slowly growing}$$

Better target:

$$K \in [32, 128]$$

For $n = 1024$, current values are:

| Mode | Current avg_k | Better target |
|---|---|---|
| `roots` | 229.4 | 64–128 |
| `trees` | 532.6 | 64–128 |

The next move is **top-K pruning after algebraic scoring**.

---

### 3. `roots` is the efficient path

$$A_{1024} = 234{,}950 \qquad F_{1024} = 1{,}048{,}576 \qquad R = 77.59\%$$

`roots` is the better candidate for an efficient transformer architecture.

But:

```text
symbolic_dependency: 0
```

`roots` is mostly not using dependency structure yet. It is routing mostly through operator similarity and compatibility rules.

$$\text{roots} = \text{efficient but shallow}$$

---

### 4. `trees` is the more semantic path

`trees` at $n = 1024$ has:

```text
symbolic_dependency: 103976
```

$$\frac{103976}{545422} \approx 19.1\%$$

`trees` is actually using structural dependency information. But:

$$\rho_{\text{trees}} \approx 0.52$$

That is too dense.

$$\text{trees} = \text{more mathematically meaningful but too expensive}$$

Best move:

$$\text{trees generates candidates} \rightarrow \text{top-K compression selects final edges}$$

---

### 5. `same_operator` is dominating too much

At $n = 1024$:

**Roots:**
```text
same_operator: 232222
allowed:       234950
```
$$\frac{232222}{234950} \approx 98.8\%$$

**Trees:**
```text
same_operator: 416316
allowed:       545422
```
$$\frac{416316}{545422} \approx 76.3\%$$

Routing is overwhelmingly controlled by a single blunt rule.

Current routing:

$$E \approx E_{\text{same\_operator}} \cup E_{\text{shape}} \cup E_{\text{composition}} \cup E_{\text{local}}$$

Better:

$$\text{score}(i,j) = w_1 f_{\text{symbolic}}(i,j) + w_2 f_{\text{composition}}(i,j) + w_3 f_{\text{shape}}(i,j) + w_4 f_{\text{embedding}}(i,j) + w_5 f_{\text{local}}(i,j) + w_6 f_{\text{operator}}(i,j)$$

Then:

$$E_i = \operatorname{TopK}_j\bigl(\text{score}(i,j),\; K\bigr)$$

**Do not union every rule directly; score edges, then keep top-K.**

---

### 6. The sparse compiled kernel is promising

At $n = 1024$:

**Roots:**
$$S_{\text{comp}} = \frac{12.688}{3.544} \approx 3.58$$

**Trees:**
$$S_{\text{comp}} = \frac{13.253}{4.243} \approx 3.12$$

Sparse compiled is **3×–4× faster** than dense attention at $n = 1024$.

$$\text{s\_comp wins at large } n$$

The compressed sparse representation is worth developing.

---

### 7. Topology construction is the current bottleneck

At $n = 1024$:

**Roots:**
$$D = \frac{1014.422}{3.544} \approx 286$$

**Trees:**
$$D = \frac{809.027}{4.243} \approx 191$$

Topology building is **190×–286× more expensive** than the compiled attention kernel.

The attention kernel is not the bottleneck. The bottleneck is:

$$\text{build routing graph / topology mask}$$

The architecture only works if topology is cached, incrementally updated, or compiled once and reused many times.

Current execution model:

$$\text{build topology every run} \rightarrow \text{too slow}$$

Better model:

$$\text{compile symbolic topology once} \rightarrow \text{reuse sparse attention many times}$$

---

## Verdict

$$\boxed{\text{Promising architecture, but routing is not sparse enough yet.}}$$

$$\boxed{S_{\text{comp}} \text{ gives real speedup at } n = 1024 \text{ (3×–4× on CPU)}}$$

$$\boxed{\bar{k} \propto n \quad \text{— the graph still scales quadratically}}$$

$$\boxed{t_{\text{topology}} \gg t_{\text{attention}} \quad \text{— topology construction dominates everything}}$$

---

## Next Architecture

### Candidate graph construction

$$G_{\text{candidate}} = G_{\text{symbolic}} \cup G_{\text{composition}} \cup G_{\text{shape}} \cup G_{\text{embedding}} \cup G_{\text{local}}$$

### Edge scoring

$$\text{score}(i,j) = \sum_r w_r f_r(i,j)$$

### Fixed-K selection

$$E_i = \operatorname{TopK}_j\bigl(\text{score}(i,j),\; K\bigr), \quad K \ll n$$

### Sparse attention

$$Y_i = \sum_{j \in E_i} \operatorname{softmax}\!\left(\frac{Q_i K_j^T}{\sqrt{D}}\right) V_j$$

**Algebra creates candidates; scoring ranks them; top-K makes it truly sparse.**

---

## Development Priority

### Sprint 1 — Hard top-K routing

**Files:** `src/topology.py`, `src/topology_cache.py`

Replace the current union-of-masks approach with a scored top-K selection.

```python
score(i, j) = w_symbolic * symbolic(i,j)
            + w_composition * composition(i,j)
            + w_shape * shape_compat(i,j)
            + w_embedding * embedding_sim(i,j)
            + w_local * local_window(i,j)
            + w_operator * same_operator(i,j)

E_i = argsort(score[i], descending=True)[:K]
```

Target: $\bar{k} \approx K$, not $\bar{k} \approx cn$.

Rerun benchmark after:

```bash
scripts/benchmark_attention.sh --sizes 256,512,1024,2048
```

Expected: `avg_k ≈ 32` at all n. `rel_red` improves beyond 77% at large n.

**Acceptance:** `avg_k` stable across n (not scaling with n).

---

### Sprint 2 — Frozen embeddings

**Files:** `src/embedder.py`, `src/model.py`, `src/topology_cache.py`

`embedder.encode_batch` currently runs every forward call even on cache hits.
Precompute `z` once at graph construction time; store alongside the topology cache.

```python
# Before: recomputed every forward
z = self.embedder.encode_batch(nodes)

# After: computed once, passed in
z = graph.frozen_embeddings  # set at construction
```

Expected: block-level timing drops from ~1060ms to ~20ms at n=1024 (removes Python loop from hot path).

**Acceptance:** `sparse_block_cached_ms ≈ full_block_ms + attention_overhead`.

---

### Sprint 3 — Amortized topology benchmark

**Files:** `src/eval.py`

Current benchmark rebuilds topology every call for `sparse_block_uncached`.
Add explicit `reuse_count` tracking to show the amortized cost.

```text
topology_build_once_ms
attention_runtime_ms
reuse_count
amortized_ms_per_forward = (topology_build_once_ms + reuse_count * attention_runtime_ms) / reuse_count
```

As `reuse_count → ∞`, `amortized_ms → attention_runtime_ms`.
This is the honest comparison: topology build is a fixed cost paid once per expression set.

---

### Sprint 4 — Reduce `same_operator` dominance

**Files:** `src/topology.py`

`same_operator` currently contributes ~99% of roots edges and ~76% of trees edges.
It is a weak semantic signal being used as the primary router.

Options:
- (a) Weight `same_operator` at 0.1 in the score; `symbolic_dependency` and `composition` at 1.0
- (b) Remove `same_operator` from the candidate graph entirely for roots mode
- (c) Replace with `same_op_class` (broader: elementwise, matmul, reduction, etc.)

Expected: `symbolic_dependency` and `composition` become the dominant relation types.
`roots` mode gains meaningful dependency structure instead of relying on operator coincidence.

---

### Sprint 5 — Scale to n=2048

Once Sprint 1 (fixed-K) is in place:

```bash
scripts/benchmark_attention.sh --sizes 512,1024,2048,4096
```

With $K = 32$ and $n = 2048$:

$$\text{FLOPs ratio} = \frac{K}{n} = \frac{32}{2048} = 0.016 \quad \text{(98.4% reduction)}$$

If $S_{\text{comp}}$ continues to grow with n, the architecture scales.

---

## Success Gates v6

### Gate 1 — Fixed-K routing works

```text
avg_k ≈ K at all n (not scaling with n)
relation_reduction > 0.85 for roots at n=1024 with K=32
```

### Gate 2 — Frozen embeddings unblock block-level

```text
sparse_block_cached_ms ≈ full_block_ms + nbr_sparse_attn_ms
topology overhead < 5% of total block time on cached path
```

### Gate 3 — Dependency relations dominate routing

```text
symbolic_dependency + composition > 50% of allowed edges (roots mode)
same_operator < 20% of allowed edges
```

### Gate 4 — Subquadratic confirmed

```text
avg_k stable (within 10%) as n doubles from 512 → 1024 → 2048
A grows as O(nK), not O(n²)
```

### Gate 5 — End-to-end speedup

```text
amortized_ms_per_forward (sparse, reuse_count=100) < full_block_ms
```

---

## Final Read

$$\boxed{\text{This proves the math-router can reduce attention and that compressed sparse execution can win.}}$$

$$\boxed{\text{It does not yet prove subquadratic transformer scaling.}}$$

$$\boxed{\text{The next breakthrough is fixed-K algebraic routing.}}$$
