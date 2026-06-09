# Conclusions — Math-Routed Sparse Transformer

**Hardware / runtime**: CUDA GPU, 4 CPU threads, Python 3.12.13, Arch Linux  
**Benchmark path**: CUDA/Triton neighbor-sparse attention through `scripts/benchmark_attention.sh`  
**Kernel**: `src/triton_attention.py::_nbr_sparse_attn_kernel` and flat-output variant  
**Current sparse block mode**: cached topology + top-`K` neighbor truncation

---

## Headline Finding

The CUDA/Triton sparse attention kernel is validated, and v7 now proves the topology itself can be compiled into a true fixed-`K` graph. Cached sparse blocks can beat dense blocks in the large-tree operating point, but uncached topology construction remains offline-only.

The earlier correction still matters: legacy benchmarks were not measuring attention over every allowed topology edge. They measured a routed top-`K` sparse attention path:

```text
symbolic topology -> priority neighbors -> top-K selected neighbors -> Triton attention
```

Therefore sparse attention runtime scales primarily with:

```text
T * K * D
```

not with:

```text
allowed_edges * D
```

For legacy `union` topology, the `allowed` column describes the full symbolic topology while the Triton kernel consumes the truncated neighbor table determined by `--max-neighbors`. For v7 `middle_preserving_topk`, the topology itself is already fixed-`K`, so `allowed ≈ T*K`.

---

## Current Implementation State

The hot path now includes these optimizations:

1. `TopologyCache` stores both `valid` and precompiled `valid_i8`.
2. The Triton wrapper no longer performs live `valid.to(torch.int8).contiguous()`.
3. The cached sparse block has a direct `forward_cached_fast_path` for `return_metadata=False`.
4. Sparse attention uses a fused QKV projection cache for inference.
5. A flat-output Triton sparse attention wrapper was added so sparse output can feed `out_proj` directly.
6. A CUDA correctness test was added for the flat-output Triton wrapper.
7. The default sparse neighbor budget is now `DEFAULT_MAX_NEIGHBORS = 16`, matching the best large-tree benchmark operating point. Passing `max_neighbors=None` still requests exact/unbounded `diag.max_k` behavior.
8. Experimental selector modes were added:
   - `topology_only`
   - `kmip_only`
   - `symbolic_kmip`
   - `symbolic_candidate_kmip`
9. Benchmark reporting now includes selector-level attention time, cached block time, and dense-output proxy metrics.
10. `stk_atn` now reports valid scored-topK attention timing instead of `0.000`.
11. A route-quality evaluation path was added:
    - `python -m src.eval --quality --quality-k 16,32,64,128`
    - optional checkpoint loading through `--checkpoint`
    - optional checkpoint saving during training through `python -m src.train --save-checkpoint ...`
12. An offline topology-search loop was added:
    - `python -m src.topology_search --checkpoint ... --k 16 --iterations 100`
    - optional unbounded mode through `--forever`
    - writes the best accepted relation weights to JSON
13. A deterministic synthetic-data generator was added:
    - `python -m src.synthetic_data --out-dir data/synthetic --train 10000 --val 1000 --test 1000 --seed 0`
    - emits mixed route and shape-validity JSONL records
    - writes a manifest with split counts
14. A one-command synthetic run script was added:
    - `scripts/train_synthetic.sh`
    - generates data, trains, then runs quality eval
    - accepts `TRAIN`, `VAL`, `TEST`, `SEED`, `MAX_STEPS`, `EVAL_INTERVAL`, `CHECKPOINT`, and related overrides
15. Quality reports now include per-expert accuracy diagnostics.
16. Training now accepts CLI overrides:
    - `--data`
    - `--max-steps`
    - `--eval-interval`
17. v7 adds an opt-in topology mode:
    - `--topology-mode middle_preserving_topk`
    - `--middle-bridge-width N`
18. v7 adds a middle-context bridge relation:
    - `middle_bridge`
    - default scored-topK weight `0.7`
    - priority `4`
19. Benchmark construction now honors `--topology-mode` and `--fixed-k` in the main reported topology path.
20. `TopologyCache` keys now include topology mode, `fixed_k`, and `middle_bridge_width`, preventing stale reuse across routing modes.
21. Priority truncation now pushes disconnected priority `0` behind real edges instead of sorting it first.

CUDA correctness status:

```text
.venv-cuda/bin/python -m pytest -q tests/test_triton_attention.py
9 passed
```

---

## v7 Fixed-K Middle-Preserving Topology Result

v7 applies the lost-in-the-middle routing lesson by adding an explicit middle bridge relation to scored top-`K` topology construction:

```text
A = alpha * I + beta * C + gamma * G
```

Where:

```text
I = identity/self route
C = existing symbolic/local/composition/shape route
G = middle_bridge route
```

The new mode is opt-in:

```bash
scripts/benchmark_attention.sh \
  --sizes 1024 \
  --node-mode trees \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1
```

Observed result:

```text
n=1024  mode=trees
allowed = 16,384
full    = 1,048,576
avg_k   = 16.0
max_k   = 16
rel_red = 0.9844

middle_bridge = 2,907
s_tri = 0.299 ms
d_attn = 1.959 ms
s_c = 2.296 ms
d_blk = 2.454 ms
```

Interpretation:

- v7 is now a true fixed-`K` topology, not a huge union mask with runtime truncation.
- `allowed = 1024 * 16 = 16,384`, which confirms the graph itself is bounded.
- `middle_bridge: 2,907` confirms middle-preserving edges survive top-`K` selection.
- Sparse Triton attention is about `1.959 / 0.299 = 6.55x` faster than dense attention in this run.
- Cached sparse block crossed dense-block parity in this run: `2.296 ms < 2.454 ms`.
- Uncached sparse remains invalid for live inference: `s_uc = 912.766 ms`.

### Before / After

| measurement | before: legacy tree topology | after: v7 middle-preserving topK |
|-------------|-----------------------------:|---------------------------------:|
| allowed edges | 545,422 | 16,384 |
| avg_k | 532.6 | 16.0 |
| max_k | 778 | 16 |
| relation reduction | 0.4798 | 0.9844 |
| same_operator selected/covered | 416,316 | 1,922 |
| middle_bridge | n/a | 2,907 |
| Triton sparse attention | 0.325 ms class | 0.299 ms |
| cached sparse block | 2.927 ms class | 2.296 ms |

The graph-size reduction is:

```text
545,422 / 16,384 = 33.3x smaller
```

Same-operator dominance also dropped sharply:

```text
416,316 / 1,922 = 216.6x lower
```

### What v7 Proves

```text
✓ The topology itself can be fixed-K, not merely runtime-truncated.
✓ Middle-context bridge edges can survive scored top-K selection.
✓ Cached sparse block parity can be crossed at n=1024 trees, K=16.
✓ The benchmark now correctly reports the selected topology mode.
✓ v7 can run inside the actual training loop with CUDA neighbor-sparse attention.
✓ v7 K=16 preserves full-attention route quality on the current synthetic validation split.
```

### v7 Training + Validation Result

Training command:

```bash
.venv-cuda/bin/python -m src.train \
  --config configs/tiny.yaml \
  --data data/synthetic/train.jsonl \
  --attention-mode neighbor_sparse \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1 \
  --device cuda \
  --max-steps 1000 \
  --eval-interval 100 \
  --save-checkpoint runs/checkpoints/v7_k16.pt \
  --save-loss-csv runs/train_curves/v7_k16.csv
```

Observed training behavior:

```text
checkpoint = runs/checkpoints/v7_k16.pt
loss at step 0   = 2.1444
loss at step 100 = 0.7210
loss at step 200 = 0.0066
loss at step 900 = 0.0008
post-warmup step time = roughly 4-7 ms in the printed checkpoints
```

Validation command:

```bash
.venv-cuda/bin/python -m src.eval \
  --quality \
  --examples data/synthetic/val.jsonl \
  --checkpoint runs/checkpoints/v7_k16.pt \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --quality-k 16 \
  --quality-device cuda
```

Validation result:

```text
examples = 6,068
full route_acc = 1.0000
v7 topology_only K=16 route_acc = 1.0000
dense_agree = 1.0000
```

Per-expert validation result:

```text
affine_expert      999/999  = 1.0000
constraint_expert 1024/1024 = 1.0000
generic_expert    1015/1015 = 1.0000
grad_expert        983/983  = 1.0000
matmul_expert     1024/1024 = 1.0000
reduction_expert  1023/1023 = 1.0000
```

Interpretation:

```text
Q_full = 1.0000
Q_v7,K=16 = 1.0000
Q_v7,K=16 / Q_full = 1.0
```

This passes the current quality gate for the synthetic route task. Sparse v7 K=16 made exactly the same route decisions as full attention on the same checkpoint weights.

### What v7 Does Not Prove

```text
✗ Better prediction quality than dense attention
✗ Better generalization on harder held-out templates
✗ Better next-token modeling
✗ Universal speedup across roots/trees/repeated runs
✗ Live topology construction viability
```

v7 is now both a scalability improvement and a validated sparse training path for the current synthetic route task. It is still not a claim that v7 is smarter than dense attention.

---

## Full Sweep Result: n = 64..1024, K = 32

Run:

```bash
scripts/benchmark_attention.sh --sizes 64,128,256,512,1024 --node-mode roots,trees
```

At `K=32`, the kernel-level result is strong:

|    n | mode  | d_attn | s_tri | dense / sparse kernel |
|------+-------+--------+-------+-----------------------|
| 1024 | roots |  1.968 | 0.312 |                 6.31x |
| 1024 | trees |  2.044 | 0.276 |                 7.41x |

The block-level result is mixed:

|    n | mode  | d_blk |   s_c | block result         |
|------+-------+-------+-------+----------------------|
|  128 | roots | 1.095 | 0.908 | sparse wins          |
|  128 | trees | 0.975 | 0.875 | sparse wins          |
| 1024 | trees | 2.498 | 2.432 | sparse wins slightly |
| 1024 | roots | 2.470 | 4.408 | sparse loses         |

The strongest system-level positive signal is:

```text
1024 trees: s_c = 2.432 ms < d_blk = 2.498 ms
```

This is only a small win, but it proves cached sparse block parity is reachable.

---

## K-Scaling Result at n = 1024

Runs:

```bash
scripts/benchmark_attention.sh --sizes 1024 --node-mode roots,trees --max-neighbors 16
scripts/benchmark_attention.sh --sizes 1024 --node-mode roots,trees --max-neighbors 32
scripts/benchmark_attention.sh --sizes 1024 --node-mode roots,trees --max-neighbors 64
scripts/benchmark_attention.sh --sizes 1024 --node-mode roots,trees --max-neighbors 128
```

### Roots

Topology:

```text
allowed = 234,950
full    = 1,048,576
avg_k   = 229.4
max_k   = 343
```

| max_neighbors | effective slots T*K | d_attn | s_tri | d_blk |   s_c | block result |
|---------------+---------------------+--------+-------+-------+-------+--------------|
|            16 | 16,384              |  2.051 | 0.292 | 2.417 | 4.274 | sparse loses |
|            32 | 32,768              |  1.973 | 0.326 | 2.691 | 4.306 | sparse loses |
|            64 | 65,536              |  1.979 | 0.375 | 2.461 | 4.796 | sparse loses |
|           128 | 131,072             |  1.973 | 0.651 | 2.423 | 4.229 | sparse loses |

Roots interpretation:

- Kernel time increases with `K`, as expected.
- Block time remains dominated by non-kernel overhead.
- The roots cached sparse block does not beat dense at n=1024 for any tested `K`.
- Lowering `K` helps, but not enough to overcome block overhead.

### Trees

Topology:

```text
allowed = 545,422
full    = 1,048,576
avg_k   = 532.6
max_k   = 778
```

| max_neighbors | effective slots T*K | d_attn | s_tri | d_blk |   s_c | block result |
|---------------+---------------------+--------+-------+-------+-------+--------------|
|            16 | 16,384              |  1.973 | 0.289 | 2.467 | 2.241 | sparse wins  |
|            32 | 32,768              |  1.961 | 0.294 | 2.438 | 2.789 | sparse loses |
|            64 | 65,536              |  2.003 | 0.432 | 2.403 | 3.296 | sparse loses |
|           128 | 131,072             |  1.954 | 0.475 | 2.466 | 2.418 | sparse wins  |

Trees interpretation:

- The best tree result is `K=16`, where cached sparse block wins:

```text
s_c = 2.241 ms < d_blk = 2.467 ms
```

- Increasing `K` increases kernel work, but block latency is noisy because the surrounding block overhead dominates.
- `K=128` now also crosses tree block parity in the latest rerun: `s_c = 2.418 ms < d_blk = 2.466 ms`.
- `K=16` remains the fastest observed large-tree operating point, while `K=128` may be preferable if quality requires a larger neighbor budget.

---

## Selector Comparison at n = 1024, K = 16

Run:

```bash
scripts/benchmark_attention.sh \
  --sizes 1024 \
  --node-mode roots,trees \
  --max-neighbors 16 \
  --selector-candidate-neighbors 64
```

This run compares four selector modes:

```text
topology_only             = cached symbolic priority table -> top-K neighbors
kmip_only                 = full q_i^T k_j top-K over all tokens
symbolic_kmip             = full alpha*q_i^T k_j + beta*R_symbolic over all tokens
symbolic_candidate_kmip   = score only 64 cached symbolic candidates, then select K=16
```

### Roots

| selector                | attention | cached block | dense L1 proxy | dense cosine proxy |
|-------------------------+-----------+--------------+----------------+--------------------|
| topology_only           | 0.293 ms  | 3.886 ms     |       0.287590 |           0.151736 |
| kmip_only               | 2.112 ms  | 5.939 ms     |       0.259210 |           0.339546 |
| symbolic_kmip           | 2.354 ms  | 6.280 ms     |       0.268954 |           0.232432 |
| symbolic_candidate_kmip | 3.590 ms  | 7.850 ms     |       0.273231 |           0.172479 |

Roots interpretation:

- `topology_only` remains fastest by a large margin.
- `kmip_only` has the best dense-output proxy similarity, but it is much slower than topology-only.
- `symbolic_kmip` is neither fastest nor best by dense proxy.
- `symbolic_candidate_kmip` is unexpectedly slower than full `symbolic_kmip` in this implementation, despite scoring only 64 symbolic candidates per row.

### Trees

| selector                | attention | cached block | dense L1 proxy | dense cosine proxy |
|-------------------------+-----------+--------------+----------------+--------------------|
| topology_only           | 0.273 ms  | 2.843 ms     |       0.284101 |           0.145990 |
| kmip_only               | 2.109 ms  | 4.284 ms     |       0.260552 |           0.348350 |
| symbolic_kmip           | 2.310 ms  | 4.295 ms     |       0.269202 |           0.206195 |
| symbolic_candidate_kmip | 3.553 ms  | 5.855 ms     |       0.272172 |           0.185624 |

Trees interpretation:

- `topology_only` remains the fastest selector, but the latest run did not repeat the prior dense-block win:

```text
topology_only: s_c = 2.843 ms > d_blk = 2.343 ms
```

- The strongest large-tree block-level result remains the previous run:

```text
topology_only: s_c = 2.090 ms < d_blk = 2.335 ms
```

- Block parity is therefore reachable but not stable yet.
- `kmip_only` again has the best dense-output proxy but loses the block-level speed objective.
- `symbolic_candidate_kmip` does not recover the expected performance benefit from candidate restriction. The likely cause is that its current PyTorch gather/scoring/topK path has more overhead than the full matrix score path at `T=1024, C=64`.
- Dense-output proxy metrics are not task quality. They are useful smoke signals only; the architecture decision still requires `Q(K)` on actual tasks.

Selector verdict:

```text
Do not promote k-MIP or symbolic candidate k-MIP yet.
Keep topology_only as the default.
Treat k-MIP selectors as experimental quality probes until selection is fused or otherwise made cheaper.
```

---

## Quality Evaluation Path

The benchmark now has a separate quality mode for the route-prediction task:

```bash
python -m src.eval \
  --quality \
  --quality-k 16,32,64,128 \
  --checkpoint runs/checkpoints/model.pt
```

This reports:

```text
mode=full           k=full  route_acc=...
mode=topology_only  k=16    route_acc=...  dense_agree=...
mode=topology_only  k=32    route_acc=...  dense_agree=...
```

Use `route_acc` as the first task-quality signal and `dense_agree` as a regression signal against the same model weights under full attention. If no checkpoint is provided, the command evaluates random initialization and is only a plumbing smoke test.

Training can now emit a checkpoint for this path:

```bash
python -m src.train \
  --config configs/tiny.yaml \
  --save-checkpoint runs/checkpoints/tiny.pt
```

This does not prove final model quality yet. It creates the measurement path needed to compare `Q(K)` and `Q(K) / Q_dense` for topology-only sparse attention.

### Tiny Route-Task Quality Result

Run:

```bash
python -m src.train \
  --config configs/tiny.yaml \
  --save-checkpoint runs/checkpoints/tiny.pt

python -m src.eval \
  --quality \
  --quality-k 16,32,64,128 \
  --checkpoint runs/checkpoints/tiny.pt
```

Result:

| mode | K | route accuracy | dense agreement |
|------|--:|---------------:|----------------:|
| full | full | 0.9333 | n/a |
| topology_only | 16 | 0.9333 | 1.0000 |
| topology_only | 32 | 0.9333 | 1.0000 |
| topology_only | 64 | 0.9333 | 1.0000 |
| topology_only | 128 | 0.9333 | 1.0000 |

Quality interpretation:

- On the tiny route-prediction task, topology-only sparse attention preserves the full-attention route decisions exactly for all tested K values.
- `K=16` satisfies the current quality threshold on this task:

```text
Q(K=16) / Q_dense = 0.9333 / 0.9333 = 1.0
```

- This is a small-data smoke result, not a final quality claim. The route task has only 15 examples and the model is evaluated on the same examples used for the short training run.
- The result is still important because it removes the first quality blocker: reducing runtime neighbors to K=16 did not degrade this trained route-task checkpoint.

---

## Offline Topology Search

Topology learning is implemented as an offline relation-weight search, not as live per-forward topology mutation.

Run:

```bash
python -m src.topology_search \
  --checkpoint runs/checkpoints/tiny.pt \
  --k 16 \
  --iterations 100 \
  --output runs/topology_search/best_weights.json
```

For an open-ended run:

```bash
python -m src.topology_search \
  --checkpoint runs/checkpoints/tiny.pt \
  --k 16 \
  --forever \
  --output runs/topology_search/best_weights.json
```

The loop mutates scored-topK relation weights:

```text
symbolic_dependency
composition
shape_compat
embedding
local_window
same_operator
```

`identity` is kept fixed so every node can always attend to itself.

The search objective is:

```text
objective = quality_weight * route_accuracy
          + (1 - quality_weight) * dense_agreement
```

Only candidates that match or improve the current objective are accepted and written to disk. The resulting JSON can be evaluated with:

```bash
python -m src.eval \
  --quality \
  --topology-mode scored_topk \
  --fixed-k 16 \
  --quality-k 16 \
  --relation-weights-json runs/topology_search/best_weights.json \
  --checkpoint runs/checkpoints/tiny.pt
```

This creates an auto-improving topology compiler loop:

```text
relation weights -> scored symbolic topology -> cached K-neighbor table -> sparse attention
```

The important constraint remains: inference should use the cached learned topology. The search loop can run for a long time offline, but only validated weight updates should be promoted.

---

## Synthetic Data and Per-Expert Quality

The project now has a deterministic synthetic-data path for expanding beyond the 15-example hand-written route task.

Primary command:

```bash
scripts/train_synthetic.sh
```

Equivalent manual stages:

```bash
python -m src.synthetic_data \
  --out-dir data/synthetic \
  --train 10000 \
  --val 1000 \
  --test 1000 \
  --seed 0

python -m src.train \
  --config configs/tiny.yaml \
  --data data/synthetic/train.jsonl \
  --save-checkpoint runs/checkpoints/synthetic_tiny.pt

python -m src.eval \
  --quality \
  --quality-k 16,32,64,128 \
  --examples data/synthetic/val.jsonl \
  --checkpoint runs/checkpoints/synthetic_tiny.pt
```

The synthetic generator emits two record types in the same JSONL stream:

```text
route records          -> consumed by route training and quality eval
shape_validity records -> consumed by shape-validity loading/tests
```

Route records carry an `expert` label. Shape-validity records may be valid or invalid and use the `valid` flag. The route loader skips non-route records, so invalid shape examples do not poison route training.

### 10k/1k/1k Synthetic Result

Run:

```bash
scripts/train_synthetic.sh
```

Data:

```text
train records = 10,000
train route records = 6,050
train shape records = 3,950
train invalid shape records = 1,597
val route records = 599
```

Quality:

```text
full route_acc = 0.8331
topology_only K=16 route_acc = 0.8331
topology_only K=16 dense_agree = 1.0000
```

Per-expert result:

```text
affine_expert     102/102
constraint_expert  99/99
generic_expert     97/97
grad_expert         0/100
matmul_expert     108/108
reduction_expert   93/93
```

Interpretation:

- The model solved five of six generated route classes after a short 100-step run.
- `grad_expert` was the only failure class in this seed/configuration.
- `topology_only` exactly preserved the full-attention decisions at `K=16..128`.
- This is a useful diagnostic result, not a final architecture claim.

### 50k/5k/5k Synthetic Result

Run:

```bash
TRAIN=50000 VAL=5000 TEST=5000 SEED=1 \
CHECKPOINT=runs/checkpoints/synthetic_big.pt \
scripts/train_synthetic.sh
```

Data:

```text
train records = 50,000
train route records = 30,065
val route records = 2,993
```

Quality:

```text
full route_acc = 1.0000
topology_only K=16 route_acc = 1.0000
topology_only K=16 dense_agree = 1.0000
```

Per-expert result:

```text
affine_expert     503/503
constraint_expert 499/499
generic_expert    521/521
grad_expert       480/480
matmul_expert     482/482
reduction_expert  508/508
```

Interpretation:

- The generated route grammar is learnable.
- The current synthetic validation distribution can be saturated.
- `topology_only` at `K=16` is behavior-preserving on this distribution.
- The task is now too easy to prove broad generalization.

### 200k/10k/10k Run Correction

The attempted overnight command generated a large dataset:

```text
train records = 200,000
train route records = 120,105
val route records = 6,068
```

But training still stopped after 100 steps because `configs/synthetic_overnight.yaml` had:

```yaml
training:
  max_steps: 100
```

The result:

```text
full route_acc = 0.8312
constraint_expert = 0/1024
all other experts = 1.0000
```

does not mean the large synthetic dataset failed. It means the run was still a short 100-step training run and did not settle the constraint class for that seed.

The script now exposes training length directly:

```bash
TRAIN=200000 VAL=10000 TEST=10000 SEED=2 \
MAX_STEPS=20000 EVAL_INTERVAL=1000 \
CHECKPOINT=runs/checkpoints/synthetic_overnight.pt \
scripts/train_synthetic.sh
```

Use this form for actual overnight runs.

### Synthetic Data Verdict

The synthetic-data path is now useful for plumbing, class diagnostics, and topology quality regression. It is not yet sufficient as the final task benchmark.

The next quality upgrade should add harder held-out template families:

1. deeper nested expressions,
2. held-out gradient forms,
3. constraint variants not seen in training,
4. mixed affine/elementwise forms,
5. topology stress examples that vary dependency, composition, local-window, same-operator, and shape-compatibility relations.

The milestone should move from:

```text
high validation accuracy on generated templates
```

to:

```text
high validation accuracy on held-out template families
```

---

## Correct Scaling Interpretation

The table should not be read as:

```text
runtime scales with allowed edges
```

It should be read as:

```text
runtime scales with selected top-K neighbor slots
```

For `n=1024`:

| mode  | allowed edges |  K | kernel slots T*K |
|-------+---------------+----+------------------|
| roots | 234,950       | 32 | 32,768           |
| trees | 545,422       | 32 | 32,768           |

Even though trees have about 2.32x more allowed edges than roots, the Triton kernel sees the same number of slots when `K` is fixed. That is why `s_tri` is similar for roots and trees at the same `K`.

The topology edge count matters for:

1. topology construction cost,
2. neighbor priority selection,
3. model quality,
4. whether a small `K` preserves enough symbolic context.

It does not directly determine Triton sparse attention runtime once top-`K` truncation is applied.

---

## Topology Build Remains Offline-Only

At `n=1024`, topology construction remains far too expensive for a live forward path:

| mode  | topo_ms range in latest runs           |
|-------+----------------------------------------|
| roots | about 2.00s to 2.30s in latest runs    |
| trees | about 1.08s to 1.33s in latest runs    |

The architecture must remain two-stage:

```text
Stage 1: compile/cache topology once
Stage 2: run repeated model forwards using cached CUDA neighbors + valid_i8
```

Any benchmark that includes topology construction in every forward is measuring topology generation, not model inference.

---

## What Is Proven

```text
✓ Triton sparse attention is correct on CUDA: 9 tests passed
✓ Sparse attention kernel is much faster than dense attention at n=1024
✓ Runtime scales with T*K, not total allowed edges
✓ Cached sparse block can beat dense block in selected large-tree cases
✓ K is now an explicit accuracy/speed control knob
✓ Topology construction must be cached/offline
✓ scored_topK attention timing is now reported
✓ topology_only at K=16 can beat dense block at n=1024 trees
✓ Route-quality evaluation can compare full attention against topology_only at multiple K values
✓ On the tiny route task, topology_only preserves full-attention route accuracy at K=16..128
✓ Offline relation-weight search can improve topology policy without changing the live sparse kernel path
✓ Synthetic route/shape data generation works and is deterministic
✓ Per-expert quality diagnostics identify class-specific failures
✓ topology_only at K=16 preserves full-attention decisions on current synthetic route validation
✓ v7 middle_preserving_topk compiles the tree topology directly to fixed K=16
✓ v7 inserts selected middle_bridge edges into the sparse topology
✓ v7 cached sparse block beats dense block in the observed n=1024 trees, K=16 run
✓ v7 K=16 trained checkpoint reaches 1.0000 validation route accuracy on current synthetic split
✓ v7 K=16 has dense_agree=1.0000 against full attention on the same checkpoint
```

---

## What Is Not Yet Proven

```text
✗ Universal cached sparse block speedup
✗ Roots-mode block-level speedup at n=1024
✗ End-to-end training or inference speedup including all model infrastructure
✗ Quality retention as K is reduced to 16 or 32 on larger/non-tiny tasks
✗ k-MIP or symbolic k-MIP selector speedup
✗ Task-quality improvement from k-MIP selectors
✗ Stable tree block-level speedup across repeated benchmark runs
✗ Generalization to held-out synthetic template families
✗ Training objective that uses invalid shape-validity examples directly
✗ Actual overnight/long-run synthetic quality result after increasing `MAX_STEPS`
✗ Prediction-quality improvement from v7 middle_bridge routing over dense attention
✗ Stable v7 block-level speedup across repeated seeds/runs
✗ v7 generalization on harder held-out synthetic template families
```

---

## Engineering Verdict

The right claim is now:

> Math routing can now compile a symbolic topology directly into a bounded fixed-K neighbor table. v7 adds middle-preserving bridge edges while keeping K fixed, the resulting cached Triton sparse block can cross dense-block parity at n=1024 trees, K=16, and a v7 K=16 trained checkpoint preserves full-attention route quality on the current synthetic validation split.

The wrong claim would be:

> Sparse attention runtime scales with all allowed symbolic edges.

It does not. With top-K truncation, runtime scales with `T*K`.

---

## Recommended Next Step

The k-MIP selector comparison has now been run, and v7 has added a better topology-only path. The result is clear: topology-only remains the only selector that satisfies the speed objective, while k-MIP variants are currently too expensive. The latest v7 K=16 checkpoint preserves full-attention route quality on the current synthetic validation split; the next question is harder generalization, not basic correctness.

Current routing is topology-prior top-`K`:

```text
N_i = TopK_j R_symbolic(i, j)
```

k-MIP-style routing is learned inner-product top-`K`:

```text
N_i = TopK_j q_i^T k_j
```

The proposed next architecture is symbolic k-MIP:

```text
score(i, j) = alpha * q_i^T k_j + beta * R_symbolic(i, j)
N_i = TopK_j score(i, j)
```

This keeps the math topology as a structural prior while allowing learned Q/K geometry to recover useful neighbors that static symbolic rules miss.

Current selector decision:

```text
Keep topology_only as the source default.
Do not promote symbolic_kmip.
Do not promote symbolic_candidate_kmip.
```

Updated priority order:

1. Add harder held-out synthetic template families and rerun `Q(K)` for `middle_preserving_topk` against `scored_topk` and full attention.
2. Keep the current 6,068-example synthetic validation result as the v7 regression baseline: `Q_v7,K16 = Q_full = 1.0000`.
3. Add harder held-out synthetic template families and rerun per-expert `Q(K)`.
4. Run a true long synthetic training run with `MAX_STEPS` set explicitly.
5. Run offline topology search against a held-out validation split, not the tiny train set.
6. Use `trees` as the primary block-parity target and `roots` as the overhead stress test.
7. Profile why the cached topology-only block still varies between winning and losing around dense parity.
8. If k-MIP is revisited, make selection cheaper before benchmarking again:
   - fused candidate scoring/topK kernel,
   - approximate topK/indexed retrieval,
   - or precomputed learned candidate tables.
9. Continue to measure both speed and quality:
   - `S_a`: sparse Triton attention latency
   - `S_c`: cached sparse block latency
   - `D_b`: dense block latency
   - `Q(K)`: task quality
   - `Q(K) / Q_dense`
10. Keep `K=16` as the current source default until quality data proves a larger `K` is necessary.
11. Do not spend more time optimizing topology build for live inference; topology build belongs in the cache/compiler layer.

Target acceptance condition:

```text
For n=1024 trees:
  find selector mode and K such that:
    S_c < D_b
    Q(K) >= 0.95 * Q_dense

For v7 middle_preserving_topk:
  current synthetic validation gate is passed:
    Q_v7(K=16) = 1.0000
    Q_dense = 1.0000
    dense_agree = 1.0000
  next gate must be harder held-out templates:
    Q_v7(K=16) >= 0.95 * Q_dense
    middle_bridge edges remain selected under K=16

For k-MIP variants:
  do not promote unless they beat or match topology_only quality
  and preserve cached sparse block parity on trees

For roots:
  use roots as an overhead stress test, not the primary architecture target
  identify why s_c remains above d_blk even when K is small
```

Decision rule:

```text
If topology_only at K=16 preserves quality:
  keep the current source default and avoid k-MIP complexity.

If topology_only loses quality but a k-MIP selector recovers it:
  first make selector runtime competitive with topology_only.
  only then consider promotion.

If kmip_only matches symbolic_kmip:
  the symbolic topology is not adding enough value and should be simplified.
```

---

## v8/v9 Prepared-Static Fast Path and Block-Token Triton Result

The latest implementation separates three different concerns that were previously conflated in benchmark interpretation:

```text
symbolic topology construction  -> offline / cacheable
prepared static topology lookup -> cheap hot-path metadata access
Triton sparse attention kernel  -> real runtime bottleneck
```

### Prepared Static Block Benchmark

The benchmark now exposes the prepared/static sparse block directly via:

```bash
scripts/benchmark_attention.sh \
  --sizes 1024 \
  --node-mode trees \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1 \
  --profile-prepared-block
```

Observed prepared-static result:

```text
n=1024 trees, K=16
allowed = 16,384
full    = 1,048,576
rel_red = 0.9844

d_blk  = 2.398 ms
s_c    = 2.017 ms
p_blk  = 0.941 ms
p_attn = 0.322 ms
p_non  = 0.619 ms
```

Derived speedups:

```text
prepared vs dense block  = 2.398 / 0.941 = 2.55x
prepared vs cached block = 2.017 / 0.941 = 2.14x
cached vs dense block    = 2.398 / 2.017 = 1.19x
```

Interpretation:

- The prepared/static sparse block crossed dense block parity decisively.
- The old cached sparse block timing hid the real prepared-fast-path performance.
- In the prepared profile, attention is no longer the majority of block time:

```text
p_attn / p_blk = 0.322 / 0.941 ≈ 34%
p_non  / p_blk = 0.619 / 0.941 ≈ 66%
```

This means the next bottleneck shifted from topology to standard transformer overhead:

```text
LayerNorm + QKV + out_proj + FFN + residual/module overhead
```

### Inference Fast-Path Cleanup

The block path now avoids several no-op dispatches in eval mode:

```text
eval dropout no-op bypass
eval residual dropout no-op bypass
eval FFN functional path
```

Direct profiler result after this cleanup:

```text
total_block_ms median ≈ 0.913 ms
attention_kernel_ms   ≈ 0.326 ms
```

The cleanup is small but real. Pure Python/module-dispatch cleanup is now mostly exhausted.

### Explicit Static Runtime API

The model now exposes an explicit static inference API:

```python
model.prepare_static_topology(nodes, env, device=device)
x = model.embed_nodes(nodes)
out = model.forward_static_fast_path(x)
```

Also at block level:

```python
block.prepare_static_topology(nodes, env, device=device)
out = block.forward_static_fast_path(x)
```

Correctness check:

```text
static fast path output == cached fast path output
```

CUDA microbenchmark:

```text
cached_fast_path median_ms = 0.627
static_fast_path median_ms = 0.646
speedup_static_vs_cached  = 0.971x
```

Interpretation:

- The explicit API is correct and useful for static deployment boundaries.
- It is not itself a latency win, because cache/topology lookup overhead was already negligible.

### Fusion Attempts

Two simple fusion attempts were implemented, validated, and left opt-in because they were slower on the tested CUDA path.

#### Fused LayerNorm + QKV

Added:

```python
triton_layernorm_qkv(...)
NeighborSparseMathAttention.enable_fused_norm_qkv = False
```

Correctness:

```text
max_diff = 4.768e-07
```

Latency:

```text
default_static_fast_path median_ms = 0.647
optin_fused_norm_qkv median_ms     = 0.838
```

Decision:

```text
correct, but slower -> opt-in only
```

#### Fused Sparse Attention + Output Projection

Added:

```python
triton_neighbor_attention_outproj(...)
NeighborSparseMathAttention.enable_fused_attn_outproj = False
```

Correctness:

```text
max_diff = 4.768e-07
```

Latency:

```text
default_static median_ms     = 0.861
fused_attn_outproj median_ms = 0.882
```

Decision:

```text
correct, but slightly slower -> opt-in only
```

### Block-Token Triton Sparse Attention

The successful kernel-level upgrade is block-token sparse attention:

```python
triton_neighbor_attention_flat_block_t(...)
NeighborSparseMathAttention.enable_block_token_attention = True
NeighborSparseMathAttention.block_token_attention_t = 2
```

This changes the CUDA sparse attention launch pattern from:

```text
one Triton program per (batch, head, token)
```

to:

```text
one Triton program per (batch, head, token-block)
```

with a default token block size:

```text
block_t = 2
```

Correctness:

```text
block_t=2 max_diff=0
block_t=4 max_diff=0
block_t=8 max_diff=0
```

Benchmark comparison:

```text
default_block_token_t2 median_ms = 0.605
legacy_one_token median_ms       = 0.617
speedup                         = 1.021x
```

Decision:

```text
block-token sparse attention is now the default CUDA sparse attention path
legacy one-token kernel remains available via enable_block_token_attention=False
```

### Current Bottom Line

The project has now crossed the important viability threshold:

```text
prepared static sparse block < dense block
0.941 ms < 2.398 ms
```

The current best-known hot-path shape is:

```text
fixed-K symbolic topology, K=16
prepared static topology buffers
Triton sparse attention with block_t=2
standard transformer FFN/LayerNorm/out_proj around it
```

The next major target is no longer symbolic routing. It is kernel specialization for the remaining transformer block overhead:

```text
attention_kernel_ms <= 0.22 ms
total_block_ms <= 0.75 ms
```

Most likely next upgrades:

```text
specialize sparse attention for H=4,D=16,K=16,T=1024
increase work per Triton program without register spilling
fuse residual/writeback logic where profitable
replace generic FFN path with inference-specialized kernels
```
