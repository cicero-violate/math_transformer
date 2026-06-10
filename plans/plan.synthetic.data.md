# Synthetic Data Plan

## Goal

Build a deterministic synthetic-data pipeline that can expand the tiny hand-written
dataset without losing the fixed evaluation signal. Synthetic data should improve
coverage for routing, shape validity, and topology search while keeping validation
and test splits stable.

## Principles

- Keep fixed `val` and `test` splits. Do not regenerate them during a search run.
- Generate new `train` data freely, but measure progress only on fixed held-out data.
- Include both route examples and shape-validity examples in JSONL. Route training
  consumes records with `expert`; shape checks consume records with `shape`.
- Make every generation run reproducible from a seed and record its manifest.
- Start with template-based examples that the current parser, normalizer, router, and
  shape engine can verify. Add harder expression families after the baseline is stable.

## Initial Schema

Route record:

```json
{
  "id": "train-route-000001",
  "task": "route",
  "expr": "affine(A, x, b)",
  "normalized": "add(matmul(A, x), b)",
  "expert": "affine_expert",
  "shape": {"A": [32, 64], "x": [64], "b": [32], "out": [32]},
  "valid": true
}
```

Shape-validity record:

```json
{
  "id": "train-shape-000001",
  "task": "shape_validity",
  "expr": "matmul(A, x)",
  "normalized": "matmul(A, x)",
  "shape": {"A": [32, 64], "x": [16]},
  "valid": false
}
```

## Curriculum

1. Simple valid routes: affine, matmul, elementwise, reduction, grad, constraint.
2. Simple invalid shapes: mismatched add, matmul, affine bias, and dot/vector cases.
3. Mixed nested expressions: affine-plus-residual, reductions over products, gradients
   of small expressions.
4. Failure-mined examples: save misrouted or shape-failed expressions and generate
   nearby variants.
5. Topology stress sets: expressions designed to vary dependency, composition,
   local-window, same-operator, and shape-compatibility relations.

## Commands

Run the full generate/train/evaluate path:

```bash
scripts/train_synthetic.sh
```

The script accepts environment overrides:

```bash
TRAIN=50000 VAL=5000 TEST=5000 SEED=1 MAX_STEPS=20000 EVAL_INTERVAL=1000 CHECKPOINT=runs/checkpoints/synthetic_big.pt scripts/train_synthetic.sh
```

Generate a baseline split:

```bash
python -m src.synthetic_data --out-dir data/synthetic --train 10000 --val 1000 --test 1000 --seed 0
```

Train on generated route records:

```bash
python -m src.train --config configs/tiny.yaml --data data/synthetic/train.jsonl --save-checkpoint runs/checkpoints/synthetic_tiny.pt
```

Evaluate on the fixed held-out split:

```bash
python -m src.eval --quality --quality-k 16,32,64 --examples data/synthetic/val.jsonl --checkpoint runs/checkpoints/synthetic_tiny.pt
```

## Acceptance Criteria

- The generator is deterministic for a fixed seed.
- Generated valid records pass `infer_shape`.
- Generated invalid shape records raise `ShapeError`.
- Route records load through `load_route_examples`.
- Shape records load through `load_shape_examples` with the correct `valid` flag.
- The manifest records counts, seed, and split filenames.

## Next Upgrade

After this baseline, add a continual generator loop that:

- trains on fresh synthetic batches,
- evaluates only on fixed held-out data,
- mines failures into a replay buffer,
- updates topology weights or examples only when held-out objective improves.
