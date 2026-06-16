# Assistant truth invariants for this project

This file records hard communication and implementation rules for future work in `math_transformer`.

## Core invariant

Do not distort expectations.

Name every artifact by what it actually is, not by what we hope it becomes.

## Forbidden mislabeling

- Do not call smoke tests real distillation.
- Do not call pseudo-targets real Qwen data.
- Do not call sampled labels Qwen logits.
- Do not call hard-label pseudo-logits raw teacher logits.
- Do not call static GGUF tensor metadata behavioral distillation.
- Do not call KL decrease on 3 rows distilled Qwen.
- Do not treat `decision=promoted` as proof that the research goal is solved.
- Do not imply Qwen was not computed if Qwen inference happened on CPU.
- Do not imply teacher-free work if teacher outputs are being generated.

## Required distinctions

Always separate these categories explicitly:

1. static checkpoint data extraction
2. sampled teacher output
3. pseudo-logit target construction
4. raw-logit teacher distillation
5. static weight-graph prior extraction
6. actual student training
7. gate/promotion smoke tests

## Canonical equations

Pseudo logits are not Qwen logits:

```text
pseudo_logits != qwen_logits
```

A sampled label is not a teacher distribution:

```text
sampled_label != p_Qwen(y | x)
```

Static checkpoint metadata is not behavioral distillation:

```text
GGUF_tensor_inventory != KL_distillation
```

A gate promotion is not scientific proof:

```text
pipeline_promoted != qwen_distilled
```

## Required wording

Use precise labels:

- smoke test
- static checkpoint prior
- sampled-label supervision
- pseudo-target training
- static tensor inventory
- bounded GGUF extraction
- frozen target artifact
- raw-logit distillation only when actual teacher logits are present

State plainly when something is not:

- not real Qwen logits
- not full Qwen distillation
- not teacher-free if teacher inference is being run
- not VRAM-free if any teacher process allocates GPU memory
- not CPU-free if teacher inference runs on CPU

## Implementation invariant

For requested actual Qwen data without teacher inference and without VRAM use:

```text
allowed: static checkpoint file parsing
forbidden: generation, logits, sampled labels, pseudo logits, CPU inference, CUDA teacher load
```

The correct name for that path is:

```text
static_qwen_checkpoint_data_extraction
```

not:

```text
real_qwen_logit_distillation
```
