# Semantic Entropy Runbooks

The MMLU semantic-entropy baseline runbook lives at `UQ/SE/runbook.md`.
The MMLU-Pro semantic-entropy baseline runbook lives at `UQ/SE_PRO/runbook.md`.

Quick MMLU smoke test:

```bash
modal run UQ/SE/modal_se.py --eval-mode split --num-eval-rows 32 --n-samples 10
```

Quick MMLU-Pro smoke test:

```bash
modal run UQ/SE_PRO/modal_se_pro.py --eval-mode split --num-eval-rows 32 --n-samples 10
```
