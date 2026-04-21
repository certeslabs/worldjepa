# Phase 0.5 v3 Execution Notes (Week 0 kickoff)

Date: 2026-04-20 Branch: `phase-0.5`

This document tracks the immediate Week-0 execution order aligned with the v3
plan:

1. Run **three mandatory micro-experiments** before architecture edits.
2. Confirm patch-token output handling (`(B, N, D)`) and per-frame reshape
   assumptions.
3. Move to V-JEPA 2.1 backbone migration and `lejepa` integration after D0.0
   outputs are saved.

## Day 1 scripts

- `scripts/micro_exp_1_pooling.py`
  - Compares `cls`, `mean`, `max` token pooling.
  - Logs isotropy/effective-rank/variance diagnostics.

- `scripts/micro_exp_2_task_difficulty.py`
  - Computes adjacent-vs-random cosine statistics after patch-token reshape.
  - Logs a task-difficulty ratio.

- `scripts/micro_exp_3_batch_effect.py`
  - Measures SIGReg variability across batch sizes using official `lejepa` API.
  - Uses `EppsPulley(n_points=17)` (code-signature aligned).

Shared utilities: `scripts/micro_exp_common.py`.

## Output location

All scripts write JSON artifacts under:

- `results/phase_0_5/micro_exps/`

## Notes

- Scripts use V-JEPA 2.1 ViT-B by default: `vjepa2_1_vit_base_384`.
- The reshape assumption enforces:
  - `tubelet=2`
  - `patch_size=16`
  - `n_tokens_total = (T // 2) * (resolution // 16)^2`

If this assertion fails, we stop and inspect encoder config before proceeding to
training changes.
