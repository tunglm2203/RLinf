# value_head — language-conditioned progress head on GR00T N1.7

## Overview

**What this is.** A trainable **progress value head**: given a frame and a sub-goal instruction, it
predicts *how far along that sub-goal the frame is* — `local_progress ∈ [0,1]`. It is trained on
**ego-centric video** (egoverse) using features from a **frozen** GR00T N1.7 backbone
(base Cosmos-Reason2-2B / Qwen3-VL, no robot or LIBERO training). Because the backbone is frozen, the
pipeline is **two-stage**: extract per-frame features once and cache them, then train a small head on
the cache in seconds — so pooling, loss, and hyper-parameters can be swept without re-running the GPU.
The head is **language-conditioned** (its prediction depends on *which* sub-goal it's given, not just
the image), which is what makes it a reward model rather than a stopwatch. This is milestone **M2** of
`../research/VALUE_HEAD_RESEARCH_PLAN.md` — the ego-pretraining stage of the hierarchical-VLA value head.

```
  egoverse RGB frame              text slot  (arm-composed, narratives EXCLUDED)
  +--------------+                +----------------------------------------------+
  |  480x640x3   |                | "<global>. Done: <grounded history>.         |
  +------+-------+                |  Now: <sub-goal>"        (sub-goal LAST)      |
         |                        +----------------------------+-----------------+
         +-------------------+--------------------------------+
                             v
          +-----------------------------------------+
          |  Cosmos-Reason2-2B  (N1.7), FROZEN      |     stage 1  build_labels     (CPU)
          |  Qwen3-VL backbone -> hidden states     |     stage 2  extract_features (GPU)
          +--------------------+--------------------+
                               v   pooling  { mean_last | LAST_LAST | mean_mid }   (last_last wins)
                    pooled feature [2048-d]  -->  cached to disk
                               |                  (train the head, no re-run of the GPU)
  - - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - - - - - - - - - - -
                               v
          +-----------------------------------------+
          |  ProgressHead (robometer recipe)        |     stage 3  train.py         (CPU ok)
          |  Linear->LayerNorm->GELU->Dropout->Lin  |
          +--------------------+--------------------+
                               v   discrete 10-bin softmax -> expected value
                       local_progress in [0,1]
                               |
             language_sensitivity = rho(pred|correct sub-goal) - rho(pred|WRONG sub-goal)
```

Everything above the dashed line is the **frozen** backbone (run once, cached); everything below
**trains** — a small MLP head, seconds per fit.

## Repo layout (value head only)

```
vr_value_head/
  build_labels.py      stage 1: egoverse -> cache/pairs.parquet + cache/frames/*.npz   [needs zarr]
  extract_features.py  stage 2: frozen N1.7 -> cache/features/*.npz (all 3 poolings)    [needs GPU]
  train.py             stage 3: train ProgressHead on the cache; best-val ckpt; report train/val/test
  evaluate.py          held-out metrics: per-segment rho, clock baseline, language_sensitivity
  model.py             ProgressHead — robometer head; continuous (huber) or discrete-CE (n_bins)
  losses.py            discrete_ce · huber · huber_neg (off) · pairwise_rank · (paraphrase, off)
  dataset.py           join pairs + feature cache -> segment-block batches
  text_arms.py         compose the text slot: {subgoal | full = global+history+subgoal}
  corpus_reader.py     FUTURE corpus-release schema (manifest+narratives); NOT the active reader
  run.py               prints the 3-stage pipeline commands
  config/ego_egoverse.yaml   the LOCKED baseline config (documented)
  README.md            this file
  cache/               (generated) pairs.parquet · frames/*.npz · features/*.npz
  runs/<name>/         (generated) head.pt · metrics.json     e.g. runs/vh_baseline_500_ce10/

driver:  bashscripts/pretrain_value_head/run.sh     (orchestrates all 3 stages)
```

Cross-directory dependency: `build_labels.py` and `evaluate.py` import `egoverse_data.py` / `metrics.py`
from `../research/probes/` — keep that directory alongside if you relocate this package.

## Installation

No `pip install` step — the pipeline runs in an existing venv. **`/opt/venv/egoverse`** already carries
everything: the full GR00T N1.7 torch stack (**torch 2.7.1+cu128, transformers 4.57.3, sm_120**) **and**
`zarr` (to read egoverse). It is the only venv that has both, so all three stages use it.

Prerequisites (paths are the defaults; override in `config/` or via env vars):

| what | path | note |
|---|---|---|
| venv | `/opt/venv/egoverse` | torch 2.7.1+cu128 (Blackwell/sm_120), transformers 4.57.3, zarr |
| backbone | `/mnt/workspace/HUGGINGFACE_HOME/hub/Cosmos-Reason2-2B` | base pretrained VLM (no robot/LIBERO) |
| dataset | `/mnt/workspace2/all_dataset/egoverse` | 500 ego episodes |
| GPU | any CUDA (sm_120 verified) | needed only for stage 2 (feature extraction) |
| disk | ~50 GB | frame cache (`cache/frames/`) for 500 episodes |

Rebuilding the venv from scratch: it must reproduce the torch 2.7.1+cu128 (sm_120) + transformers 4.57.3
+ zarr combination — a fresh install that pins torch 2.6/cu124 will **not** drive the backbone on
Blackwell GPUs (see the repo's `gr00t-venv-blackwell` notes).

## Quick start

One command reproduces the locked baseline (defaults = the baseline config):

```bash
bash bashscripts/pretrain_value_head/run.sh
```

Stages, and how to skip the expensive ones (the feature cache is reusable):

```bash
# full pipeline: build_labels (CPU ~20m) -> extract_features (GPU ~60m) -> train (seconds)
bash bashscripts/pretrain_value_head/run.sh                     # STAGE=all (default)

# once the cache exists, re-train only — sweep in seconds, no GPU:
STAGE=train POOL=last_last N_BINS=10 bash bashscripts/pretrain_value_head/run.sh
STAGE=train N_BINS=1                 bash bashscripts/pretrain_value_head/run.sh   # continuous huber
```

Output → `runs/<RUN_NAME>/{head.pt, metrics.json}` and logs under
`logs/pretrain_value_head/<RUN_NAME>/`. Expected held-out result:

```
test  spearman_head 0.32   language_sensitivity 0.31   (clears M2 GO bar >= 0.3)
```

Common env overrides (all have baseline defaults): `EPISODES POOL N_BINS EPOCHS LR STRIDE ARM
TIME_WARP SEGS_PER_BATCH BATCH_SIZE DEVICE CACHE RUN_NAME STAGE`.

---

## LOCKED baseline (2026-08)

```
config   last_last pooling · robometer head (Linear->LN->GELU->Dropout->Linear)
         · AdamW + cosine + 10% warmup + wd 0.01 + grad-clip 1.0
         · DISCRETE 10-bin cross-entropy · negatives-excluded from loss
         · 500 episodes · 150 epochs · best-val checkpoint, reported once on test
reproduce   bash bashscripts/pretrain_value_head/run.sh        (defaults = this)
artifact    runs/vh_baseline_500_ce10/{head.pt, metrics.json}

held-out result           train    val     test
  spearman_head           0.81     0.41    0.32
  language_sensitivity    0.81     0.45    0.31      <- clears M2 GO bar (>= 0.3)
```

**What's settled** (each an experiment, not a guess):
- `last_last` (final token, carries the sub-goal) >> `mean_last` / `mean_mid` (which dilute it).
- **Discrete CE > continuous Huber** (test rho 0.32 vs 0.25) — robometer's shipped choice; it even beats
  the linear-Ridge ceiling (~0.30), so classification extracts more from frozen features than regression.
- **Wrong-task-negatives folded into the loss HURT** at our ~49% negative fraction (robometer dilutes
  them to ~25% with reverse/rewind, which a per-frame head can't make) -> excluded from the loss, kept
  only for the language metric.
- The old numbers (rho 0.53) were **inflated ~2-3x** by final-epoch selection + test leakage; fixed with
  best-val selection + report-once-on-test.

**Known limitations** (honest, for handover):
- **Overfitting**: train rho 0.81 vs test 0.32 — the CE head can memorize. A regularization sweep
  (dropout 0.1->0.3, weight-decay up) likely recovers more test rho; not yet done.
- **Frozen-feature ceiling**: rho ~0.32 is near what the *base* Cosmos-Reason2-2B VLM encodes about
  progress. The lever to go higher is **VLM fine-tuning** (deferred).
- `ARM=subgoal` for now; `ARM=full` (global + grounded history + sub-goal) is the intended handover
  config — it needs a re-extract (stage 2), since the text changes the features.

## Reading the metrics

`train.py` reports `spearman_head` (per-segment rho of predicted vs proxy-progress) and
`language_sensitivity` on all three splits. **Language sensitivity is the load-bearing one**:
`rho(pred | correct sub-goal) - rho(pred | WRONG sub-goal)`. High is good — it means the head uses the
sub-goal (a goal-conditioned reward model); ~0 means it ignores language (an image-only clock, the M2
kill criterion). `beats_clock` is negative by construction here (the local-progress target *is* the
elapsed-frame clock), so it is not the bar — language sensitivity is.

## The PAIRS-table contract

`build_labels.py` emits `cache/pairs.parquet` — one row per **(frame, text)** to encode:

```
pair_id  episode_id  frame  seg_idx(=k)  n_segs  local_progress
role {pos | neg_subgoal}  arm  text  split  split_group  global_task
```

A `neg_subgoal` row is the *same frame* under a *different* sub-goal — a distinct cached feature, which
is what makes the language metric possible offline. `extract_features.py` caches one vector per row;
`dataset.py` joins them back. (`neg` rows carry `local_progress = 0`; excluded from the loss, used for
the language metric. Paraphrase rows are omitted — egoverse has no rephrasings; wired for the corpus release.)

## Gotchas

- **Single venv**: use `/opt/venv/egoverse` for every stage. `gr00t_n1d7` can do stages 2–3 but lacks zarr.
- **Last-token pooling** must index the last *real* token by padding side, **not** `attention_mask.sum(1)-1`
  (that lands on a pad and silently breaks — see `extract_features.py`).
- **Backbone is the base VLM**, not the LIBERO/robot checkpoint — deliberate (ego-pretraining stage).
- Changing `ARM` or `STRIDE` requires a **re-extract** (stage 2); changing `POOL`, `N_BINS`, loss, or
  epochs is **stage-3 only** (reuses the cache).
