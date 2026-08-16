<div><h2> VR - Value Head</h2></div>
<br>

## Overview

A **value/progress head pretrained on ego-centric data**. Given a frame and a sub-goal instruction,
it predicts *how far along that sub-goal the frame is* — `local_progress ∈ [0,1]`. It is trained on
ego-centric video (e.g., **egoverse** dataset) using features from a frozen GR00T N1.7 backbone
(base=Cosmos-Reason2-2B / Qwen3-VL, no robot or LIBERO training). Because the backbone is frozen, the
pipeline is two-stage: (1) extract per-frame features once and cache them, (2) then train a small head on
the cache, so pooling / loss / hyper-parameters can be swept without re-running the GPU. The
head is language-conditioned: its prediction depends on *which* sub-goal it is given, not just the
image — that is what makes it a reward model rather than a stopwatch. This is the ego-pretraining stage
(**M2**) of the hierarchical-VLA value head; see `research/VALUE_HEAD_RESEARCH_PLAN.md`.

```
  egoverse RGB frame              text slot  (arm-composed, narratives EXCLUDED)
  +--------------+                +----------------------------------------------+
  |  480x640x3   |                | "<global>. Done: <grounded history>.         |
  +------+-------+                |  Now: <sub-goal>"        (sub-goal LAST)     |
         |                        +----------------------------+-----------------+
         +-------------------+--------------------------------+
                             v
          +-----------------------------------------+
          |  Cosmos-Reason2-2B  (N1.7), FROZEN      |     stage 1.1  build_labels     (CPU)
          |  Qwen3-VL backbone -> hidden states     |     stage 1.2  extract_features (GPU)
          +--------------------+--------------------+
                               v   pooling  { mean_last | LAST_LAST | mean_mid }   (last_last is best)
                    pooled feature [2048-d]  -->  cached to disk
                               |                  (train the head, no re-run of the GPU)
  - - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - - - - - - - - - - -
                               v
          +-----------------------------------------+
          |  ProgressHead (robometer recipe)        |     stage 2  train.py         (CPU ok)
          |  Linear->LayerNorm->GELU->Dropout->Lin  |
          +--------------------+--------------------+
                               v   discrete 10-bin softmax -> expected value
                       local_progress in [0,1]
                               |
             language_sensitivity = rho(pred|correct sub-goal) - rho(pred|WRONG sub-goal)
```

Everything above the dashed line is the **frozen** backbone (run once, cached); everything below
**trains** — a small MLP head, seconds per fit. **Locked baseline** (500 episodes, discrete CE). 
Full baseline details, findings, and gotchas are in **`vr_value_head/README.md`**.

## TODO

**Improve the baseline (cheap, stage-2 only, reuse the cache):**
- [ ] Regularization sweep (dropout 0.1→0.3, weight-decay↑) to close the train→test overfit gap (0.81 vs 0.32).
- [ ] Fix the `remaining` head: it has no training target yet, so derived global progress `(k+local)/(k+remaining)` is unusable.

**Bigger levers (need a re-extract or backbone training):**
- [ ] `ARM=full` — the faithful hierarchical span (global + grounded history + sub-goal); the intended handover config. Needs a re-extract.
- [ ] Denser frames (`STRIDE=5`) and/or more ego sources — more within-segment signal. Needs a re-extract.
- [ ] **VLM fine-tuning** — unfreeze the backbone; the main lever to break the frozen-feature ceiling (~0.32) and enable robometer's `<|prog_token|>` + reverse/rewind augmentations.
- [ ] Paraphrase-consistency loss — robustness to wording; turns on with the corpus release (egoverse has no rephrasings).

**Research milestones (`research/VALUE_HEAD_RESEARCH_PLAN.md`):**
- [ ] M3 — cross-domain transfer: does ego pretraining beat from-scratch on a robot/LIBERO progress head? (the core research bet)
- [ ] M6 — wire the head into RL: reward shaping / critic init in RLinf.

## Layout

Value-head-relevant paths (the rest of the repo is the RLinf RL framework, not needed to run the head):

```
vr_value_head/                     the value-head package
  build_labels.py                  stage 1.1: egoverse -> cache/pairs.parquet + frames/*.npz [zarr]
  extract_features.py              stage 1.2: frozen N1.7 -> cache/features/*.npz            [GPU]
  train.py                         stage 2:   train ProgressHead; best-val ckpt; report train/val/test
  evaluate.py                      held-out metrics (per-segment rho, language_sensitivity)
  model.py                         ProgressHead — robometer head; continuous huber or discrete-CE
  losses.py                        discrete_ce · huber · huber_neg(off) · pairwise_rank
  dataset.py                       join pairs + feature cache -> segment-block batches
  text_arms.py                     compose the text slot: {subgoal | full}
  corpus_reader.py                 FUTURE corpus-release reader (not active)
  config/ego_egoverse.yaml         the LOCKED baseline config
  README.md                        detailed package doc (baseline, findings, PAIRS contract, gotchas)
  cache/  runs/                    (generated) feature cache / trained heads + metrics

bashscripts/pretrain_value_head/run.sh   driver — orchestrates all 3 stages
research/probes/                         egoverse reader + metrics (imported by vr_value_head)
research/VALUE_HEAD_RESEARCH_PLAN.md      the M0–M7 research plan
logs/pretrain_value_head/                (generated) per-run logs
```

## Installation

No `pip install` step — the pipeline runs in an existing venv. **`/opt/venv/egoverse`** already carries
everything: the full GR00T N1.7 torch stack (**torch 2.7.1+cu128, transformers 4.57.3, sm_120**) **and**
`zarr` (to read egoverse). It is the only venv with both, so all three stages use it.

| what | path (default) | note |
|---|---|---|
| venv | `/opt/venv/egoverse` | torch 2.7.1+cu128 (Blackwell/sm_120), transformers 4.57.3, zarr |
| backbone | `/mnt/workspace/HUGGINGFACE_HOME/hub/Cosmos-Reason2-2B` | base pretrained VLM (no robot/LIBERO) |
| dataset | `/mnt/workspace2/all_dataset/egoverse` | 500 ego episodes |
| GPU | any CUDA (sm_120 verified) | needed only for stage 2 (feature extraction) |
| disk | ~50 GB | frame cache for 500 episodes |

Rebuilding the venv from scratch must reproduce the torch 2.7.1+cu128 (sm_120) + transformers 4.57.3 +
zarr combination — a fresh install that pins torch 2.6/cu124 will **not** drive the backbone on
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

Output → `vr_value_head/runs/<RUN_NAME>/{head.pt, metrics.json}`, logs under
`logs/pretrain_value_head/<RUN_NAME>/`. Expected held-out result:

```
test  spearman_head 0.32   language_sensitivity 0.31   (clears M2 GO bar >= 0.3)
```

Common env overrides (all have baseline defaults): `EPISODES POOL N_BINS EPOCHS LR STRIDE ARM
TIME_WARP SEGS_PER_BATCH BATCH_SIZE DEVICE CACHE RUN_NAME STAGE`.
