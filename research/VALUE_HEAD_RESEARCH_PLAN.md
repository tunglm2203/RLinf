# Value-Head Research Plan — hierarchical VLA with a language-conditioned progress head

**Design under test** — hierarchical, company-fixed foundation (2026-08). The value head attaches
to it; the backbone / policy structure is not ours to change.

```
high-level  :  VLM(RGB, global instruction, textual history)     → sub-goal instruction
   (backbone)                                                       + per-hand & head narratives  (explainability)
low-level   :  VLM_features + proprio + noisy action chunk        → action chunk
   (flow-matching action expert)
value head  :  MLP( pooled VLM_features )                         → local_progress ∈ [0,1]
   (ours)      same postfix span as the action expert, minus         (+ remaining_subgoals aux)
               proprio & noisy action                                 global progress DERIVED, never regressed
```

- **High-level replans step by step** (per chunk): no plan is known up front; the sub-goal can
  change when the environment is dynamic or the current plan is poor. Ultimate target is
  **real-world** deployment. ⇒ the number of sub-goals `K` is *never* an input, and global progress
  is **non-monotone** (discovering more work makes it drop) — so global is a diagnostic, not an RL
  potential. See M2 and M6.
- **Working assumption from the backbone team**: the pretrained VLM predicts accurate sub-goals,
  consistent across chunks (paraphrase accepted). This assumption is what makes a *postfix* value
  feature safe (see M2, "Why postfix is safe") — if it fails, revisit the span.

Stages: ego-centric pretrain → robot pretrain (OXE) → SFT on demos (action expert + value head) → RL.

---

## Guiding principle: DO NOT serialize behind backbone pretraining

The value-head design can be **falsified or validated with a frozen off-the-shelf VLM**. Backbone
pretraining (Qwen3.5-VL-0.8B on ego + OXE) is a multi-GPU-week job that is *not* feasible on
2×RTX-5090 and must not gate the research question.

```
track A (cheap, 2 GPUs) : value-head design  ── M0 → M6 ──►  validated recipe
track B (cluster)       : backbone pretrain  ────────────►  swap in when ready
```

Run track A on a frozen public VLM (Qwen3-VL-4B, or your 0.8B base once it exists). Every milestone
below is backbone-agnostic: if the design works on a frozen public VLM, it will work better on your
pretrained one. If it fails there, pretraining will not save it.

---

## Kill criteria (read this first)

Stop and rethink — do not proceed to the next milestone — if:

| signal | meaning |
|---|---|
| M1 linear probe ρ < 0.3 | pooled VLM features do not encode progress → fix features/pooling, not data |
| M2 language-sensitivity ≈ 0 (matched-frame) | head ignores the instruction; it is an image-only clock, not a reward model |
| M2 time-warp collapses the head | it *was* a clock; ego pretraining fit a stopwatch → the whole approach, not the data |
| M2 paraphrase spread ≳ per-chunk Δ | Φ jitters on rewording more than on progress → unusable as shaping reward at any ρ |
| M3 ego-pretrain gives no gain over from-scratch | ego data is not buying transfer → drop it, train in-domain only |
| M4 success/failure AUROC < 0.7 | head cannot tell success from failure → useless as RL reward |
| M6 reward ↑ but success ↓ | reward hacking → tighten shaping, keep sparse anchor |

---

## M0 — Instrumentation & baselines  *(3–5 days, no training)*

**Build**
1. Label generator: corpus → records `(clip_id, frame, instruction, granularity, progress, split)`.
   - level-2 sub-goal instruction + within-segment progress (primary)
   - level-1 global instruction + whole-clip progress (20–30% mix)
   - filters: drop `joystick_v1`; drop degenerate segments (<5 frames); merge duplicate spans;
     flag egodex 2s-grid boundaries as `soft_boundary=True`
   - negatives: wrong-instruction pairs (clip frames × another clip's sub-goal)
2. Eval harness with these metrics:
   - **Spearman ρ** between predicted and **labelled (proxy) progress**, computed **per trajectory**
     then averaged.
     ⚠️ The label is a *temporal proxy*, **not** true progress. True progress `p(s)` is a function
     of the **state** — how much of the goal condition is satisfied. Our labels
     (`local` / `global` / `composite`) are all functions of **position in the trajectory**, so they
     silently assume the demonstrator advances toward the goal at a uniform rate. Where that
     assumption breaks (dithering, backtracking, failure), the label is wrong and the metric
     rewards a model for reproducing the error. This is exactly why a `frame_index` clock is such a
     strong baseline — and why beating the clock, not the absolute ρ, is the real bar.
   - **pairwise rank accuracy** (frame i earlier than j → lower progress)
   - **monotonicity** = fraction of consecutive steps with Δprogress ≥ 0
   - **language sensitivity** = ρ(correct instruction) − ρ(wrong instruction)   ← the critical one
   - **success/failure AUROC** (once failures exist, M4)

   **Evaluation protocol — the flow** (as built in `probes/`):

   ```
   (frame, instruction)
        │
        ▼  frozen VLM, torch.no_grad          extract_features.py
   hidden states [B, ~314, 2048]
        │
        ▼  pool → one 2048-d vector           {mean_last | last_last | mean_mid}
   .npz cache on disk                          ← extract ONCE (~6 min / 16k frames)
        │
        ▼  StandardScaler + Ridge(alpha)      run_probe.py — the ONLY fitted part
   predicted progress, clipped to [0,1]        (2048 weights + bias; no hidden layer)
        │
        ▼  evaluate on HELD-OUT EPISODES      metrics.py
   ρ · rank acc · monotonicity · lang-sensitivity · AUROC
   ```

   Three protocol rules that materially change the numbers:
   - **split by EPISODE, never by frame** — adjacent frames are near-duplicates; frame-level
     splitting leaks them across train/test and inflates ρ;
   - **sweep `alpha`** [1 … 10⁴] — with 2048 features and a few thousand samples this dominates
     (M4: alpha 1 → 0.818 vs alpha 100 → 0.878 AUROC);
   - **fit the scaler on train only**, and keep the backbone frozen so the cache is reusable —
     extraction is minutes, each probe fit is seconds, so pooling × alpha × context × target can
     all be swept without touching the GPU again.
3. Baselines to beat: constant 0.5; frame-index oracle (upper bound on purely temporal info);
   image-only head (no language) — this is the one language sensitivity must beat.

**Exit** — harness runs end-to-end on the 6-clip sample; baseline numbers reproduce sanely.

---

## M1 — Feasibility probe: are VLM features progress-informative?  *(3–5 days)*

The cheapest decisive experiment. **Do not skip.**

**Do**: freeze the VLM, extract mean-pooled `vl_embs` for every frame, fit a **linear** probe → progress.
Run it twice: on ego clips, and on a handful of robot/LIBERO demos.

**Data**: 50–200 clips is enough. (The 6-clip bundle only smoke-tests the pipeline.)

**Metric / GO**: held-out per-trajectory Spearman **ρ ≥ 0.5** → strong signal, proceed.
ρ 0.3–0.5 → weak; try last-token vs mean pooling, a different layer, more frames of context.
**ρ < 0.3 → STOP**: the representation, not the data, is the problem.

---

## M2 — Value head v1 on ego data  *(1–2 weeks)*

Rewritten 2026-08 for the hierarchical foundation above. Supersedes the pre-hierarchical version,
which assumed the head took `(RGB, sub-goal)` directly.

### Head spec

| | |
|---|---|
| **input** | pooled VLM features from the **postfix span** = RGB + global instruction + grounded history + predicted sub-goal (**narratives excluded**). Detached from the backbone. |
| **pooling** | sweep `mean_last` / `last_last` / `mean_mid` on cached features (seconds per fit). ⚠️ `last_last` must index `mask_length − 1`, **not** `attention_mask.sum(1) − 1` — the latter lands on a pad on N1.7 and cost a 0.40-vs-0.99 eval gap once. |
| **trunk** | 2-layer MLP. Deliberately small — capacity in the head fits ego-specific appearance, which is exactly what must *not* transfer. |
| **output** | `local_progress ∈ [0,1]` (primary). `remaining_subgoals` (auxiliary, optional). Global is **derived** `= (k + local)/(k + remaining)` with `k` read from grounded history — **never regressed**, because `K` is not an input at rollout. |

**Why postfix (and not prefix).** Postfix matches the action expert's conditioning span exactly
(minus proprio + noisy action), so the head and the policy read the *same* state — a prerequisite if
the head ever initializes the critic. It costs state-purity: Φ now reads the generated sub-goal, so
strictly Φ = Φ(s, a_text), and PBRS policy-invariance no longer holds exactly.

**Why postfix is safe *given the consistency assumption*.** Within a sub-goal segment the text is
constant, so in the shaping term `γΦ(s′) − Φ(s)` the sub-goal contribution **telescopes away** — the
per-chunk Δ is driven by RGB/state change, not by the text. Text-dependence only bites (a) at sub-goal
boundaries, where the text genuinely changes → handled by per-sub-goal shaping + completion bonus in
M6, boundary Δ never fed raw; and (b) via paraphrase jitter chunk-to-chunk → why the paraphrase loss
below is **required, not optional**. Per-chunk replanning also means the backbone runs every chunk, so
postfix features are available every chunk (Φ updates at chunk rate). *If the consistency assumption
fails, revisit the span — prefix-only is the fallback and buys back the PBRS guarantee.*

### Losses — each tied to an observed failure mode, not a hunch

| loss | the failure it targets |
|---|---|
| Huber on `local_progress` | the base signal |
| **pairwise ranking** within a segment | absolute progress values do not cross the ego→robot gap; order does |
| **wrong-sub-goal negative** (matched frames, different goal) | language sensitivity — the untested M2 kill criterion |
| **paraphrase consistency** (same frames, reworded sub-goal → small L2) | paraphrase jitter entering the reward as spurious Δ. Load-bearing under postfix, since the varying text sits *inside* the pooled feature |
| **time-warp augmentation** (non-uniform temporal resampling) | **the clock.** Decorrelates elapsed-frame index from true progress while visual progress is unchanged → a clock baseline collapses, a vision head is unaffected. Converts "we hope it isn't a clock" into "it structurally cannot be one" |

**Pseudo-failure pretraining** (ego is all-successful, so the plateau behaviour has no real
supervision): add cheap negatives ego *can* produce — backward playback (progress must decrease),
temporally scrambled segments (must not advance), cross-episode frames under this sub-goal. This
pretrains the *capability to not advance*; it does **not** substitute for M4 (real stalling ≠ reversed
video).

### The measurement that gates everything: beat the clock, not the threshold

Three independent results say that **on successful data a clock matches vision** — ego ρ 0.369 vs
0.366, LIBERO ρ 0.956 vs 0.946, both "passing" while measuring almost nothing. Postfix does *not*
reduce this: within a segment the sub-goal is constant, so local progress still comes only from RGB +
elapsed time. Every M2 number is reported **against a baseline, never absolute**:
- vs **history-length predictor** (`k`-only) for anything touching global;
- vs **honest clock** (raw elapsed frames) for local;
- and under **time-warp**, where the clock is destroyed by construction.

Language sensitivity is *easier to pass* under postfix (swapping sub-goal tokens changes the pooled
feature directly), so the honest test is the **matched-frame** negative — task 6's two sub-goals on
identical frames — which checks the head uses the sub-goal to *mean* something, not merely registers
that some tokens differ.

**Data**: the ego release (500 episodes / 4,493 segments / 7.6 h already on disk for LIBERO-side dev;
ego corpus is the pretraining set). Prefer **egoverse** (semantic boundaries); down-weight egodex
(2 s grid). Split by `split_group` so no recording leaks across train/val.

**GO criteria (all three), each vs its baseline**
| metric | threshold |
|---|---|
| per-trajectory Spearman ρ | ≥ 0.7 **and > honest clock** |
| monotonicity | ≥ 0.8 |
| **language sensitivity (matched-frame)** | **≥ 0.3** |
| paraphrase spread | **≪ per-chunk Δ** intended for shaping (else unusable as reward regardless of ρ) |

Falsifiers: lang-sensitivity ≈ 0 → head is image-only, hierarchical conditioning buys nothing.
Time-warp performance collapses → it was a clock; ego pretraining fit a stopwatch. Two of these are
measurable on the **LIBERO segments we already have**, before any ego pretraining runs.

---

## M3 — Cross-domain transfer  *(1 week)*  ← the make-or-break for using ego data

Tests the assumption that human ego video helps a robot progress head.

**Do**: three-way comparison on the *same* robot eval set (OXE subset and/or your LIBERO demos):
| arm | training |
|---|---|
| A | ego-only, zero-shot on robot |
| B | from-scratch on robot data (no ego) |
| C | ego-pretrained → finetuned on the same robot data as B |

**GO**: **C > B** by a clear margin at equal robot-data budget (e.g. ρ +0.1), *or* C reaches B's
ρ using ≤ 30% of the robot data. Zero-shot A ρ ≥ 0.4 is a bonus, not required.

**NO-GO**: C ≈ B → ego pretraining buys nothing. That is a legitimate, valuable negative result:
drop the ego stage for the value head and invest in in-domain data instead.

---

## M4 — Failure calibration  *(1–2 weeks)*  ← the gap nothing else fills

Ego, OXE, and demos are **all successful execution**. The head will be confidently wrong exactly
where RL explores.

**Do**: roll out your existing SFT checkpoint, label episodes by **true sim success**, keep the
failures (you already have the checkpoints and the sim). Retrain/finetune the head with:
- successes → progress = t/T
- failures → progress rises then **plateaus**; never force to 1.0

Target ≥ 1k rollouts, ≥ 30% failures (at ~0.3–0.5 SFT success this is automatic).

**GO**: success-vs-failure **AUROC ≥ 0.8** on held-out rollouts using terminal predicted progress,
and mean terminal progress on failures clearly below successes.

---

## M5 — Offline RL-readiness check  *(3–5 days)*  ← cheap insurance before burning GPU

Before wiring any RL. On held-out rollouts, compute the shaped reward `r_t = Δ progress` and check:

1. **corr(episode shaped return, true success) ≥ 0.5**
2. **return variance ≫ sparse-reward baseline** — this is precisely what killed the earlier PPO runs
   (near-zero return variance → dead critic)
3. no pathological spikes at sub-goal boundaries (the 1.0→0.0 reset trap)

**GO**: all three. This predicts RL benefit without running RL.

---

## M6 — RL integration, single task  *(1–2 weeks)*

**Do**: wire into RLinf as (a) reward shaping via the reward worker and/or (b) critic init.
Shape, don't replace. **Shape per sub-goal on LOCAL progress — never on global** (global is
non-monotone under online replanning, so it would punish honest plan revision):

```
Φ = local_progress of the CURRENT sub-goal
r = reward_coef·success  +  λ·(γΦ(s') − Φ(s))  +  λ·bonus·1[sub-goal completed]
```

- The **completion bonus** cancels the local 1→0 reset at boundaries (the reset trap M5 checks for) —
  the standard options treatment. Φ never touches `K̂` (replanning is free) or `k` (closes the
  history-counter exploit).
- **Sub-goal completion must be *grounded***, not "the text changed" — under step-by-step replanning
  a sub-goal can end by being *abandoned*, which looks identical to completion from the outside. In
  sim = a predicate; in the real world this is the open problem (see M4 / ground-truth discussion).
- **γ inside** the potential is required for γ<1 (Ng et al.); the shaping then telescopes to
  `γ^T·Φ(s_T) − Φ(s_0)` and preserves the optimal policy — *approximately*, since postfix makes Φ read
  the generated sub-goal (M2). Within a segment the text is constant so this holds; the anchor covers
  the residual.

Start with **LIBERO task 0**, GR00T N1.7 (or N1.5), the known SFT baseline for reference.

**GO**: success rate **> SFT baseline by ≥ 5 points**, with
- `explained_variance` off the floor (was −10⁴…−10⁸ with sparse reward)
- **no reward hacking**: shaped reward and true success move together

**Hacking check**: if shaped reward climbs while success is flat/down → reduce λ, strengthen the
sparse anchor, or add uncertainty gating. Three channels specific to this architecture, to watch:
sub-goal churn farming the completion bonus (→ ground the signal); spurious history completions
inflating `k` (→ ground the history); paraphrase spikes in Φ (→ M2 paraphrase loss). If the high-level
is RL-trained (not frozen), add a **KL anchor to the pretrained high-level** — otherwise RL erodes the
very consistency assumption that makes postfix Φ safe.

---

## M7 — Scale-out  *(ongoing)*

1. All 10 LIBERO tasks / your full benchmark.
2. Mixed-granularity querying: verify one head serves both `V(RGB, global)` and `V(RGB, sub-task)`.
3. **Hierarchical coupling test** — the risk unique to this architecture: condition the value head on
   sub-goals **predicted by the high-level policy** instead of ground truth. Measure the drop.
   Mitigate with scheduled sampling (train on predicted sub-goals) + paraphrase augmentation.
4. Swap in the pretrained Qwen3.5-VL-0.8B backbone (track B) and re-run M2/M5/M6 as regression tests.

---

## Data scale summary

| milestone | data needed | have today |
|---|---|---|
| M0 | 6 clips (pipeline smoke test) | ✅ sample bundle |
| M1 | 50–200 clips | ❌ need release |
| M2 | ≥ 1k clips (~5k segments) | ❌ need release |
| M3 | + robot data (OXE subset / LIBERO demos) | ⚠️ LIBERO demos yes |
| M4 | ≥ 1k self-generated rollouts, ≥30% failures | ✅ generatable now |
| M6 | RL compute | ✅ 2×5090 |

**Two things you can start today without more corpus data:** M0 (instrumentation) and M4's rollout
collection (self-generated failure data from the existing SFT checkpoint). Both are on the critical
path and neither is blocked.

---

## Suggested ordering (parallelism)

```
week 1-2   M0 instrumentation            ║ start M4 rollout collection (background, free)
week 2-3   M1 linear probe               ║ (rollouts accumulating)
week 3-5   M2 value head v1              ║
week 5-6   M3 transfer  ← key decision   ║
week 6-8   M4 failure calibration
week 8-9   M5 offline readiness check
week 9-11  M6 RL integration, task 0
then       M7 scale-out + backbone swap
```

Track B (backbone pretraining) runs independently on cluster resources and is swapped in at M7.
