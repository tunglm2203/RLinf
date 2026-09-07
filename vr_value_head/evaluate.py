"""Held-out metrics — reuses probes/metrics.py and adds the head-specific tests.

The absolute ρ is not the bar; BEATING THE CLOCK is. Reported against baselines:
  clock          per-frame elapsed index within its segment (the thing vision must beat)
  language_gap   ρ(correct sub-goal) − pred-on-wrong (matched-frame language sensitivity)
  time_warp      ρ under non-uniform temporal resampling (a clock collapses here by design)

Also reports the robometer / GVL progress metrics (same definitions as robometer):
  pearson_voc    Value-Order Correlation — per-segment Pearson(pred, target)
  kendall_tau_a  per-segment Kendall τ-a, TIE-IGNORING (robometer's hand-rolled variant)
  succ_fail_diff mean terminal predicted progress on SUCCESS − on FAILURE (needs failure data)
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "probes"))
from metrics import evaluate as probe_eval  # noqa: E402  (spearman, rank acc, monotonicity)


def kendall_tau_a(x, y) -> float:
    """Kendall τ-a, tie-ignoring — robometer's exact impl (eval_metrics_utils.py). NOT scipy's
    τ-b: concordant−discordant over n(n−1)/2, pairs with a tie on either side skipped."""
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    if n < 2:
        return float("nan")
    C = D = 0
    for i, j in itertools.combinations(range(n), 2):
        dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if dx == 0 or dy == 0:
            continue
        C += (dx == dy); D += (dx != dy)
    return (C - D) / (n * (n - 1) / 2)


def _per_seg(df, pcol, tcol, fn) -> float:
    """Apply correlation `fn(pred, target)` per (episode, segment) and average."""
    rs = [fn(g[pcol].to_numpy(), g[tcol].to_numpy())
          for _, g in df.groupby(["episode_id", "seg_idx"])
          if g[pcol].nunique() > 2 and g[tcol].nunique() > 1]
    return float(np.nanmean(rs)) if rs else float("nan")


def success_failure_diff(rows: pd.DataFrame) -> dict:
    """robometer `succ_fail_diff`: mean TERMINAL predicted progress on success episodes minus on
    failures. Needs a `success` column (failure-inclusive dataset); returns {} if absent, NaN if
    only one class present. Terminal = the last frame of each episode."""
    if "success" not in rows.columns:
        return {}
    pos = rows[rows.role == "pos"]
    term = pos.loc[pos.groupby("episode_id").frame.idxmax()]      # last frame per episode
    s, f = term[term.success.astype(bool)], term[~term.success.astype(bool)]
    if len(s) == 0 or len(f) == 0:
        return {"succ_fail_diff": float("nan")}
    return {"succ_fail_diff": float(s.pred.mean() - f.pred.mean()),
            "term_pred_success": float(s.pred.mean()), "term_pred_failure": float(f.pred.mean())}


def suc_fail_pref_acc(rows: pd.DataFrame, n_pairs: int = 1000, seed: int = 0,
                      task_col: str = "global_task") -> dict:
    """robometer 'Suc-Fail' (Table III) = pairwise PREFERENCE ACCURACY, not a mean gap. Sample
    (success, failure) trajectory pairs OF THE SAME TASK; correct iff the model ranks the success
    above the failure by TERMINAL predicted progress (progress_chosen[-1] > progress_rejected[-1],
    per robometer's compute_preference_accuracy_from_progress). n_pairs = num_examples_per_quality_pr
    (robometer default 1000). Range [0,1], chance 0.5. N/A (absent) without both classes."""
    if "success" not in rows.columns or task_col not in rows.columns:
        return {}
    pos = rows[rows.role == "pos"]
    term = pos.loc[pos.groupby("episode_id").frame.idxmax()]          # terminal frame per episode
    per_task = []                                                     # (success_terminals, failure_terminals)
    for _, g in term.groupby(task_col):
        s = g[g.success.astype(bool)].pred.to_numpy()
        f = g[~g.success.astype(bool)].pred.to_numpy()
        if len(s) and len(f):
            per_task.append((s, f))
    if not per_task:
        return {"suc_fail_acc": float("nan"), "suc_fail_n": 0}
    rng = np.random.default_rng(seed)
    w = np.array([len(s) * len(f) for s, f in per_task], dtype=float)  # sample tasks ∝ #possible pairs
    n = int(min(n_pairs, w.sum()))
    picks = rng.choice(len(per_task), size=n, p=w / w.sum())
    correct = 0
    for ti in picks:
        s, f = per_task[ti]
        correct += int(s[rng.integers(len(s))] > f[rng.integers(len(f))])
    return {"suc_fail_acc": correct / n, "suc_fail_n": n}


@torch.no_grad()
def predict(head, view, device="cpu") -> pd.DataFrame:
    b = view.full()
    head.eval()
    pred = head(b["feats"].to(device))["local"].cpu().numpy()
    rows = view.rows.copy()
    rows["pred"] = pred
    return rows


def clock_baseline(rows: pd.DataFrame) -> float:
    """Per-segment Spearman of the elapsed-frame clock vs the true local target (positives)."""
    from scipy.stats import spearmanr
    pos = rows[rows.role == "pos"]
    rs = []
    for _, g in pos.groupby(["episode_id", "seg_idx"]):
        if g.frame.nunique() > 2:
            rs.append(spearmanr(g.frame, g.local_progress).correlation)
    return float(np.nanmean(rs)) if rs else float("nan")


# --- canonical 7 metrics, applicability, and the experiment signature ----------------------------
METRIC7 = [("spearman_head", "ρ"), ("pearson_voc", "VOC"), ("kendall_tau_a", "τa"),
           ("suc_fail_acc", "SucFail"), ("language_sensitivity", "lang"), ("succ_fail_diff", "SFΔ"),
           ("term_pred_success", "P_succ"), ("term_pred_failure", "P_fail")]


def make_signature(pretrain="none", downstream="none", eval_ds="?", cond="?", target="?",
                   vlm_pre=None, vlm_ft=None) -> str:
    """Canonical experiment signature: P:… → D:… → E:…[cond,target]. vlm_* ∈ {frozen, LoRA}."""
    def stage(data, vlm):
        return "none" if (not data or data == "none") else (f"{data}(VLM={vlm})" if vlm else data)
    return (f"P:{stage(pretrain, vlm_pre)} → D:{stage(downstream, vlm_ft)} "
            f"→ E:{eval_ds}[cond={cond},target={target}]")


def fmt7(m: dict, signature: str = None) -> str:
    """One-line canonical result: signature + the 7 metrics, with N/A where inapplicable."""
    def v(x): return "  N/A " if (x is None or (isinstance(x, float) and x != x)) else f"{x:+.3f}"
    sig = signature or m.get("signature") or ""
    body = "  ".join(f"{lab}={v(m.get(k))}" for k, lab in METRIC7)
    return f"{sig}\n  {body}   (clock={v(m.get('clock'))})"


def report(head, view, device="cpu", signature=None) -> dict:
    """The 7 canonical metrics from the frozen two-stage path (pooled features in `view`)."""
    return report_metrics(predict(head, view, device), signature)


def report_metrics(rows: pd.DataFrame, signature=None) -> dict:
    """The 7 canonical metrics from a rows frame carrying a `pred` column — works for BOTH the frozen
    two-stage path and on-the-fly LoRA encoding, so the two report identically. ROBOMETER convention:
    progress correlations (ρ/VOC/τa) and language sensitivity on SUCCESS trajectories only when the
    eval set carries a `success` label; SFΔ/P_succ/P_fail over ALL tiers, auto-N/A without failures."""
    from scipy.stats import pearsonr, spearmanr
    pos = rows[rows.role == "pos"].rename(columns={"local_progress": "target"})
    # success-only subset for correlations + language (robometer's rule); all-pos if unlabeled
    corr = pos[pos.success.astype(bool)] if "success" in pos.columns else pos

    out = {"clock": clock_baseline(rows), "signature": signature}
    out["spearman_head"] = _per_seg(corr, "pred", "target", lambda a, b: spearmanr(a, b).correlation)
    out["pearson_voc"] = _per_seg(corr, "pred", "target",
                                  lambda a, b: pearsonr(a, b)[0] if np.std(a) and np.std(b) else np.nan)
    out["kendall_tau_a"] = _per_seg(corr, "pred", "target", kendall_tau_a)
    out["beats_clock"] = (out["spearman_head"] or 0) - (out["clock"] or 0)

    # language sensitivity = rho(correct sub-goal) - rho(wrong sub-goal), on the SAME success subset
    neg = rows[rows.role == "neg_subgoal"]
    if len(neg):
        truth = corr[["episode_id", "frame", "seg_idx", "target"]]
        negm = neg.merge(truth, on=["episode_id", "frame", "seg_idx"], how="inner")
        rc = _per_seg(corr, "pred", "target", lambda a, b: spearmanr(a, b).correlation)
        rw = _per_seg(negm, "pred", "target", lambda a, b: spearmanr(a, b).correlation)
        out["rho_correct_subgoal"], out["rho_wrong_subgoal"] = rc, rw
        out["language_sensitivity"] = (rc or 0.0) - (rw or 0.0)

    # quality-separation (ALL tiers). success_failure_diff returns {} when no `success` col -> N/A
    out.update(success_failure_diff(rows))
    out.update(suc_fail_pref_acc(rows))          # robometer Table-III Suc-Fail (pairwise pref accuracy)
    return out


@torch.no_grad()
def time_warp_report(head, view, device="cpu", n=5, seed=0) -> dict:
    """ρ under repeated non-uniform temporal resampling. A clock drops sharply here; a genuine
    vision head barely moves. The gap between the two is the honest anti-clock evidence."""
    from scipy.stats import spearmanr
    head.eval()
    hs, cs = [], []
    for r in range(n):
        for b in view.batches(segs_per_batch=64, time_warp=True, seed=seed + r):
            pos = b["pos_mask"].numpy()
            if pos.sum() < 3:
                continue
            pred = head(b["feats"].to(device))["local"].cpu().numpy()[pos]
            tgt = b["target_local"].numpy()[pos]
            gid = b["group_id"].numpy()[pos]
            for gg in np.unique(gid):
                mm = gid == gg
                if mm.sum() > 2:
                    hs.append(spearmanr(pred[mm], tgt[mm]).correlation)
    return {"spearman_head_timewarp": float(np.nanmean(hs)) if hs else float("nan")}
