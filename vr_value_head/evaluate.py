"""Held-out metrics — reuses probes/metrics.py and adds the head-specific tests.

The absolute ρ is not the bar; BEATING THE CLOCK is. Every number is reported against a
baseline:
  clock          per-frame elapsed index within its segment (the thing vision must beat)
  language_gap   ρ(correct sub-goal) − pred-on-wrong (matched-frame language sensitivity)
  time_warp      ρ under non-uniform temporal resampling (a clock collapses here by design)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "probes"))
from metrics import evaluate as probe_eval  # noqa: E402  (spearman, rank acc, monotonicity)


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


def report(head, view, device="cpu") -> dict:
    rows = predict(head, view, device)
    pos = rows[rows.role == "pos"].rename(columns={"local_progress": "target"})
    # per-trajectory (per-segment) Spearman of prediction vs proxy target
    m = probe_eval(pos.assign(episode_id=pos.episode_id + "#" + pos.seg_idx.astype(str)),
                   pred_col="pred", target_col="target")
    out = {"spearman_head": m.get("spearman"), "rank_acc": m.get("pairwise_rank_acc"),
           "monotonicity": m.get("monotonicity"), "clock": clock_baseline(rows)}
    out["beats_clock"] = (out["spearman_head"] or 0) - (out["clock"] or 0)

    # Language sensitivity = rho(pred | correct sub-goal) - rho(pred | WRONG sub-goal), both
    # per-segment Spearman against the SAME true local progress (from the matched positive).
    # This is the plan's definition and is RANK-based, so it is immune to the abstain-to-0.5
    # offset that made a raw mean(correct-wrong) gap ~0 by construction. If the head ignores
    # language, pred is identical under either sub-goal -> rho_wrong == rho_correct -> sens ~ 0.
    # If it uses language, the wrong sub-goal should break progress tracking -> rho_wrong lower.
    from scipy.stats import spearmanr

    def per_seg_rho(df, pcol, tcol):
        rs = [spearmanr(g[pcol], g[tcol]).correlation
              for _, g in df.groupby(["episode_id", "seg_idx"])
              if g[pcol].nunique() > 2 and g[tcol].nunique() > 2]
        return float(np.nanmean(rs)) if rs else float("nan")

    neg = rows[rows.role == "neg_subgoal"].copy()
    if len(neg):
        truth = pos[["episode_id", "frame", "seg_idx", "target"]]     # true progress per frame
        negm = neg.merge(truth, on=["episode_id", "frame", "seg_idx"], how="inner")
        rho_correct = per_seg_rho(pos, "pred", "target")
        rho_wrong = per_seg_rho(negm, "pred", "target")
        out["rho_correct_subgoal"] = rho_correct
        out["rho_wrong_subgoal"] = rho_wrong
        out["language_sensitivity"] = (rho_correct or 0.0) - (rho_wrong or 0.0)
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
