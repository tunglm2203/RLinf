"""M0 — evaluation metrics for a progress / value head.

All metrics are computed PER TRAJECTORY then averaged, because a global correlation over
pooled frames is dominated by between-episode variance and looks good even when the model
cannot order frames *within* an episode — which is the thing RL actually needs.

    from metrics import evaluate, baseline_table
    evaluate(df, pred_col="pred", target_col="composite")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _per_traj(df: pd.DataFrame, fn, group="episode_id") -> float:
    vals = [fn(g) for _, g in df.groupby(group) if len(g) >= 3]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def spearman(df: pd.DataFrame, pred_col: str, target_col: str) -> float:
    """Rank correlation between prediction and target, averaged over trajectories."""
    def f(g):
        if g[target_col].nunique() < 2 or g[pred_col].nunique() < 2:
            return None
        return spearmanr(g[pred_col], g[target_col]).correlation
    return _per_traj(df, f)


def pairwise_rank_accuracy(df: pd.DataFrame, pred_col: str, target_col: str,
                           max_pairs: int = 2000, seed: int = 0) -> float:
    """P(model orders a random frame pair the same way the label does)."""
    rng = np.random.default_rng(seed)

    def f(g):
        p, t = g[pred_col].to_numpy(), g[target_col].to_numpy()
        n = len(g)
        k = min(max_pairs, n * (n - 1) // 2)
        if k <= 0:
            return None
        i = rng.integers(0, n, k)
        j = rng.integers(0, n, k)
        m = t[i] != t[j]
        if m.sum() == 0:
            return None
        return float((np.sign(p[i] - p[j])[m] == np.sign(t[i] - t[j])[m]).mean())
    return _per_traj(df, f)


def spearman_pooled(df: pd.DataFrame, pred_col: str, target_col: str) -> float:
    """Rank correlation over ALL held-out frames pooled, i.e. ACROSS episodes.

    Why this matters: when an episode has no sub-goal structure, progress == normalised time,
    so a clock scores a perfect per-trajectory rho by construction and the metric says nothing.
    Cross-episode ranking ("is this frame of episode A further along than that frame of
    episode B?") cannot be answered by a clock, because at rollout time the episode length T
    is unknown. This is therefore the honest test of *visual* progress understanding on
    single-segment data such as LIBERO demos.
    """
    d = df.dropna(subset=[pred_col, target_col])
    if d[target_col].nunique() < 2 or d[pred_col].nunique() < 2:
        return float("nan")
    return float(spearmanr(d[pred_col], d[target_col]).correlation)


def monotonicity(df: pd.DataFrame, pred_col: str, time_col: str = "frame") -> float:
    """Fraction of consecutive steps where the prediction does not decrease."""
    def f(g):
        d = g.sort_values(time_col)[pred_col].diff().dropna()
        return float((d >= -1e-9).mean()) if len(d) else None
    return _per_traj(df, f)


def language_sensitivity(df: pd.DataFrame, pred_col: str, target_col: str,
                         neg_col: str = "is_negative") -> dict:
    """Does the head actually use the instruction?

    Compares predictions on correct-instruction rows vs the SAME frames paired with a
    wrong instruction. If the gap is ~0 the model is an image-only clock and the language
    conditioning is decorative — which silently breaks the whole premise at RL time.
    """
    if neg_col not in df.columns or not df[neg_col].any():
        return {"lang_sensitivity": float("nan"), "note": "no negatives in df"}
    pos, neg = df[~df[neg_col]], df[df[neg_col]]
    rho_pos = spearman(pos, pred_col, target_col)
    key = ["episode_id", "frame"]
    merged = pos.merge(neg[key + [pred_col]], on=key, suffixes=("", "_neg"))
    gap = float((merged[pred_col] - merged[f"{pred_col}_neg"]).abs().mean()) if len(merged) else float("nan")
    return {"rho_correct_instruction": rho_pos,
            "mean_abs_pred_gap": gap,
            "n_paired": len(merged)}


def evaluate(df: pd.DataFrame, pred_col: str = "pred", target_col: str = "composite",
             include_negatives: bool = False) -> dict:
    d = df if include_negatives else df[~df.get("is_negative", pd.Series(False, index=df.index))]
    d = d.dropna(subset=[pred_col, target_col])
    out = {
        "n_rows": len(d),
        "n_episodes": d["episode_id"].nunique() if "episode_id" in d else 0,
        "spearman": spearman(d, pred_col, target_col),
        "spearman_pooled": spearman_pooled(d, pred_col, target_col),
        "pairwise_rank_acc": pairwise_rank_accuracy(d, pred_col, target_col),
        "monotonicity": monotonicity(d, pred_col),
        "mae": float((d[pred_col] - d[target_col]).abs().mean()),
    }
    if "is_negative" in df.columns and df["is_negative"].any():
        out.update(language_sensitivity(df.dropna(subset=[pred_col]), pred_col, target_col))
    return out


def baseline_table(df: pd.DataFrame, target_col: str = "composite") -> pd.DataFrame:
    """Reference points a real model must beat.

    constant      — no information at all
    frame_index   — normalized time; the *upper bound on purely temporal* information.
                    A vision model that only matches this has learned nothing visual.
    seg_index     — knows which segment it is in, but not where inside it
    """
    d = df[~df.get("is_negative", pd.Series(False, index=df.index))].copy()
    rows = []
    for name, col in [
        ("constant_0.5", np.full(len(d), 0.5)),
        ("frame_index", (d["frame"] / d["n_frames"].clip(lower=1)).to_numpy()),
        ("seg_index", (d["seg_idx"] / d["n_segs"].clip(lower=1)).to_numpy()),
    ]:
        d["_p"] = col
        m = evaluate(d, "_p", target_col)
        rows.append({"baseline": name, **{k: m[k] for k in
                     ("spearman", "spearman_pooled", "pairwise_rank_acc", "monotonicity", "mae")}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/workspace/vla_bench/RLinf/research/probes/labels/progress_labels.parquet"
    df = pd.read_parquet(p)
    print(f"loaded {len(df):,} rows / {df.episode_id.nunique()} episodes from {p}\n")
    for tgt in ("composite", "local_progress", "global_progress"):
        print(f"=== baselines vs target `{tgt}` ===")
        print(baseline_table(df, tgt).to_string(index=False), "\n")
