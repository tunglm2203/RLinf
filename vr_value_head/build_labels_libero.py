"""M3 adapter: LIBERO-10 segments -> the SAME PAIRS table + frame cache that egoverse produces.

The robot side of the cross-domain transfer test (M3). Mirrors build_labels.py, but the source is
the LIBERO rollout segments (predicate-grounded) instead of egoverse:

  segments : research/probes/labels/libero_segments_clean.csv   (subgoal + start/end frame per seg)
  frames   : research/collected_rollouts/libero_segments_v2/<episode_id>.npz  (key 'frames' [N,H,W,3])

Output is byte-compatible with extract_features.py / dataset.py:
  <out>/pairs.parquet          pos + neg_subgoal rows, same schema as egoverse (+ task_id)
  <out>/frames/<eid>.npz       {frame_idx, img}   (sampled frames, egoverse cache format)

Split is EPISODE-level, PER TASK (so every task appears in both train pool and test). transfer.py
then subsamples k demos/task from the train pool. `local_progress = (f - start)/(len-1)` matches
egoverse's within-segment definition; negatives = the same frame under the OTHER segment's sub-goal.

    /opt/venv/egoverse/bin/python build_labels_libero.py --arm subgoal --test-frac 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import text_arms  # noqa: E402

HERE = Path(__file__).parent
REPO = HERE.parent
CSV = REPO / "research/probes/labels/libero_segments_clean.csv"
FRAMES_DIR = REPO / "research/collected_rollouts/libero_segments_v2"


def episode_split(clean: pd.DataFrame, test_frac: float, min_train: int, seed: int) -> dict:
    """Deterministic EPISODE-level split within each task (so every task is in both train & test)."""
    rng = np.random.default_rng(seed)
    split = {}
    for tid, g in clean.groupby("task_id"):
        eps = sorted(g.episode_id.unique())
        rng.shuffle(eps)
        n_test = max(1, min(round(test_frac * len(eps)), len(eps) - min_train))
        for i, e in enumerate(eps):
            split[e] = "test" if i < n_test else "train"
    return split


def build(out_dir: Path, arm: str, stride: int, n_neg: int, test_frac: float,
          min_train: int, seed: int):
    clean = pd.read_csv(CSV)
    split = episode_split(clean, test_frac, min_train, seed)
    frames_out = out_dir / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    rows, pair_id = [], 0
    for eid, ep in clean.groupby("episode_id"):
        segs = ep.sort_values("seg_idx").to_dict("records")
        src = FRAMES_DIR / f"{eid}.npz"
        if not src.exists():
            continue
        frames = np.load(src)["frames"]                       # [N,H,W,3] uint8
        all_subgoals = [s["subgoal"] for s in segs]
        sp = split.get(eid, "train")

        keep = set()
        for s in segs:
            k = int(s["seg_idx"])
            length = int(s["end_frame"]) - int(s["start_frame"])
            history = all_subgoals[:k]                         # grounded: prior segments' sub-goals
            for f in range(int(s["start_frame"]), int(s["end_frame"]), stride):
                if f >= frames.shape[0]:
                    break
                keep.add(f)
                local = 0.0 if length <= 1 else min(1.0, (f - s["start_frame"]) / (length - 1))
                text = text_arms.compose(arm, s["subgoal"], s["global_task"], history)
                rows.append(dict(pair_id=pair_id, episode_id=eid, frame=int(f), seg_idx=k,
                                 n_segs=len(segs), local_progress=float(local), role="pos",
                                 arm=arm, text=text, split=sp, split_group=eid,
                                 global_task=s["global_task"], task_id=int(s["task_id"])))
                pair_id += 1
                # negative: same frame, a DIFFERENT segment's sub-goal (same episode)
                others = [t for t in all_subgoals if t != s["subgoal"]]
                for wt in (others[:n_neg] if others else []):
                    rows.append(dict(pair_id=pair_id, episode_id=eid, frame=int(f), seg_idx=k,
                                     n_segs=len(segs), local_progress=float("nan"),
                                     role="neg_subgoal", arm=arm,
                                     text=text_arms.compose(arm, wt, s["global_task"], history),
                                     split=sp, split_group=eid, global_task=s["global_task"],
                                     task_id=int(s["task_id"])))
                    pair_id += 1

        idx = sorted(keep)
        np.savez_compressed(frames_out / f"{eid}.npz",
                            frame_idx=np.asarray(idx, dtype=np.int64),
                            img=frames[idx].astype(np.uint8))

    pairs = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "cache_libero")
    ap.add_argument("--arm", choices=["subgoal", "full"], default="subgoal")
    ap.add_argument("--stride", type=int, default=1, help="LIBERO segments are short; 1 keeps all")
    ap.add_argument("--neg", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--min-train", type=int, default=6, help="keep >= this many train eps/task (task 0 cap)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = build(args.out, args.arm, args.stride, args.neg, args.test_frac, args.min_train, args.seed)
    if pairs.empty:
        raise SystemExit("no pairs produced (check frame npz / CSV paths)")
    pos = pairs[pairs.role == "pos"]
    print(f"pairs {len(pairs):,}  ({len(pos):,} pos / {int((pairs.role=='neg_subgoal').sum()):,} neg)")
    print(f"episodes {pairs.episode_id.nunique()} | tasks {sorted(pairs.task_id.unique())} | arm={args.arm}")
    ep = pairs.drop_duplicates("episode_id")
    print("\ntrain/test episodes per task (train pool feeds the k-sweep):")
    tab = ep.groupby(["task_id", "split"]).size().unstack(fill_value=0)
    print(tab.to_string())
    print(f"\nframes cached -> {args.out}/frames/ ; pairs -> {args.out}/pairs.parquet")


if __name__ == "__main__":
    main()
