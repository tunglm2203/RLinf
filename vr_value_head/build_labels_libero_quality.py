"""Failure-inclusive LIBERO adapter — the robometer-style quality-tiered eval set (M4).

Unlike build_labels_libero.py (clean successes, sub-goal-conditioned local progress), this uses
ALL rollouts (success + failure) with:
  - GLOBAL-task conditioning     (single task string per episode; failures have no clean sub-goals)
  - predicate-fraction progress  = satisfied predicates / K over time — GROUNDED, so it naturally
                                   PLATEAUS for failures (success 0->1.0, partial 0->0.5, fail ~0)
  - 3 quality tiers              successful / suboptimal / failure (robometer's tiers)
This makes succ_fail_diff and quality-ranking meaningful, and is CONSISTENT with the robometer
datasets (also global-task). Emits the standard PAIRS schema + `success` + `quality` columns.

    /opt/venv/egoverse/bin/python build_labels_libero_quality.py --stride 1 --neg 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parent
NPZ_DIR = REPO / "research/collected_rollouts/libero_segments_v2"
TASK_MAP = REPO / "research/probes/libero10_task_map.json"


def quality_of(success: bool, max_frac: float) -> str:
    if success:
        return "successful"
    return "suboptimal" if max_frac > 0 else "failure"


def build(out_dir: Path, stride: int, n_neg: int, test_frac: float, seed: int):
    tmap = {int(k): v["language"] for k, v in json.load(open(TASK_MAP)).items()}
    rng = np.random.default_rng(seed)
    frames_out = out_dir / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    files = sorted(NPZ_DIR.glob("*.npz"))
    # episode-level split per task (deterministic)
    by_task = {}
    for f in files:
        by_task.setdefault(int(np.load(f)["task_id"]), []).append(f.stem)
    split = {}
    for t, eps in by_task.items():
        eps = sorted(eps); rng.shuffle(eps)
        n_test = max(1, round(test_frac * len(eps)))
        for i, e in enumerate(eps):
            split[e] = "test" if i < n_test else "train"

    all_tasks = sorted(tmap)
    rows, pair_id = [], 0
    for f in files:
        d = np.load(f)
        eid, tid, success = f.stem, int(d["task_id"]), bool(d["success"])
        frames, P = d["frames"], d["predicates_per_frame"]
        if P.size == 0:
            continue
        K = P.shape[1]
        frac = P.sum(1) / K                                  # grounded progress, plateaus on failure
        quality = quality_of(success, float(frac.max()))
        gtask = tmap.get(tid, f"task_{tid}")
        sp = split.get(eid, "train")

        idx = list(range(0, frames.shape[0], stride))
        np.savez_compressed(frames_out / f"{eid}.npz",
                            frame_idx=np.asarray(idx, dtype=np.int64),
                            img=frames[idx].astype(np.uint8))
        wrong_tasks = [tmap[t] for t in all_tasks if t != tid]
        for fr in idx:
            rows.append(dict(pair_id=pair_id, episode_id=eid, frame=int(fr), seg_idx=0, n_segs=1,
                             local_progress=float(frac[fr]), role="pos", arm="global", text=gtask,
                             split=sp, split_group=eid, global_task=gtask, task_id=tid,
                             success=success, quality=quality))
            pair_id += 1
            for wt in rng.choice(wrong_tasks, size=min(n_neg, len(wrong_tasks)), replace=False):
                rows.append(dict(pair_id=pair_id, episode_id=eid, frame=int(fr), seg_idx=0, n_segs=1,
                                 local_progress=float("nan"), role="neg_subgoal", arm="global",
                                 text=str(wt), split=sp, split_group=eid, global_task=gtask,
                                 task_id=tid, success=success, quality=quality))
                pair_id += 1

    pairs = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "cache_libero_quality")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--neg", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = build(args.out, args.stride, args.neg, args.test_frac, args.seed)
    if pairs.empty:
        raise SystemExit("no pairs produced")
    ep = pairs.drop_duplicates("episode_id")
    print(f"pairs {len(pairs):,} ({int((pairs.role=='pos').sum()):,} pos / "
          f"{int((pairs.role=='neg_subgoal').sum()):,} neg) | episodes {ep.shape[0]}")
    print("\nquality tiers (episodes):")
    print(ep.quality.value_counts().to_string())
    print("\nquality x split (episodes):")
    print(ep.groupby(["quality", "split"]).size().unstack(fill_value=0).to_string())
    print(f"\nframes -> {args.out}/frames/ ; pairs -> {args.out}/pairs.parquet")


if __name__ == "__main__":
    main()
