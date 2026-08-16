"""Preprocess: original egoverse → a PAIRS table + a sampled-frame cache.

Runs in the **egoverse venv** (needs `zarr`, via probes/egoverse_data). It does NOT touch the
backbone — it only reads frames and writes:

  <out>/frames/<episode_id>.npz      sampled RGB frames for this episode  {frame_idx[], img[N,H,W,3]}
  <out>/pairs.parquet                one row per (frame, text) to encode later

The PAIRS table is the contract that keeps the whole head two-stage (see README). Columns:
  pair_id  episode_id  frame  seg_idx(=k)  n_segs(=K, eval-only)  local_progress
  role {pos|neg_subgoal}  arm  text  split  split_group  global_task
extract_features.py (backbone, same egoverse venv) consumes pairs+frames → one feature vector per
row; dataset.py joins them back. Paraphrase rows are omitted (egoverse has no rephrasings).

    /opt/venv/egoverse/bin/python build_labels.py --episodes 200 --stride 10 --arm subgoal --neg 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "probes"))
from egoverse_data import list_episodes, load_episode, DEFAULT_ROOT, split_of  # noqa: E402

import text_arms  # noqa: E402

HERE = Path(__file__).parent


def build(root, out_dir, n_episodes, stride, arm, n_neg, min_seg_frames, seed):
    rng = np.random.default_rng(seed)
    paths = list_episodes(root)[:n_episodes]
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows, pair_id = [], 0
    for path in paths:
        ep = load_episode(path)
        segs = [s for s in ep.segments if s.n_frames >= min_seg_frames]
        if len(segs) < 1:
            continue
        split = split_of(ep.episode_id)
        # sample frames per segment; `k` is the position in the FILTERED list (short segments
        # were dropped), NOT s.idx (the original episode index) — mixing the two overruns segs.
        keep_frames, seg_of_frame, k_of_seg = [], {}, {id(s): k for k, s in enumerate(segs)}
        for s in segs:
            for f in range(s.start, s.end, stride):
                keep_frames.append(f)
                seg_of_frame[f] = s
        if len(keep_frames) < 2:
            continue
        keep_frames = sorted(set(keep_frames))

        # frame extraction is the expensive part — skip if this episode's npz already exists
        # (resumable; a re-run reuses the cache instead of re-decoding video)
        npz_path = frames_dir / f"{ep.episode_id}.npz"
        if not npz_path.exists():
            imgs = np.stack([ep.read_frame(f) for f in keep_frames]).astype(np.uint8)
            np.savez_compressed(npz_path,
                                frame_idx=np.asarray(keep_frames, dtype=np.int64), img=imgs)

        all_texts = [s.text for s in segs]
        for f in keep_frames:
            s = seg_of_frame[f]
            k = k_of_seg[id(s)]                                 # position in filtered list
            history = [segs[i].text for i in range(k)]          # grounded: prior segments
            text = text_arms.compose(arm, s.text, ep.global_task, history)
            rows.append(dict(pair_id=pair_id, episode_id=ep.episode_id, frame=int(f),
                             seg_idx=k, n_segs=len(segs),
                             local_progress=float(s.local_progress(f)),
                             role="pos", arm=arm, text=text, split=split,
                             split_group=ep.episode_id, global_task=ep.global_task))
            pair_id += 1
            # hard negatives: same frame, a DIFFERENT segment's sub-goal (same episode)
            others = [t for t in all_texts if t != s.text]
            for wt in rng.choice(others, size=min(n_neg, len(others)), replace=False) if others else []:
                neg_text = text_arms.compose(arm, wt, ep.global_task, history)
                rows.append(dict(pair_id=pair_id, episode_id=ep.episode_id, frame=int(f),
                                 seg_idx=k, n_segs=len(segs),
                                 local_progress=float("nan"), role="neg_subgoal", arm=arm,
                                 text=neg_text, split=split, split_group=ep.episode_id,
                                 global_task=ep.global_task))
                pair_id += 1

    pairs = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=HERE / "cache")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--arm", choices=["subgoal", "full"], default="subgoal")
    ap.add_argument("--neg", type=int, default=1, help="wrong-sub-goal negatives per positive frame")
    ap.add_argument("--min-seg-frames", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = build(args.root, args.out, args.episodes, args.stride, args.arm,
                  args.neg, args.min_seg_frames, args.seed)
    if pairs.empty:
        raise SystemExit("no pairs produced")
    print(f"pairs {len(pairs):,}  ({int((pairs.role=='pos').sum()):,} pos / "
          f"{int((pairs.role=='neg_subgoal').sum()):,} neg)")
    print(f"episodes {pairs.episode_id.nunique()} | arm={args.arm} | "
          f"splits {pairs.groupby('split').size().to_dict()}")
    print(f"frames cached under {args.out}/frames/ ; pairs -> {args.out}/pairs.parquet")
    print("\nexample rows:")
    print(pairs[["role", "seg_idx", "local_progress", "text"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
