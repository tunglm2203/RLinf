"""Adapter: robometer processed datasets (MetaWorld / LIBERO / real-robot OOD) -> our PAIRS format.

robometer tar layout (per dataset):
  <name>/frames/trajectory_<uuid>.npz        key 'frames' [N,H,W,3] uint8   (we RE-ENCODE with our backbone)
  <name>/index_mappings.json                 task_indices{task_str:[idx]}, quality_indices{tier:[idx]}, ...
  <name>/embeddings/trajectory_<idx>_<uuid>_embeddings.pt   (robometer's own features — IGNORED; used only
                                                             for the idx<->uuid map via the FILENAME)

Global-task conditioning (these are single-task episodes), progress target = t/T (frame position),
3 quality tiers (successful / suboptimal / failure) carried as `quality` + `success`. Wrong-task
negatives (a different task's string, same dataset) for language sensitivity. All rows marked
split='test' (EVAL sets — evaluate a head trained elsewhere). Emits the standard PAIRS schema
(+ success, quality, dataset), byte-compatible with extract_features.py / dataset.py.

    /opt/venv/egoverse/bin/python build_labels_rbm.py --root /mnt/workspace2/all_dataset/robometer_dataset \
        --include metaworld usc_ mit_franka so101 --out cache_rbm
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
_EMB = re.compile(r"trajectory_(\d+)_(.+)_embeddings\.pt$")   # id may be a hex uuid OR a descriptive name


def ensure_extracted(root: Path, name: str) -> Path:
    """Untar <name>.tar into root/<name>/ if not already extracted."""
    d = root / name
    if d.exists():
        return d
    tar = root / f"{name}.tar"
    with tarfile.open(tar) as t:
        t.extractall(root)
    return d


def load_dataset(dpath: Path):
    """Yield (uuid, task_str, quality, frames_npz_path) per trajectory — LAZY (no frame load here,
    so the caller can cap/subsample before paying to decompress frames)."""
    mp = json.loads((dpath / "index_mappings.json").read_text())
    idx_task = {i: t for t, idxs in mp.get("task_indices", {}).items() for i in idxs}
    idx_qual = {i: q for q, idxs in mp.get("quality_indices", {}).items() for i in idxs}
    idx_uuid = {}
    for p in (dpath / "embeddings").glob("*.pt"):
        m = _EMB.search(p.name)
        if m:
            idx_uuid[int(m.group(1))] = m.group(2)
    for i, uuid in sorted(idx_uuid.items()):
        fnpz = dpath / "frames" / f"trajectory_{uuid}.npz"
        if not fnpz.exists() or i not in idx_task:
            continue
        yield uuid, idx_task[i], idx_qual.get(i, "successful"), fnpz


def build(root: Path, includes, out_dir: Path, stride: int, n_neg: int, max_frames: int, seed: int,
          names=None, cap: int = 0, linspace: bool = False):
    rng = np.random.default_rng(seed)
    # datasets are available either as <name>.tar OR as a pre-extracted <name>/ dir (processed_datasets)
    avail = {t.stem for t in root.glob("*.tar")}
    avail |= {d.name for d in root.iterdir() if d.is_dir() and (d / "index_mappings.json").exists()}
    if names:                                             # exact-name selection (e.g. the eval groups)
        chosen = [n for n in names if n in avail]
        missing = [n for n in names if n not in avail]
        if missing:
            print(f"WARNING: {len(missing)} requested dataset(s) not found under {root}:")
            for m in missing:
                print(f"  - {m}")
    else:                                                 # legacy substring selection
        chosen = sorted(n for n in avail if any(inc in n for inc in includes))
    frames_out = out_dir / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    rows, pair_id = [], 0
    for name in chosen:
        d = ensure_extracted(root, name)
        trajs = list(load_dataset(d))
        if cap and len(trajs) > cap:                      # per-dataset episode cap (random subsample)
            trajs = [trajs[i] for i in sorted(rng.choice(len(trajs), size=cap, replace=False))]
        tasks_here = sorted({t for _, t, _, _ in trajs})
        for uuid, task, quality, fnpz in trajs:
            frames = np.load(fnpz)["frames"]
            N = frames.shape[0]
            if linspace and N > max_frames:                  # robometer's eval subsampling (max_frames=8)
                idx = sorted({int(i) for i in np.linspace(0, N - 1, max_frames).round().astype(int)})
            else:
                idx = list(range(0, N, stride))[:max_frames]
            if len(idx) < 2:
                continue
            eid = f"{name}__{uuid}"
            np.savez_compressed(frames_out / f"{eid}.npz",
                                frame_idx=np.asarray(idx, dtype=np.int64),
                                img=frames[idx].astype(np.uint8))
            success = quality == "successful"
            wrong = [t for t in tasks_here if t != task]
            for f in idx:
                rows.append(dict(pair_id=pair_id, episode_id=eid, frame=int(f), seg_idx=0, n_segs=1,
                                 local_progress=float(f / max(1, N - 1)), role="pos", arm="global",
                                 text=task, split="test", split_group=eid, global_task=task,
                                 task_id=-1, success=success, quality=quality, dataset=name))
                pair_id += 1
                for wt in (rng.choice(wrong, size=min(n_neg, len(wrong)), replace=False) if wrong else []):
                    rows.append(dict(pair_id=pair_id, episode_id=eid, frame=int(f), seg_idx=0, n_segs=1,
                                     local_progress=float("nan"), role="neg_subgoal", arm="global",
                                     text=str(wt), split="test", split_group=eid, global_task=task,
                                     task_id=-1, success=success, quality=quality, dataset=name))
                    pair_id += 1

    pairs = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/workspace2/all_dataset/robometer_dataset"))
    ap.add_argument("--include", nargs="+", default=["metaworld", "usc_", "mit_franka", "so101", "libero256"])
    ap.add_argument("--groups", action="store_true",
                    help="select the 18 datasets of the 3 standard eval groups (rbm_eval_groups.ALL_DATASETS)")
    ap.add_argument("--out", type=Path, default=HERE / "cache_rbm")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--neg", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--linspace", action="store_true",
                    help="evenly subsample each trajectory to --max-frames (robometer eval uses linspace to 8)")
    ap.add_argument("--cap", type=int, default=0, help="max episodes per dataset (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = None
    if args.groups:
        from rbm_eval_groups import ALL_DATASETS
        names = ALL_DATASETS
    pairs = build(args.root, args.include, args.out, args.stride, args.neg, args.max_frames, args.seed,
                  names=names, cap=args.cap, linspace=args.linspace)
    if pairs.empty:
        raise SystemExit("no pairs produced (check --root / --include patterns)")
    ep = pairs.drop_duplicates("episode_id")
    print(f"pairs {len(pairs):,} ({int((pairs.role=='pos').sum()):,} pos / "
          f"{int((pairs.role=='neg_subgoal').sum()):,} neg) | episodes {len(ep)}")
    print("\nepisodes x quality per dataset:")
    print(ep.groupby(["dataset", "quality"]).size().unstack(fill_value=0).to_string())
    print(f"\nframes -> {args.out}/frames/ ; pairs -> {args.out}/pairs.parquet")


if __name__ == "__main__":
    main()
