"""Join the PAIRS table with the feature cache → training batches.

Loads all features for a split into memory (a few thousand × 2048 floats — trivial), then
serves SEGMENT-BLOCK minibatches: each batch is the frames of a handful of whole segments, so
the within-segment pairwise-ranking loss has real pairs to work with. A frame-shuffled batch
would mostly contain singletons and starve that loss.

Time-warp augmentation lives here (not in losses): per epoch, a segment's positive frames get
their local targets recomputed as rank-within-a-random-monotone-subset. Elapsed-frame index
then no longer maps linearly to progress, so a clock cannot fit while a vision head is
unaffected — the anti-clock control. Toggle with `time_warp`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


class PairFeatures:
    def __init__(self, cache: Path, pool: str = "mean_last"):
        self.pairs = pd.read_parquet(cache / "pairs.parquet")
        feat = {}
        for f in sorted((cache / "features").glob("*.npz")):
            d = np.load(f)
            for pid, vec in zip(d["pair_id"], d[pool]):
                feat[int(pid)] = vec
        have = self.pairs.pair_id.isin(feat.keys())
        if not have.all():
            self.pairs = self.pairs[have].reset_index(drop=True)   # only encoded rows
        self.feat = feat
        self.dim = len(next(iter(feat.values()))) if feat else 0
        # No external StandardScaler: the ProgressHead's in-head LayerNorm (robometer recipe)
        # renormalizes after the first projection, so raw features don't saturate the sigmoid.

    def split(self, split: str) -> "SplitView":
        return SplitView(self, self.pairs[self.pairs.split == split].reset_index(drop=True))


class SplitView:
    def __init__(self, parent: PairFeatures, rows: pd.DataFrame):
        self.parent, self.rows = parent, rows
        # a stable global segment id for grouping (episode + seg_idx)
        self.rows = self.rows.assign(
            gid=(self.rows.episode_id.astype(str) + "#" + self.rows.seg_idx.astype(str)))
        self.seg_ids = self.rows[self.rows.role == "pos"].gid.unique()

    def _tensor(self, sub: pd.DataFrame, time_warp: bool, rng) -> dict:
        feats = np.stack([self.parent.feat[p] for p in sub.pair_id])
        target = sub.local_progress.to_numpy(dtype=np.float32).copy()
        pos = (sub.role == "pos").to_numpy()
        if time_warp:                                   # relabel pos targets by warped rank
            for gid in sub.gid.unique():
                m = (sub.gid.to_numpy() == gid) & pos
                order = np.where(m)[0]
                if len(order) >= 2:
                    keep = np.sort(rng.choice(len(order), size=max(2, int(len(order) *
                                   rng.uniform(0.5, 1.0))), replace=False))
                    ranks = np.searchsorted(keep, np.arange(len(order)), side="right") - 1
                    target[order] = np.clip(ranks / (len(keep) - 1), 0, 1)
        codes = pd.factorize(sub.gid)[0]
        return {
            "feats": torch.from_numpy(feats).float(),
            "target_local": torch.from_numpy(np.nan_to_num(target)).float(),
            "group_id": torch.from_numpy(codes).long(),
            "pos_mask": torch.from_numpy(pos),
            "neg_mask": torch.from_numpy((sub.role == "neg_subgoal").to_numpy()),
            "seg_idx": torch.from_numpy(sub.seg_idx.to_numpy()).long(),
            "n_segs": torch.from_numpy(sub.n_segs.to_numpy()).long(),
        }

    def batches(self, segs_per_batch: int = 16, time_warp: bool = False, shuffle: bool = True,
                seed: int = 0):
        rng = np.random.default_rng(seed)
        order = self.seg_ids.copy()
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), segs_per_batch):
            gids = set(order[i:i + segs_per_batch])
            sub = self.rows[self.rows.gid.isin(gids)].reset_index(drop=True)
            if len(sub):
                yield self._tensor(sub, time_warp, rng)

    def full(self) -> dict:
        return self._tensor(self.rows, time_warp=False, rng=np.random.default_rng(0))
