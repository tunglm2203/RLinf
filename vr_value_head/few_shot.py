"""M12 — few-shot ego-init curve (the gate). Does an ego-pretrained head, fine-tuned on k target
episodes, beat a scratch head on the same k? Frozen features, both backbones, transfer trio, CPU.

  arm A (scratch)  : fresh head trained on the k-episode subsample
  arm B (ego-init) : ego-pretrained head, fine-tuned on the SAME subsample
  (arm C zero-shot ego, k=0, already in the trio table.)
k = episodes PER TASK. Eval on the set's held-out test split (LIBERO native; rbm per-task 70/30).
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import PairFeatures, SplitView
from model import ProgressHead
from evaluate import report
from transfer import train_head

HERE = Path(__file__).parent
BACKBONES = {
    "Cosmos": dict(libero="cache_libero", rbm="cache_rbm", ego="runs/vh_baseline_500_ce10/head.pt"),
    "Qwen3VL": dict(libero="cache_libero_qwenbase", rbm="cache_rbm_qwenbase", ego="runs/vh_ego_qwenbase/head.pt"),
}
SETS = [("ours_libero", "libero", None),
        ("rbm_libero", "rbm", "abraranwar_libero_rfm_libero256_10"),
        ("rbm_metaworld", "rbm", "aliangdw_metaworld_metaworld_eval")]
KS, SEED, EPOCHS, LR, BATCH, NBINS = [1, 2, 5, 10], 0, 150, 1e-3, 16, 10


def subsample(pool, k, seed):
    rng = np.random.default_rng(seed); chosen = []
    for _, g in pool.groupby("global_task"):
        eps = sorted(g.episode_id.unique()); rng.shuffle(eps); chosen += list(eps[:k])
    return pool[pool.episode_id.isin(chosen)]


def split(data, kind, filt):
    if kind == "libero":
        tr, te = data.pairs[data.pairs.split == "train"], data.pairs[data.pairs.split == "test"]
    else:
        rows = data.pairs[data.pairs.dataset == filt]
        rng = np.random.default_rng(SEED); tr_eps = []
        for _, g in rows.groupby("global_task"):
            eps = sorted(g.episode_id.unique()); rng.shuffle(eps)
            tr_eps += list(eps[max(1, round(0.3 * len(eps))):])
        tr, te = rows[rows.episode_id.isin(tr_eps)], rows[~rows.episode_id.isin(tr_eps)]
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def rho(head, data, te):
    return report(head, SplitView(data, te)).get("spearman_head")


out = []
for bk, cfg in BACKBONES.items():
    caches = {"libero": PairFeatures(HERE / cfg["libero"], pool="last_last"),
              "rbm": PairFeatures(HERE / cfg["rbm"], pool="last_last")}
    for label, kind, filt in SETS:
        data = caches[kind]
        ego = ProgressHead(input_dim=data.dim, n_bins=NBINS)
        ego.load_state_dict(torch.load(HERE / cfg["ego"], map_location="cpu", weights_only=False)["state_dict"])
        trpool, te = split(data, kind, filt)
        for k in KS:
            sub = subsample(trpool, k, SEED).reset_index(drop=True)
            neps = sub.episode_id.nunique()
            torch.manual_seed(SEED)
            hA = train_head(ProgressHead(input_dim=data.dim, n_bins=NBINS), SplitView(data, sub), EPOCHS, LR, BATCH, "cpu")
            hB = copy.deepcopy(ego); train_head(hB, SplitView(data, sub), EPOCHS, LR, BATCH, "cpu")
            rA, rB = rho(hA, data, te), rho(hB, data, te)
            out.append(dict(backbone=bk, E=label, k=k, neps=int(neps),
                            scratch=round(rA, 3), ego_init=round(rB, 3), gain=round(rB - rA, 3)))
            print(f"  [{bk:7s}] {label:14s} k={k:2d} ({neps:3d}eps)  scratch ρ={rA:+.3f}  ego-init ρ={rB:+.3f}  (Δ={rB - rA:+.3f})", flush=True)

df = pd.DataFrame(out)
(HERE / "runs").mkdir(exist_ok=True)
df.to_csv(HERE / "runs" / "m12_few_shot.csv", index=False)
print(f"\nsaved runs/m12_few_shot.csv ({len(df)} rows)")
