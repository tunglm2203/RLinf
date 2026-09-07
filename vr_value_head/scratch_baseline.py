"""'Without ego-pretraining' control: frozen backbone, head trained FROM SCRATCH on each transfer
eval set (P:none -> D:eval(frozen) -> E:eval). Direct counterpart to configs 1 & 3 (which use the
ego-pretrained head). Both backbones, frozen cached features, CPU-only.

  ours_libero  : uses the cache's own train/test split (same test set as configs 1-4)
  rbm_*        : per-task 70/30 episode split (no native split); eval on the 30% held-out
"""
from pathlib import Path
import numpy as np
import torch

from dataset import PairFeatures, SplitView
from model import ProgressHead
from evaluate import report, fmt7, make_signature
from transfer import train_head

HERE = Path(__file__).parent
CACHES = {"Cosmos": ("cache_libero", "cache_rbm"),
          "Qwen3VL": ("cache_libero_qwenbase", "cache_rbm_qwenbase")}
SETS = [  # (label, kind, dataset_filter, cond, target)
    ("ours_libero", "libero", None, "subgoal", "time_seg"),
    ("rbm_libero", "rbm", "abraranwar_libero_rfm_libero256_10", "global", "t/T"),
    ("rbm_metaworld", "rbm", "aliangdw_metaworld_metaworld_eval", "global", "t/T"),
]
SEED, EPOCHS, LR, BATCH, NBINS = 0, 150, 1e-3, 16, 10


def split_rbm(data, ds, test_frac=0.3, seed=SEED):
    rows = data.pairs[data.pairs.dataset == ds]
    rng = np.random.default_rng(seed)
    tr_eps = []
    for _, g in rows.groupby("global_task"):
        eps = sorted(g.episode_id.unique()); rng.shuffle(eps)
        tr_eps += list(eps[max(1, round(test_frac * len(eps))):])
    tr = rows[rows.episode_id.isin(tr_eps)].reset_index(drop=True)
    te = rows[~rows.episode_id.isin(tr_eps)].reset_index(drop=True)
    return tr, te


for backbone, (libero_cache, rbm_cache) in CACHES.items():
    for label, kind, filt, cond, target in SETS:
        data = PairFeatures(HERE / (libero_cache if kind == "libero" else rbm_cache), pool="last_last")
        if kind == "libero":
            tr = data.pairs[data.pairs.split == "train"].reset_index(drop=True)
            te = data.pairs[data.pairs.split == "test"].reset_index(drop=True)
        else:
            tr, te = split_rbm(data, filt)
        torch.manual_seed(SEED)
        head = train_head(ProgressHead(input_dim=data.dim, n_bins=NBINS),
                          SplitView(data, tr), EPOCHS, LR, BATCH, "cpu")
        sig = f"[{backbone}] " + make_signature(
            pretrain="none", downstream=label, eval_ds=label, cond=cond, target=target, vlm_ft="frozen")
        print(fmt7(report(head, SplitView(data, te), signature=sig)), flush=True)
        print(flush=True)
