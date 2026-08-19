"""Frozen configs 1 & 3, zero-shot on the robometer trio members (global · t/T).

  RBM-LIBERO   = abraranwar_libero_rfm_libero256_10  (success-only -> trio N/A)
  RBM-MetaWorld= aliangdw_metaworld_metaworld_eval    (2 quality tiers -> full 7)
Cosmos head reads Cosmos features (cache_rbm); base head reads base features (cache_rbm_qwenbase).
"""
from pathlib import Path
import torch
from dataset import PairFeatures, SplitView
from model import ProgressHead
from evaluate import report, fmt7, make_signature

HERE = Path(__file__).parent
HEADS = {"Cosmos": ("cache_rbm", "runs/vh_baseline_500_ce10/head.pt"),
         "Qwen3VL": ("cache_rbm_qwenbase", "runs/vh_ego_qwenbase/head.pt")}
DS = {"rbm_libero": "abraranwar_libero_rfm_libero256_10",
      "rbm_metaworld": "aliangdw_metaworld_metaworld_eval"}


def load(p, dim):
    h = ProgressHead(input_dim=dim, n_bins=10)
    h.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["state_dict"])
    return h.eval()


# load each backbone's cache once, reuse across datasets
cache_data = {bk: PairFeatures(HERE / c, pool="last_last") for bk, (c, _) in HEADS.items()}
for eds, ds_name in DS.items():
    print(f"--- E:{eds} ({ds_name}) " + "-" * 60)
    for backbone, (_, head_path) in HEADS.items():
        data = cache_data[backbone]
        h = load(HERE / head_path, data.dim)
        rows = data.pairs[data.pairs.dataset == ds_name].reset_index(drop=True)
        view = SplitView(data, rows)
        sig = f"[{backbone}] " + make_signature(
            pretrain="ego", downstream="none", eval_ds=eds,
            cond="global", target="t/T", vlm_pre="frozen")
        print(fmt7(report(h, view, signature=sig)))
    print()
