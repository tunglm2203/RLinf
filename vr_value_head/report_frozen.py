"""Frozen configs 1 & 3 — canonical fmt7 across eval sets (zero-shot, D:none).

Both heads are ego-pretrained on frozen features with the locked recipe; the only
difference is the backbone the features came from:
  config 3 = Cosmos-Reason2-2B   -> runs/vh_baseline_500_ce10
  config 1 = Qwen3-VL-2B base    -> runs/vh_ego_qwenbase
Each backbone reads its OWN feature cache (Cosmos features for config 3, base for config 1).
"""
import sys
from pathlib import Path
import torch
from dataset import PairFeatures
from model import ProgressHead
from evaluate import report, fmt7, make_signature

HERE = Path(__file__).parent
HEADS = {"Cosmos": "runs/vh_baseline_500_ce10/head.pt",
         "Qwen3VL": "runs/vh_ego_qwenbase/head.pt"}
SUF = {"Cosmos": "", "Qwen3VL": "_qwenbase"}          # cache suffix per backbone

# eval sets: label -> (cache stem, split)
EVAL = {"ego": ("cache", "test"),
        "ours_libero": ("cache_libero", "test")}
# restrict to a subset of eval sets if given on the CLI (e.g. `python report_frozen.py ours_libero`)
want = sys.argv[1:] or list(EVAL)


def load(p, dim):
    h = ProgressHead(input_dim=dim, n_bins=10)
    h.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["state_dict"])
    return h.eval()


for eds in want:
    stem, split = EVAL[eds]
    print(f"--- E:{eds} " + "-" * 80)
    for backbone, head_path in HEADS.items():
        d = PairFeatures(HERE / f"{stem}{SUF[backbone]}", pool="last_last")
        h = load(HERE / head_path, d.dim)
        sig = f"[{backbone}] " + make_signature(
            pretrain="ego", downstream="none", eval_ds=eds,
            cond="subgoal", target="time_seg", vlm_pre="frozen")
        print(fmt7(report(h, d.split(split), signature=sig)))
    print()
