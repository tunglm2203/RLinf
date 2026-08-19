"""Config 1 vs 3 — in-domain (E:ego) Cosmos-recipe isolation.

Both heads: frozen VLM, ego-pretrained, locked recipe (last_last · 10-bin CE · 150ep).
The ONLY difference is the backbone the features came from:
  config 3 = Cosmos-Reason2-2B   (cache)           runs/vh_baseline_500_ce10
  config 1 = Qwen3-VL-2B base    (cache_qwenbase)  runs/vh_ego_qwenbase
"""
from pathlib import Path
import torch
from dataset import PairFeatures
from model import ProgressHead
from evaluate import report, fmt7, make_signature

HERE = Path(__file__).parent


def load(p, dim):
    h = ProgressHead(input_dim=dim, n_bins=10)
    h.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["state_dict"])
    return h.eval()


def line(backbone, cache, head_path):
    d = PairFeatures(HERE / cache, pool="last_last")
    h = load(HERE / head_path, d.dim)
    sig = f"[{backbone}] " + make_signature(
        pretrain="ego", downstream="none", eval_ds="ego",
        cond="subgoal", target="time_seg", vlm_pre="frozen")
    return fmt7(report(h, d.split("test"), signature=sig))


print("=" * 96)
print("CONFIG 3  (Cosmos backbone)")
print(line("Cosmos", "cache", "runs/vh_baseline_500_ce10/head.pt"))
print()
print("CONFIG 1  (Qwen3-VL-2B base backbone)")
print(line("Qwen3VL", "cache_qwenbase", "runs/vh_ego_qwenbase/head.pt"))
print("=" * 96)
