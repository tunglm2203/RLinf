"""Trio eval for the LoRA-adapted configs (2 = base+LoRA, 4 = Cosmos+LoRA).

Unlike the frozen path there is no feature cache — the eval frames go through the LoRA-adapted VLM
on the fly (reuses finetune.encode + Frames), then the SAME report_metrics() as the frozen configs
(success-only correlations, N/A quality trio). One GPU; run once per config.

    CUDA_VISIBLE_DEVICES=0 python eval_lora.py --config 2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import ProgressHead
from finetune import encode, Frames
from evaluate import report_metrics, fmt7, make_signature

HERE = Path(__file__).parent
COSMOS = "/mnt/workspace/HUGGINGFACE_HOME/hub/Cosmos-Reason2-2B"
QWEN = "/mnt/workspace/HUGGINGFACE_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"

CONFIGS = {
    "2": dict(backbone="Qwen3VL", model=QWEN, run="runs/lora_ego_qwenbase"),
    "4": dict(backbone="Cosmos", model=COSMOS, run="runs/lora_ego_cosmos"),
}
# (label, frames_cache, pairs_cache, (filter_col, filter_val), cond, target)
EVALSETS = [
    ("ours_libero", "cache_libero", "cache_libero", ("split", "test"), "subgoal", "time_seg"),
    ("rbm_libero", "cache_rbm", "cache_rbm", ("dataset", "abraranwar_libero_rfm_libero256_10"), "global", "t/T"),
    ("rbm_metaworld", "cache_rbm", "cache_rbm", ("dataset", "aliangdw_metaworld_metaworld_eval"), "global", "t/T"),
]


def load(model_path, run, n_bins, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(HERE / run / "lora")).to(device).eval()
    dim = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else 2048
    head = ProgressHead(input_dim=dim, n_bins=n_bins).to(device)
    sd = torch.load(HERE / run / "head.pt", map_location=device)
    head.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
    head.eval()
    pad = getattr(getattr(proc, "tokenizer", None), "padding_side", "right")
    return proc, model, head, pad


@torch.no_grad()
def predict_rows(proc, model, head, frames, rows, device, pad, bs=16):
    preds = np.zeros(len(rows), np.float32)
    for i in range(0, len(rows), bs):
        r = rows.iloc[i:i + bs]
        imgs = [frames.img(e, f) for e, f in zip(r.episode_id, r.frame)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds[i:i + bs] = head(encode(proc, model, imgs, list(r.text), device, pad))["local"].float().cpu().numpy()
    rr = rows.copy(); rr["pred"] = preds
    return rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    c = CONFIGS[args.config]
    proc, model, head, pad = load(c["model"], c["run"], args.n_bins, args.device)
    fcache = {}
    for label, fc, pc, (col, val), cond, target in EVALSETS:
        pairs = pd.read_parquet(HERE / pc / "pairs.parquet")
        rows = pairs[pairs[col] == val].reset_index(drop=True)
        frames = fcache.setdefault(fc, Frames(HERE / fc))
        rr = predict_rows(proc, model, head, frames, rows, args.device, pad)
        sig = f"[{c['backbone']}] " + make_signature(
            pretrain="ego", downstream="none", eval_ds=label, cond=cond, target=target, vlm_pre="LoRA")
        print(fmt7(report_metrics(rr, sig)), flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
