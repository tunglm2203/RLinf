"""VLM fine-tuning of the progress value head — LoRA on the FROZEN backbone + trainable head.

The frozen-feature pipeline hit a ceiling (ρ~0.3 in-domain, ~0 cross-domain — features aren't
progress-transferable). This unfreezes the backbone via LoRA (robometer's approach), so the
features themselves adapt to progress. Fits on ONE RTX 5090 (32 GB): base stays bf16 (~4.3 GB),
only LoRA adapters + head get optimizer state, gradient checkpointing keeps activations small.

Unlike the two-stage pipeline there is NO feature cache — frames go through the backbone every
step. Reuses a cache's pairs.parquet + frames/*.npz as the (frame, text, target) source.

    /opt/venv/egoverse/bin/python finetune.py --cache cache_libero_quality --epochs 3 --lora-r 16
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import evaluate as ev
import losses
from model import ProgressHead

HERE = Path(__file__).parent
MODEL_PATH = "/mnt/workspace/HUGGINGFACE_HOME/hub/Cosmos-Reason2-2B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build(lora_r, lora_alpha, dropout, n_bins, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()                       # required: grad-ckpt through frozen base
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=dropout, bias="none",
        target_modules=TARGETS, task_type="FEATURE_EXTRACTION"))
    model.to(device)
    head = ProgressHead(input_dim=model.config.text_config.hidden_size if hasattr(model.config, "text_config")
                        else 2048, n_bins=n_bins).to(device)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA trainable: {ntrain/1e6:.1f}M backbone params + {sum(p.numel() for p in head.parameters())/1e6:.2f}M head")
    return proc, model, head


def _chat(proc, prompt):
    return proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True)


def encode(proc, model, imgs, texts, device, pad_side):
    """(images, texts) -> pooled last-real-token feature [B, H], WITH grad."""
    inputs = proc(text=[_chat(proc, t) for t in texts], images=[Image.fromarray(i) for i in imgs],
                  return_tensors="pt", padding=True).to(device)
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[-1]
    attn = inputs["attention_mask"]
    idx = (torch.full((attn.shape[0],), attn.shape[1] - 1, device=device) if pad_side == "left"
           else attn.sum(1) - 1)
    return h[torch.arange(h.shape[0], device=device), idx].float()


class Frames:
    """Lazy per-episode frame loader from a cache's frames/*.npz."""
    def __init__(self, cache): self.dir = Path(cache) / "frames"; self.c = {}
    def img(self, eid, frame):
        if eid not in self.c:
            d = np.load(self.dir / f"{eid}.npz")
            imgs = d["img"]                                   # decompress ONCE (not per-frame!)
            self.c[eid] = {int(f): imgs[i] for i, f in enumerate(d["frame_idx"])}
        return self.c[eid][int(frame)]


def run_eval(proc, model, head, frames, rows, device, pad_side, bs=16):
    model.eval(); head.eval()
    preds = np.zeros(len(rows), np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows.iloc[i:i + bs]
            imgs = [frames.img(e, f) for e, f in zip(r.episode_id, r.frame)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds[i:i + bs] = head(encode(proc, model, imgs, list(r.text), device, pad_side))["local"].float().cpu().numpy()
    rr = rows.copy(); rr["pred"] = preds
    pos = rr[rr.role == "pos"].rename(columns={"local_progress": "target"})
    from scipy.stats import pearsonr, spearmanr
    out = {"rho": ev._per_seg(pos, "pred", "target", lambda a, b: spearmanr(a, b).correlation),
           "voc": ev._per_seg(pos, "pred", "target", lambda a, b: pearsonr(a, b)[0] if np.std(a) and np.std(b) else np.nan),
           "tau_a": ev._per_seg(pos, "pred", "target", ev.kendall_tau_a)}
    neg = rr[rr.role == "neg_subgoal"]
    if len(neg):
        j = neg.merge(pos[["episode_id", "frame", "seg_idx", "target"]], on=["episode_id", "frame", "seg_idx"])
        out["lang"] = out["rho"] - ev._per_seg(j, "pred", "target", lambda a, b: spearmanr(a, b).correlation)
    out.update(ev.success_failure_diff(rr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=HERE / "cache_libero_quality")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4, help="images per forward (grad-ckpt keeps this small)")
    ap.add_argument("--accum", type=int, default=4, help="grad accumulation -> effective batch batch*accum")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "finetune")
    args = ap.parse_args()

    pairs = pd.read_parquet(args.cache / "pairs.parquet")
    tr = pairs[(pairs.split == "train") & (pairs.role == "pos")].reset_index(drop=True)  # neg masked in CE
    te = pairs[pairs.split == "test"].reset_index(drop=True)
    frames = Frames(args.cache)
    proc, model, head = build(args.lora_r, args.lora_alpha, 0.05, args.n_bins, args.device)
    pad_side = getattr(getattr(proc, "tokenizer", None), "padding_side", "right")
    print(f"cache={args.cache.name} | train pairs={len(tr)} | test pairs={len(te)} | pad={pad_side}")

    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(tr) / (args.batch * args.accum)) * args.epochs
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)

    print(f"\nBEFORE fine-tuning (zero LoRA):"); print("  ", run_eval(proc, model, head, frames, te, args.device, pad_side))
    for ep in range(args.epochs):
        model.train(); head.train()
        order = np.random.default_rng(ep).permutation(len(tr))
        opt.zero_grad(); acc = 0
        for bi in range(0, len(order), args.batch):
            r = tr.iloc[order[bi:bi + args.batch]]
            imgs = [frames.img(e, f) for e, f in zip(r.episode_id, r.frame)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = encode(proc, model, imgs, list(r.text), args.device, pad_side)
                preds = head(feats)
                tgt = torch.tensor(np.nan_to_num(r.local_progress.to_numpy(np.float32)), device=args.device)
                pos = torch.tensor((r.role == "pos").to_numpy(), device=args.device)
                loss = losses.discrete_ce(preds["local_logits"], tgt, pos) / args.accum
            loss.backward(); acc += 1
            if acc == args.accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(); acc = 0
            if bi % (args.batch * 50) == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  ep{ep} step{bi//args.batch} loss={loss.item()*args.accum:.3f} peakmem={mem:.1f}GB", flush=True)
        m = run_eval(proc, model, head, frames, te, args.device, pad_side)
        print(f"ep{ep} TEST:", {k: round(v, 3) for k, v in m.items() if v == v})

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out / "lora"); torch.save(head.state_dict(), args.out / "head.pt")
    print(f"\nsaved LoRA + head -> {args.out}")


if __name__ == "__main__":
    main()
