"""Stage 2: frozen N1.7 backbone → one pooled feature vector per PAIRS row.

Runs in the **egoverse venv** (which already has torch + Qwen3-VL). Reads build_labels' output;
the two stages stay decoupled by a materialization boundary (the frame npz) so extraction is
cacheable, exactly like the LIBERO pipeline.

For each pairs row it encodes (frame pixels, composed text) through Cosmos-Reason2-2B and pools
the last hidden state. All three pooling variants are cached so the head can sweep them without
re-extraction:  mean_last · last_last · mean_mid.

    /opt/venv/egoverse/bin/python extract_features.py --cache cache            # after build_labels
Output: <cache>/features/<episode_id>.npz  {pair_id[], mean_last[N,D], last_last[N,D], mean_mid[N,D]}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL_PATH = "/mnt/workspace/HUGGINGFACE_HOME/hub/Cosmos-Reason2-2B"
HERE = Path(__file__).parent


def build_model(device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16).to(device).eval()
    return proc, model


def _chat(proc, prompt):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _last_real_index(attn_mask: torch.Tensor, padding_side: str) -> torch.Tensor:
    """Index of the last REAL (non-pad) token per row. Robust to padding side — the naive
    `attention_mask.sum(1)-1` is only correct for right padding and silently lands on a pad
    otherwise (the bug that cost a 0.40-vs-0.99 eval gap on N1.7's last-token pooling)."""
    if padding_side == "left":
        return torch.full((attn_mask.shape[0],), attn_mask.shape[1] - 1, device=attn_mask.device)
    return attn_mask.sum(1) - 1


@torch.no_grad()
def encode(proc, model, images, prompts, device, mid_layer, padding_side):
    texts = [_chat(proc, p) for p in prompts]
    inputs = proc(text=texts, images=[Image.fromarray(im) for im in images],
                  return_tensors="pt", padding=True).to(device)
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    attn = inputs["attention_mask"]
    mask = attn.unsqueeze(-1).to(torch.float32)

    def mean_pool(h):
        h = h.float() * mask
        return (h.sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()

    last, mid = out.hidden_states[-1], out.hidden_states[mid_layer]
    idx = _last_real_index(attn, padding_side)
    last_tok = last[torch.arange(last.shape[0], device=last.device), idx].float().cpu().numpy()
    return {"mean_last": mean_pool(last), "last_last": last_tok, "mean_mid": mean_pool(mid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=HERE / "cache")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mid-layer", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pairs = pd.read_parquet(args.cache / "pairs.parquet")
    feat_dir = args.cache / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    proc, model = build_model(args.device)
    padding_side = getattr(getattr(proc, "tokenizer", None), "padding_side", "right")
    print(f"padding_side={padding_side} | {len(pairs):,} pairs | "
          f"{pairs.episode_id.nunique()} episodes")

    for eid, g in pairs.groupby("episode_id"):
        out_path = feat_dir / f"{eid}.npz"
        if out_path.exists():
            continue
        fr = np.load(args.cache / "frames" / f"{eid}.npz")
        # decompress the image array ONCE — `fr["img"]` on an NpzFile re-reads and
        # re-decompresses the whole array on every access, so referencing it inside the
        # comprehension would decompress it once per frame (hundreds of times per episode).
        imgs_arr, frame_idx = fr["img"], fr["frame_idx"]
        frame_to_img = {int(f): imgs_arr[i] for i, f in enumerate(frame_idx)}
        acc = {k: [] for k in ("mean_last", "last_last", "mean_mid")}
        pids = g.pair_id.to_numpy()
        rows = g.to_dict("records")
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i:i + args.batch_size]
            imgs = [frame_to_img[r["frame"]] for r in chunk]
            enc = encode(proc, model, imgs, [r["text"] for r in chunk],
                         args.device, args.mid_layer, padding_side)
            for k in acc:
                acc[k].append(enc[k])
        np.savez_compressed(out_path, pair_id=pids,
                            **{k: np.concatenate(v).astype(np.float32) for k, v in acc.items()})
        print(f"  {eid}: {len(rows)} pairs -> {out_path.name}")

    print(f"done -> {feat_dir}")


if __name__ == "__main__":
    main()
