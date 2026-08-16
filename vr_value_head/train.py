"""Train the ProgressHead on cached features (frozen backbone). Seconds, not GPU-hours.

    /opt/venv/egoverse/bin/python train.py --cache cache --pool mean_last --time-warp

Logs per-term losses and, each eval, the head's held-out Spearman AGAINST THE CLOCK plus the
matched-frame language gap — the two numbers that decide whether the head is a reward model or
a stopwatch. Checkpoints the head (same nn.Module as the RL critic).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import losses
from dataset import PairFeatures
from evaluate import report
from model import ProgressHead

HERE = Path(__file__).parent
# Baseline: robometer head + regression, negatives EXCLUDED (huber_neg=0, folding hurts at our
# ~49% neg fraction), plus OUR pairwise ranking (rank=0.5) — an ordering signal robometer lacks
# that helps, and is orthogonal to their (inapplicable) reverse/rewind anti-clock.
DEFAULT_WEIGHTS = {"ce": 1.0, "huber": 1.0, "huber_neg": 0.0, "rank": 0.5}


def train(cache, pool, epochs, lr, segs_per_batch, time_warp, weights, device, out, n_bins=1):
    import copy
    import math
    from transformers import get_cosine_schedule_with_warmup
    data = PairFeatures(Path(cache), pool=pool)
    # Proper hygiene: SELECT the checkpoint on `val`, REPORT once on held-out `test`. The old
    # code trained-and-selected on test (leakage) AND saved the final epoch — with ρ bouncing
    # 0.06→0.53 across epochs on a small eval set, the final epoch was a coin flip.
    tr, va, te = data.split("train"), data.split("val"), data.split("test")
    print(f"dim={data.dim} | train segs={len(tr.seg_ids)} | val {len(va.seg_ids)} | test {len(te.seg_ids)}")
    head = ProgressHead(input_dim=data.dim, n_bins=n_bins).to(device)
    print(f"loss: {'discrete CE (%d bins)' % n_bins if n_bins > 1 else 'continuous huber'}")
    # robometer optimizer recipe: AdamW + weight_decay 0.01 + cosine LR + 10% warmup + grad-clip 1.0.
    # (We keep lr=1e-3 for this frozen-feature MLP; robometer's 2e-5 is for fine-tuning a 4B VLM.)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(tr.seg_ids) / segs_per_batch))
    total_steps = epochs * steps_per_epoch
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    best_val, best_state, best_ep = -2.0, copy.deepcopy(head.state_dict()), -1
    for ep in range(epochs):
        head.train()
        agg = {}
        for b in tr.batches(segs_per_batch, time_warp=time_warp, seed=ep):
            feats = b["feats"].to(device)
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            preds = head(feats)
            loss, parts = losses.total_loss(preds, b, weights)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        vm = report(head, va, device)                       # select on VAL every epoch
        vs = vm["spearman_head"]
        if vs == vs and vs > best_val:                      # best-val checkpoint (NaN-safe)
            best_val, best_state, best_ep = vs, copy.deepcopy(head.state_dict()), ep
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            tr_s = " ".join(f"{k}={v/max(1,len(list(tr.seg_ids))//segs_per_batch+1):.3f}"
                            for k, v in agg.items())
            print(f"ep{ep:3d} | {tr_s} | val ρ={vs:.3f} (best {best_val:.3f}@{best_ep}) "
                  f"lang_sens={vm.get('language_sensitivity', float('nan')):+.3f}")

    head.load_state_dict(best_state)                        # restore best-val checkpoint
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "pool": pool, "dim": data.dim, "best_epoch": best_ep},
               out / "head.pt")
    # report on ALL THREE splits — train shows the fit/overfit gap; test is the held-out number
    metrics = {name: report(head, view, device)
               for name, view in (("train", tr), ("val", va), ("test", te))}
    metrics["best_epoch"] = best_ep
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {out}/head.pt (best val ρ={best_val:.3f}@ep{best_ep})")
    for name in ("train", "val", "test"):
        m = metrics[name]
        print(f"  {name:5s} ρ={m['spearman_head']:+.3f}  "
              f"lang_sens={m.get('language_sensitivity', float('nan')):+.3f}  "
              f"rho_wrong={m.get('rho_wrong_subgoal', float('nan')):+.3f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--pool", choices=["mean_last", "last_last", "mean_mid"], default="mean_last")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--segs-per-batch", type=int, default=16)
    parser.add_argument("--time-warp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "head")
    parser.add_argument("--n-bins", type=int, default=1,
                    help="1=continuous huber; >1=discrete cross-entropy over bins (robometer ships 10)")
    args = parser.parse_args()
    train(args.cache, args.pool, args.epochs, args.lr, args.segs_per_batch,
          args.time_warp, DEFAULT_WEIGHTS, args.device, args.out, n_bins=args.n_bins)


if __name__ == "__main__":
    main()
