"""M3 — cross-domain transfer: does ego pretraining help progress prediction on robot data?

Runs the three-arm × k-sweep protocol (VALUE_HEAD_RESEARCH_PLAN.md, M3) on the LIBERO robot cache:

  A  scratch   : head trained on robot-train-k only            (in-domain baseline)
  B  transfer  : ego-pretrained head, finetuned on the SAME robot-train-k as A
  C  zero-shot : ego-pretrained head, evaluated on robot-test with NO robot training

Key control: at each k, A and B see the IDENTICAL robot_train_k and the IDENTICAL schedule —
only the initialization differs. Per-task budget cap (task 0 -> 5, others -> 10). Single seed
(first pass, no error bars). Reports per-segment ρ and language_sensitivity on robot_test, and
plots ρ vs k (A, B curves; C and clock as reference lines).

    /opt/venv/egoverse/bin/python transfer.py --libero-cache cache_libero \
        --ego-head runs/vh_baseline_500_ce10/head.pt --pool last_last --n-bins 10
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import losses
from dataset import PairFeatures, SplitView
from evaluate import report
from model import ProgressHead

HERE = Path(__file__).parent
WEIGHTS = {"ce": 1.0, "huber": 1.0, "huber_neg": 0.0, "rank": 0.5}
CAP = {0: 5}          # per-task budget cap; default 10


def subsample_train(data: PairFeatures, k: int, seed: int) -> SplitView:
    """k demos/task from the robot TRAIN pool, capped per task (task 0 -> 5)."""
    rng = np.random.default_rng(seed)
    pool = data.pairs[data.pairs.split == "train"]
    chosen = []
    for task, g in pool.groupby("task_id"):
        eps = sorted(g.episode_id.unique())
        rng.shuffle(eps)
        chosen += list(eps[: min(k, CAP.get(int(task), 10))])
    return SplitView(data, pool[pool.episode_id.isin(chosen)].reset_index(drop=True))


def train_head(head, view, epochs, lr, segs_per_batch, device):
    """Fixed-epoch trainer (no val selection — impossible at k=1). AdamW + cosine, same recipe."""
    from transformers import get_cosine_schedule_with_warmup
    head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    steps = max(1, math.ceil(len(view.seg_ids) / segs_per_batch)) * epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for ep in range(epochs):
        for b in view.batches(segs_per_batch, time_warp=False, seed=ep):
            feats = b["feats"].to(device)
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            loss, _ = losses.total_loss(head(feats), b, WEIGHTS)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step()
    return head


def load_ego_head(path, dim, n_bins, device):
    d = torch.load(path, map_location="cpu", weights_only=False)
    head = ProgressHead(input_dim=dim, n_bins=n_bins)
    head.load_state_dict(d["state_dict"])
    return head.to(device)


def metrics_row(head, test_view, device, **tags):
    m = report(head, test_view, device)
    return {**tags, "rho": m["spearman_head"], "lang_sens": m.get("language_sensitivity", float("nan")),
            "rho_wrong": m.get("rho_wrong_subgoal", float("nan")), "clock": m.get("clock", float("nan"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-cache", type=Path, default=HERE / "cache_libero")
    ap.add_argument("--ego-head", type=Path, default=HERE / "runs/vh_baseline_500_ce10/head.pt")
    ap.add_argument("--pool", default="last_last")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--segs-per-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "m3_transfer")
    args = ap.parse_args()

    data = PairFeatures(args.libero_cache, pool=args.pool)
    test = data.split("test")
    print(f"robot cache: dim={data.dim} | train segs={len(data.split('train').seg_ids)} | "
          f"test segs={len(test.seg_ids)}")

    head_ego = load_ego_head(args.ego_head, data.dim, args.n_bins, args.device)
    rows = []

    # Arm C — ego zero-shot (k-independent)
    rows.append(metrics_row(head_ego, test, args.device, arm="C_zeroshot", k=0))
    print(f"  C zero-shot         : ρ={rows[-1]['rho']:+.3f}  lang_sens={rows[-1]['lang_sens']:+.3f}")

    for k in args.ks:
        train_k = subsample_train(data, k, args.seed)
        n_ep = train_k.rows.episode_id.nunique()
        # Arm A — scratch
        torch.manual_seed(args.seed)
        head_A = train_head(ProgressHead(input_dim=data.dim, n_bins=args.n_bins),
                            train_k, args.epochs, args.lr, args.segs_per_batch, args.device)
        rows.append(metrics_row(head_A, test, args.device, arm="A_scratch", k=k))
        # Arm B — transfer (ego init, SAME data + schedule)
        head_B = copy.deepcopy(head_ego)
        train_head(head_B, train_k, args.epochs, args.lr, args.segs_per_batch, args.device)
        rows.append(metrics_row(head_B, test, args.device, arm="B_transfer", k=k))
        a, b = rows[-2], rows[-1]
        print(f"  k={k:<2d} ({n_ep:3d} eps)  A ρ={a['rho']:+.3f}  B ρ={b['rho']:+.3f}  "
              f"(B−A={b['rho']-a['rho']:+.3f}) | A lang={a['lang_sens']:+.3f} B lang={b['lang_sens']:+.3f}")

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "results.csv", index=False)
    plot(df, args.out / "rho_vs_k.png")
    print(f"\nsaved {args.out}/results.csv  and  rho_vs_k.png")


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = df[df.arm == "A_scratch"].sort_values("k")
    B = df[df.arm == "B_transfer"].sort_values("k")
    C = df[df.arm == "C_zeroshot"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(A.k, A.rho, "-o", label="A: scratch (robot only)", color="tab:blue")
    ax.plot(B.k, B.rho, "-o", label="B: ego-pretrained → finetune", color="tab:green")
    if len(C):
        ax.axhline(C.rho.iloc[0], ls="--", color="tab:orange", label="C: ego zero-shot")
    ax.axhline(1.0, ls=":", color="grey", alpha=.6, label="clock (local target)")
    ax.set_xlabel("robot demos / task  (k)"); ax.set_ylabel("held-out test ρ")
    ax.set_title("M3 cross-domain transfer — does ego pretraining help?")
    ax.set_xticks(sorted(df[df.k > 0].k.unique())); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
