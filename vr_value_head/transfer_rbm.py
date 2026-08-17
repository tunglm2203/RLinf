"""M3 transfer sweep (A/B/C) on a robometer dataset — same protocol as transfer.py, but the robot
target is a robometer set (MetaWorld / real-robot OOD) instead of our LIBERO.

  A scratch   : head trained on target-train-k only
  B transfer  : ego-pretrained head, finetuned on the SAME target-train-k
  C zero-shot : ego-pretrained head, no target training

Groups by global_task (k demos/TASK). Two source modes:
  --train-include / --test-include : separate train & test datasets (e.g. metaworld_train / _eval)
  --target                         : one dataset, split episodes per task by --test-frac

    python transfer_rbm.py --train-include metaworld_train --test-include metaworld_eval --ks 1 2 3 5
    python transfer_rbm.py --target mit_franka_rfm_rfm --ks 1 2 5 10
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import PairFeatures, SplitView
from evaluate import report
from model import ProgressHead
from transfer import train_head

HERE = Path(__file__).parent


def subsample(pool_rows, k, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for task, g in pool_rows.groupby("global_task"):
        eps = sorted(g.episode_id.unique()); rng.shuffle(eps)
        chosen += list(eps[:k])
    return pool_rows[pool_rows.episode_id.isin(chosen)]


def metrics(head, view, dev, **tags):
    m = report(head, view, dev)
    return {**tags, "rho": m.get("spearman_head"), "voc": m.get("pearson_voc"),
            "tau_a": m.get("kendall_tau_a"), "lang": m.get("language_sensitivity"),
            "sf_diff": m.get("succ_fail_diff")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=HERE / "cache_rbm")
    ap.add_argument("--ego-head", type=Path, default=HERE / "runs/vh_baseline_500_ce10/head.pt")
    ap.add_argument("--train-include", default=None)
    ap.add_argument("--test-include", default=None)
    ap.add_argument("--target", default=None, help="single dataset; split per task by --test-frac")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--pool", default="last_last")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = PairFeatures(args.cache, pool=args.pool)
    P = data.pairs
    if args.target:
        rows = P[P.dataset.str.contains(args.target)].copy()
        rng = np.random.default_rng(args.seed); tr_eps = []
        for t, g in rows.groupby("global_task"):
            eps = sorted(g.episode_id.unique()); rng.shuffle(eps)
            tr_eps += list(eps[max(1, round(args.test_frac * len(eps))):])
        pool_rows = rows[rows.episode_id.isin(tr_eps)]
        test_rows = rows[~rows.episode_id.isin(tr_eps)]
        name = args.target
    else:
        pool_rows = P[P.dataset.str.contains(args.train_include)]
        test_rows = P[P.dataset.str.contains(args.test_include)]
        name = args.train_include
    test = SplitView(data, test_rows.reset_index(drop=True))
    print(f"target={name} | train pool eps={pool_rows.episode_id.nunique()} "
          f"({pool_rows.global_task.nunique()} tasks) | test eps={test_rows.episode_id.nunique()}")

    head_ego = ProgressHead(input_dim=data.dim, n_bins=args.n_bins)
    head_ego.load_state_dict(torch.load(args.ego_head, map_location="cpu", weights_only=False)["state_dict"])
    head_ego.to(args.device)

    out = [metrics(head_ego, test, args.device, arm="C_zeroshot", k=0)]
    print(f"  C zero-shot        : ρ={out[-1]['rho']:+.3f} voc={out[-1]['voc']:+.3f} lang={out[-1]['lang']:+.3f}")
    for k in args.ks:
        tr = SplitView(data, subsample(pool_rows, k, args.seed).reset_index(drop=True))
        n = tr.rows.episode_id.nunique()
        torch.manual_seed(args.seed)
        hA = train_head(ProgressHead(input_dim=data.dim, n_bins=args.n_bins), tr, args.epochs, 1e-3, 16, args.device)
        hB = copy.deepcopy(head_ego); train_head(hB, tr, args.epochs, 1e-3, 16, args.device)
        a = metrics(hA, test, args.device, arm="A_scratch", k=k)
        b = metrics(hB, test, args.device, arm="B_transfer", k=k)
        out += [a, b]
        print(f"  k={k:<2d} ({n:3d} eps)  A ρ={a['rho']:+.3f}  B ρ={b['rho']:+.3f}  (B−A={b['rho']-a['rho']:+.3f}) "
              f"| A lang={a['lang']:+.3f} B lang={b['lang']:+.3f}")
    df = pd.DataFrame(out)
    o = HERE / "runs" / f"m3_transfer_{name.replace('/','_')}.csv"; o.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(o, index=False); print(f"saved {o}")


if __name__ == "__main__":
    main()
