"""Evaluate a trained head across datasets (per-dataset breakdown) with the robometer/GVL metrics.

For each `dataset` in the cache, reports progress correlation (spearman / pearson_voc / kendall_tau_a),
language sensitivity, and the quality-tier separation (succ_fail_diff + terminal predicted progress per
tier). Datasets with <2 quality tiers only get progress + language metrics.

    /opt/venv/egoverse/bin/python eval_all.py --cache cache_rbm --head runs/head_libero_quality.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import PairFeatures, SplitView
from evaluate import report, predict
from model import ProgressHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--pool", default="last_last")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = PairFeatures(args.cache, pool=args.pool)
    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    head = ProgressHead(input_dim=data.dim, n_bins=args.n_bins)
    head.load_state_dict(ck["state_dict"]); head.to(args.device).eval()

    has_ds = "dataset" in data.pairs.columns
    groups = data.pairs.dataset.unique() if has_ds else ["ALL"]
    rows = []
    for g in sorted(groups):
        sub = data.pairs if not has_ds else data.pairs[data.pairs.dataset == g]
        view = SplitView(data, sub.reset_index(drop=True))
        m = report(head, view, args.device)
        # per-tier terminal predicted progress
        pr = predict(head, view, args.device)
        pos = pr[pr.role == "pos"]
        term = pos.loc[pos.groupby("episode_id").frame.idxmax()]
        tiers = term.groupby("quality").pred.mean() if "quality" in term else pd.Series(dtype=float)
        rows.append(dict(
            dataset=g[:42], n_ep=pos.episode_id.nunique(),
            rho=m.get("spearman_head"), voc=m.get("pearson_voc"), tau_a=m.get("kendall_tau_a"),
            lang=m.get("language_sensitivity"), sf_diff=m.get("succ_fail_diff"),
            t_succ=tiers.get("successful", np.nan), t_subopt=tiers.get("suboptimal", np.nan),
            t_fail=tiers.get("failure", np.nan)))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 44)
    print(df.round(3).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False); print(f"\nsaved {args.out}")
    # macro-average over datasets that have all 3 tiers (the quality-ranking axis)
    tri = df.dropna(subset=["t_fail", "t_subopt", "t_succ"])
    if len(tri):
        print(f"\n3-tier datasets ({len(tri)}): mean terminal  "
              f"fail={tri.t_fail.mean():.3f} < subopt={tri.t_subopt.mean():.3f} < succ={tri.t_succ.mean():.3f}"
              f"  | mean sf_diff={tri.sf_diff.mean():+.3f}")
        print(f"progress (all {len(df)} datasets): mean ρ={df.rho.mean():.3f} voc={df.voc.mean():.3f} "
              f"tau_a={df.tau_a.mean():.3f} lang={df.lang.mean():.3f}")


if __name__ == "__main__":
    main()
