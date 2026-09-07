"""Standard evaluation on the 3 robometer eval groups (RBM-EVAL-ID / RBM-EVAL-OOD / LIBERO-mixed).

Frozen two-stage path: the cache must already carry features (run extract_features.py on it first).
For each group it pools the group's datasets and reports the 7 canonical metrics (report_metrics/
fmt7); --per-dataset also breaks each group down by dataset. cond=global, target=t/T throughout;
LIBERO-mixed carries success+failure so its quality trio (SFΔ/P_succ/P_fail) is populated.

    python eval_rbm_groups.py --cache cache_rbm_eval --head runs/vh_baseline_500_ce10/head.pt --per-dataset
"""
import argparse
from pathlib import Path

import torch

from dataset import PairFeatures, SplitView
from model import ProgressHead
from evaluate import report, fmt7
from rbm_eval_groups import GROUPS

HERE = Path(__file__).parent


def load(p, dim, n_bins):
    h = ProgressHead(input_dim=dim, n_bins=n_bins)
    sd = torch.load(p, map_location="cpu", weights_only=False)
    h.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
    return h.eval()


def _n(x):
    return float("nan") if x is None else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=HERE / "cache_rbm_eval")
    ap.add_argument("--head", type=Path, default=HERE / "runs/vh_baseline_500_ce10/head.pt")
    ap.add_argument("--pool", default="last_last")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--label", default="head")
    ap.add_argument("--per-dataset", action="store_true")
    args = ap.parse_args()

    data = PairFeatures(args.cache, pool=args.pool)
    head = load(args.head, data.dim, args.n_bins)
    avail = set(data.pairs.dataset.unique())

    for gname, datasets in GROUPS.items():
        present = [d for d in datasets if d in avail]
        if not present:
            print(f"--- E:{gname}: no datasets present in cache, skipped ---\n")
            continue
        rows = data.pairs[data.pairs.dataset.isin(present)].reset_index(drop=True)
        sig = (f"[{args.label}] E:{gname}  (cond=global, target=t/T | "
               f"{len(present)}/{len(datasets)} datasets, {rows.episode_id.nunique()} eps)")
        print(fmt7(report(head, SplitView(data, rows), signature=sig)))
        if args.per_dataset:
            for d in present:
                dr = rows[rows.dataset == d].reset_index(drop=True)
                m = report(head, SplitView(data, dr))
                print(f"    {d[:52]:52s} ρ={_n(m['spearman_head']):+.3f}  "
                      f"lang={_n(m.get('language_sensitivity')):+.3f}  "
                      f"SFΔ={_n(m.get('succ_fail_diff')):+.3f}  ({dr.episode_id.nunique()}ep)")
        print()


if __name__ == "__main__":
    main()
