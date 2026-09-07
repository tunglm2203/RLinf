"""M8 quality sweep — cheap head-only levers on the ego cache (Cosmos features), no GPU/re-extract.
Grid: pooling {last_last,mean_last,mean_mid} × (n_bins, rank_weight). Same locked recipe
(AdamW+cosine+warmup+grad-clip, 150 ep, select-on-val, report-on-test).
Baseline = pool=last_last, bins=10, rank=0.5 -> E:ego test ρ 0.321."""
import copy
import math
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

import losses
from dataset import PairFeatures
from evaluate import report
from model import ProgressHead

HERE = Path(__file__).parent
CACHE = HERE / "cache"
POOLS = ["last_last", "mean_last", "mean_mid"]
CONFIGS = [(10, 0.5), (10, 1.0), (10, 2.0), (20, 0.5), (1, 0.5)]   # (n_bins, rank_weight); (1,*)=huber
EPOCHS, LR, SPB, SEED = 150, 1e-3, 16, 0


def train_one(data, n_bins, rank):
    w = {"ce": 1.0, "huber": 1.0, "huber_neg": 0.0, "rank": rank}
    torch.manual_seed(SEED)
    tr, va, te = data.split("train"), data.split("val"), data.split("test")
    head = ProgressHead(input_dim=data.dim, n_bins=n_bins)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    total = EPOCHS * max(1, math.ceil(len(tr.seg_ids) / SPB))
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)
    best_val, best_state = -2.0, copy.deepcopy(head.state_dict())
    for ep in range(EPOCHS):
        head.train()
        for b in tr.batches(SPB, time_warp=False, seed=ep):
            loss, _ = losses.total_loss(head(b["feats"]), b, w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step()
        vs = report(head, va)["spearman_head"]
        if vs == vs and vs > best_val:
            best_val, best_state = vs, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    m = report(head, te)
    return m["spearman_head"], m.get("language_sensitivity")


rows = []
for pool in POOLS:
    data = PairFeatures(CACHE, pool=pool)
    for n_bins, rank in CONFIGS:
        rho, lang = train_one(data, n_bins, rank)
        rows.append((pool, n_bins, rank, rho, lang))
        print(f"  pool={pool:9s} bins={n_bins:2d} rank={rank:.1f}  test ρ={rho:+.3f}  lang={lang:+.3f}", flush=True)

rows.sort(key=lambda r: -(r[3] if r[3] == r[3] else -9))
print("\n=== ranked by E:ego test ρ (baseline last_last/10/0.5 = 0.321) ===")
for pool, nb, rk, rho, lang in rows:
    print(f"  ρ={rho:+.3f}  lang={lang:+.3f}  | pool={pool:9s} bins={nb:2d} rank={rk}")
