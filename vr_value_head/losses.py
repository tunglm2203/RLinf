"""Value-head losses — each tied to a failure mode from VALUE_HEAD_RESEARCH_PLAN.md (M2).

    huber          base local-progress regression
    pairwise_rank  order within a segment (absolute values don't cross the ego→robot gap; order does)
    wrong_subgoal  same frame + WRONG sub-goal → abstain to 0.5 (creates language sensitivity)
    paraphrase     same frame + reworded sub-goal → small L2 (disabled on egoverse: no rephrasings)
    time_warp      handled as a DATASET augmentation (non-uniform temporal resampling), not here —
                   it relabels local targets so a frame-index clock can't fit; see dataset.py

Every function takes already-computed predictions and returns a scalar; the training loop
weights and sums them. Batches carry a `role` mask so pos/neg rows get the right term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def huber_local(pred_local: torch.Tensor, target_local: torch.Tensor,
                pos_mask: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """Regression on POSITIVE rows only (correct sub-goal)."""
    if pos_mask.sum() == 0:
        return pred_local.new_zeros(())
    return F.huber_loss(pred_local[pos_mask], target_local[pos_mask], delta=delta)


def discrete_ce(logits: torch.Tensor, target_local: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """robometer's discrete progress loss: cross-entropy over n_bins classes. The continuous
    target ∈ [0,1] is binned into `n_bins = logits.shape[-1]` classes. Applied on `mask` rows."""
    if mask.sum() == 0:
        return logits.new_zeros(())
    n_bins = logits.shape[-1]
    bins = (target_local * n_bins).long().clamp(0, n_bins - 1)     # [0,1] → class id
    return F.cross_entropy(logits[mask], bins[mask])


def pairwise_rank(pred_local: torch.Tensor, target_local: torch.Tensor,
                  group_id: torch.Tensor, pos_mask: torch.Tensor,
                  margin: float = 0.0, max_pairs: int = 4096) -> torch.Tensor:
    """Within-segment ordering: if target_i < target_j then pred_i should be < pred_j.

    Margin-ranking over pairs sharing a `group_id` (= segment). Order transfers across the
    domain gap even when absolute progress does not, so this is the load-bearing signal for
    cross-domain use. Pairs are formed only among positives, capped at `max_pairs`.
    """
    idx = torch.nonzero(pos_mask, as_tuple=False).flatten()
    if idx.numel() < 2:
        return pred_local.new_zeros(())
    i, j = torch.meshgrid(idx, idx, indexing="ij")
    same = group_id[i] == group_id[j]
    diff = target_local[i] - target_local[j]
    valid = same & (diff < -1e-6)                       # i strictly earlier than j
    ii, jj = i[valid], j[valid]
    if ii.numel() == 0:
        return pred_local.new_zeros(())
    if ii.numel() > max_pairs:
        sel = torch.randperm(ii.numel(), device=ii.device)[:max_pairs]
        ii, jj = ii[sel], jj[sel]
    # want pred_i < pred_j  ⇒  margin_ranking with target = -1
    return F.margin_ranking_loss(pred_local[ii], pred_local[jj],
                                 torch.full_like(pred_local[ii], -1.0), margin=margin)


def wrong_subgoal(pred_local: torch.Tensor, neg_mask: torch.Tensor,
                  abstain: float = 0.5) -> torch.Tensor:
    """NEGATIVE rows = same frame under a WRONG sub-goal. With the wrong goal the head has no
    basis to claim progress, so it is pulled toward an uninformative constant (0.5). This is
    what makes ρ(correct) ≫ ρ(wrong) — the language-sensitivity kill criterion. Design knob:
    `abstain`. (An alternative is to decorrelate; abstain-to-mid is the simplest defensible one.)"""
    if neg_mask.sum() == 0:
        return pred_local.new_zeros(())
    tgt = torch.full_like(pred_local[neg_mask], abstain)
    return F.mse_loss(pred_local[neg_mask], tgt)


def paraphrase_consistency(pred_a: torch.Tensor, pred_b: torch.Tensor) -> torch.Tensor:
    """Same frame, two paraphrases of the SAME sub-goal → predictions should match. Load-bearing
    under postfix (the varying text sits inside the pooled feature). Disabled on egoverse
    (no rephrasings); wired for the corpus release. `pred_a`/`pred_b` are aligned pairs."""
    if pred_a.numel() == 0:
        return pred_a.new_zeros(())
    return F.mse_loss(pred_a, pred_b)


def total_loss(preds: dict, batch: dict, weights: dict) -> tuple[torch.Tensor, dict]:
    """Weighted sum. `batch` provides target_local, group_id, pos_mask, neg_mask.

    robometer-aligned: the wrong-sub-goal negatives are FOLDED into the same regression with
    target 0.0 (their DIFFERENT_TASK strategy = "predict 0 progress if not the same task"),
    not a separate abstain term. neg `target_local` is already 0 (NaN→0 in the dataset), so
    the regression mask is pos ∪ neg. `rank` is ours-extra (robometer has no ranking loss);
    default weight 0 to mirror robometer, raise it to re-enable ordering pressure.
    """
    pl = preds["local"]
    # `huber_neg` is robometer's wrong-task→0 term, kept SEPARATE and OFF by default (weight 0):
    # their target-0 samples are diluted to ~25% by REVERSE/REWIND augmentations that a per-frame
    # head can't produce, so at our ~49% negative fraction, folding them in monotonically hurts
    # (ρ 0.32→0.15 as its weight goes 0→1). Language sensitivity comes from last_last features,
    # not from this term. Raise `huber_neg` only if negatives get diluted (e.g. a sequence head).
    terms = {
        # DISCRETE (robometer): cross-entropy over bins when the head is in discrete mode; else
        # CONTINUOUS: huber regression. The two are mutually exclusive (whichever the head emits).
        "ce": (discrete_ce(preds["local_logits"], batch["target_local"], batch["pos_mask"])
               if "local_logits" in preds else pl.new_zeros(())),
        "huber": (huber_local(pl, batch["target_local"], batch["pos_mask"])
                  if "local_logits" not in preds else pl.new_zeros(())),
        "huber_neg": huber_local(pl, batch["target_local"], batch["neg_mask"]),   # neg target=0
        "rank": pairwise_rank(pl, batch["target_local"], batch["group_id"], batch["pos_mask"]),
    }
    if "para_a" in preds and "para_b" in preds:
        terms["para"] = paraphrase_consistency(preds["para_a"], preds["para_b"])
    total = sum(weights.get(k, 0.0) * v for k, v in terms.items())
    return total, {k: float(v.detach()) for k, v in terms.items()}


if __name__ == "__main__":       # smoke on synthetic data
    torch.manual_seed(0)
    B = 32
    pred = {"local": torch.rand(B, requires_grad=True)}
    batch = {
        "target_local": torch.rand(B),
        "group_id": torch.randint(0, 4, (B,)),
        "pos_mask": torch.rand(B) > 0.3,
        "neg_mask": torch.rand(B) > 0.7,
    }
    w = {"huber": 1.0, "rank": 0.5, "wrong": 0.3}
    tot, parts = total_loss(pred, batch, w)
    tot.backward()
    print("total:", float(tot), "| parts:", {k: round(v, 4) for k, v in parts.items()})
    print("grad flows:", pred["local"].grad is not None)
