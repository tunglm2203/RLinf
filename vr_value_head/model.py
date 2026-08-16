"""The progress value head — robometer's progress-head architecture.

Matches robometer's continuous progress head (robometer/models/heads.py):

    Linear(hidden_dim, hidden_dim // 2) -> LayerNorm -> GELU -> Dropout(0.1)
    -> Linear(hidden_dim // 2, 1) -> Sigmoid

The in-head LayerNorm is why we DON'T need an external StandardScaler on the features: it
renormalizes after the first projection, so the raw Cosmos-Reason2 hidden states (std ~14,
range ~[-100, 500]) no longer saturate the output sigmoid.

Note: this deliberately does NOT reuse rlinf's ValueHead — the company progress recipe follows
robometer here, so the offline head is no longer byte-identical to the RL critic module. When
the head is wired into RL, port these weights into a matching module (or give the critic this
same architecture).

Input  : a POOLED VLM feature vector (2048-d), pooled at extraction (mean_last/last_last/mean_mid).
Output : local_progress ∈ (0,1). `remaining` (for global = (k+local)/(k+remaining)) is optional
         and OFF by default — it has no training target yet, so global-progress derivation is
         deferred (see VALUE_HEAD_RESEARCH_PLAN.md, M2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProgressHead(nn.Module):
    def __init__(self, input_dim: int = 2048, dropout: float = 0.1,
                 predict_remaining: bool = False, n_bins: int = 1):
        """n_bins=1 → robometer's CONTINUOUS head (Linear→LN→GELU→Dropout→Linear→Sigmoid, L2/Huber).
        n_bins>1 → robometer's DISCRETE head: same trunk but final Linear→n_bins logits, NO sigmoid,
        trained with cross-entropy over `progress` binned into n_bins classes (robometer ships 10)."""
        super().__init__()
        self.predict_remaining = predict_remaining
        self.n_bins = n_bins
        h = input_dim // 2
        layers = [nn.Linear(input_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout),
                  nn.Linear(h, n_bins)]
        if n_bins == 1:
            layers.append(nn.Sigmoid())
        self.progress_head = nn.Sequential(*layers)
        if n_bins > 1:      # bin centers in [0,1] for the expected-value readout
            self.register_buffer("bin_centers", (torch.arange(n_bins) + 0.5) / n_bins)
        if predict_remaining:                       # optional aux (no target yet); ≥0 via softplus
            self.remaining_head = nn.Sequential(
                nn.Linear(input_dim, h), nn.LayerNorm(h), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(h, 1), nn.Softplus())

    def forward(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        """feats: [B, input_dim] pooled VLM features → dict of predictions. `local` is always a
        scalar in [0,1] (for eval/reward); discrete mode also returns `local_logits` for the CE loss."""
        raw = self.progress_head(feats)
        if self.n_bins == 1:
            out = {"local": raw.squeeze(-1)}                         # sigmoid → (0,1)
        else:
            probs = torch.softmax(raw, dim=-1)
            out = {"local": (probs * self.bin_centers).sum(-1),      # expected bin value ∈ (0,1)
                   "local_logits": raw}                              # [B, n_bins] for cross-entropy
        if self.predict_remaining:
            out["remaining"] = self.remaining_head(feats).squeeze(-1)
        return out

    @staticmethod
    def global_progress(local: torch.Tensor, remaining: torch.Tensor,
                        k: torch.Tensor) -> torch.Tensor:
        """Compose global progress from local/remaining and the grounded count k (# sub-goals
        done). Never regressed. Only usable once `remaining` has a training target."""
        return (k + local) / (k + remaining).clamp(min=1e-6)


if __name__ == "__main__":       # smoke
    h = ProgressHead(input_dim=2048)
    x = torch.randn(4, 2048)
    p = h(x)
    print("local:", p["local"].shape, "in [0,1]:",
          bool((p["local"] >= 0).all() and (p["local"] <= 1).all()))
    print("params:", sum(n.numel() for n in h.parameters()))
