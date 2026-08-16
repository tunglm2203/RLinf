"""Compose the N1.7 text-slot string from the hierarchical context.

Two arms, chosen in config (see VALUE_HEAD_RESEARCH_PLAN.md, M2 + the N1.7-prototype discussion):

  subgoal   : the current sub-goal alone. In-distribution for N1.7 (LIBERO SFT saw single
              clauses); the accuracy-ceiling reference. This is the PRIMARY arm.
  full      : "<global>. Done: <grounded history>. Now: <sub-goal>". The faithful stand-in for
              the hierarchical postfix span (global + grounded history + current sub-goal). The
              sub-goal is placed LAST so postfix pooling sits right after it (causal attention).
              OOD text for N1.7 → expected to be noisier; run it as the comparison arm.

`grounded_history` is the list of ALREADY-COMPLETED sub-goal texts (state-grounded, not
self-reported) — on egoverse, the texts of segments before the current one. `k = len(history)`.
"""

from __future__ import annotations


def compose(arm: str, subgoal: str, global_task: str = "",
            grounded_history: list[str] | None = None, max_history: int = 6) -> str:
    subgoal = subgoal.strip()
    if arm == "subgoal":
        return subgoal
    if arm == "full":
        hist = grounded_history or []
        if len(hist) > max_history:                 # keep the most recent, bound the string
            hist = hist[-max_history:]
        done = "; ".join(h.strip() for h in hist) if hist else "nothing yet"
        g = global_task.strip() or "complete the task"
        # sub-goal LAST — the position postfix pooling reads
        return f"{g}. Done: {done}. Now: {subgoal}"
    raise ValueError(f"unknown text arm: {arm!r} (expected 'subgoal' or 'full')")


if __name__ == "__main__":
    hist = ["put the yellow cup on the left", "put the blue cup on the left"]
    for arm in ("subgoal", "full"):
        print(f"[{arm}] {compose(arm, 'put the red cup on the left', 'Portion out snacks', hist)}")
