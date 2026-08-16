"""Read the ego-centric corpus release into segment records for the value head.

NOTE: this reader targets the FUTURE corpus-release schema (manifest.parquet + narratives/).
The active egoverse prototype uses probes/egoverse_data.py instead (build_labels.py). Kept for
when the release lands (it adds rephrasings -> the paraphrase loss).

The corpus release (see ego_dataset_example/corpus_sample_bundle/corpus/SCHEMA.md) stores:
  manifest.parquet                              one row per clip (the queryable index)
  narratives/<source>/**/*.parquet             one row per SEGMENT: text, rephrasings, bounds
  videos/<source>/<orig_id>.mp4                 the frames (native fps)

A segment's `text` is the sub-goal instruction; `rephrasings` are paraphrases of it (→ the
paraphrase-consistency loss); `lh_text`/`rh_text`/`ego_text` are the per-hand + head narratives
(excluded from the value feature by design, kept here for completeness). Segment boundaries
`start_f`/`end_f` are absolute source-video frames.

This reader is deliberately thin: it joins manifest↔narratives and yields plain dataclasses.
Frame *pixels* are pulled lazily by build_labels via the video path, so nothing here decodes
video.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# 'joystick_v1' narratives are the coarse 2 s-grid auto-labels; drop per the plan's M0 filter.
_DROP_GEN_MODELS = {"joystick_v1"}


@dataclass
class Segment:
    clip_id: str
    seg_idx: int          # position within the clip, in start-time order
    start_f: int          # absolute source-video frame, inclusive
    end_f: int            # exclusive
    text: str             # the sub-goal instruction
    rephrasings: list[str]
    lh_text: str = ""
    rh_text: str = ""
    ego_text: str = ""

    @property
    def n_frames(self) -> int:
        return self.end_f - self.start_f

    @property
    def is_composite_narrative(self) -> bool:
        """True when `text` is the per-hand narration ("[ego] … | [left hand] …"), not a
        task-level sub-goal. egodex narrates this way; egoverse gives clean sub-goals. The
        value head conditions on the SUB-GOAL and excludes narratives, so build_labels uses
        this to decide whether `text` is usable as a sub-goal (see the reader's --help)."""
        return "[left hand]" in self.text or "[right hand]" in self.text or "[ego]" in self.text

    def local_progress(self, frame: int) -> float:
        if self.n_frames <= 1:
            return 1.0
        return float(min(1.0, max(0.0, (frame - self.start_f) / (self.n_frames - 1))))


@dataclass
class Clip:
    clip_id: str
    source: str
    orig_id: str
    fps: float
    n_frames: int
    task_name: str
    split: str
    split_group: str
    video_path: Path | None
    segments: list[Segment] = field(default_factory=list)

    @property
    def global_task(self) -> str:
        # The release has no free-text global instruction; task_name is the closest handle.
        # Underscores → spaces so it reads as an instruction ("change the light bulb on a lamp").
        return self.task_name.replace("freeform_", "").replace("_", " ").strip()


def _narrative_files(corpus_root: Path) -> list[str]:
    return sorted(glob.glob(str(corpus_root / "narratives" / "**" / "*.parquet"), recursive=True))


def _resolve_video(corpus_root: Path, source: str, orig_id: str, source_video: str | None) -> Path | None:
    local = corpus_root / "videos" / source / f"{orig_id}.mp4"
    if local.exists():
        return local
    if source_video and Path(source_video).exists():   # native-annotation sources point out
        return Path(source_video)
    return None


def _coerce_rephrasings(val) -> list[str]:
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    if isinstance(val, str) and val.strip():
        try:
            j = json.loads(val)
            return [str(x) for x in j] if isinstance(j, list) else [val]
        except json.JSONDecodeError:
            return [val]
    return []


def load_clips(corpus_root: str | Path, min_seg_frames: int = 5,
               sources: tuple[str, ...] | None = None) -> list[Clip]:
    """Join manifest ↔ narratives into Clip objects with their Segments.

    `min_seg_frames` drops degenerate segments (< 5 frames by default, per the plan's M0
    filter). `sources` optionally restricts to e.g. ('egoverse',) — egoverse has semantic
    boundaries and is preferred over egodex's 2 s grid.
    """
    corpus_root = Path(corpus_root)
    man = pd.read_parquet(corpus_root / "manifest.parquet").set_index("clip_id")
    narr = pd.concat([pd.read_parquet(f) for f in _narrative_files(corpus_root)], ignore_index=True)
    if "gen_model" in narr.columns:
        narr = narr[~narr.gen_model.isin(_DROP_GEN_MODELS)]

    clips: list[Clip] = []
    for clip_id, g in narr.groupby("clip_id"):
        if clip_id not in man.index:
            continue
        row = man.loc[clip_id]
        if sources and row.get("source") not in sources:
            continue
        segs, seg_i, seen = [], 0, set()
        for _, s in g.sort_values("start_f").iterrows():
            start_f, end_f = int(s.start_f), int(s.end_f)
            text = str(s.get("text", "")).strip()
            if end_f - start_f < min_seg_frames or not text:
                continue
            key = (start_f, end_f, text)          # drop exact-duplicate spans (sample-bundle quirk)
            if key in seen:
                continue
            seen.add(key)
            segs.append(Segment(
                clip_id=clip_id, seg_idx=seg_i, start_f=start_f, end_f=end_f, text=text,
                rephrasings=_coerce_rephrasings(s.get("rephrasings")),
                lh_text=str(s.get("lh_text") or ""), rh_text=str(s.get("rh_text") or ""),
                ego_text=str(s.get("ego_text") or ""),
            ))
            seg_i += 1
        if not segs:
            continue
        clips.append(Clip(
            clip_id=clip_id, source=str(row.get("source")), orig_id=str(row.get("orig_id")),
            fps=float(row.get("fps", 30.0)), n_frames=int(row.get("n_frames", 0)),
            task_name=str(row.get("task_name", "")),
            split=str(row.get("split", "train")), split_group=str(row.get("split_group", clip_id)),
            video_path=_resolve_video(corpus_root, str(row.get("source")), str(row.get("orig_id")),
                                      row.get("source_video")),
            segments=segs,
        ))
    return clips


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True, help="…/corpus dir with manifest.parquet")
    ap.add_argument("--sources", nargs="*", default=None)
    args = ap.parse_args()

    clips = load_clips(args.corpus, sources=tuple(args.sources) if args.sources else None)
    nseg = sum(len(c.segments) for c in clips)
    print(f"{len(clips)} clips | {nseg} segments | sources "
          f"{sorted({c.source for c in clips})}")
    for c in clips[:3]:
        print(f"\n{c.clip_id}  [{c.source}]  split={c.split}  \"{c.global_task}\"  "
              f"({len(c.segments)} segs, video={'yes' if c.video_path else 'MISSING'})")
        for s in c.segments[:4]:
            print(f"   seg{s.seg_idx} [{s.start_f}-{s.end_f}] \"{s.text}\"  "
                  f"(+{len(s.rephrasings)} rephrasings)")
