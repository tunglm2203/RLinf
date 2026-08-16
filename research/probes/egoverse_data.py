"""Shared reader for the raw EgoVerse zarr episodes.

One place that knows the raw layout, so the label generator, the probe and the
inspection notebook all agree.

Raw layout (verified on 500 scale/human_bimanual episodes):
    <episode>.zarr/
        zarr.json                 attributes: task_name, task_description, total_frames,
                                  fps, embodiment, lab, scene, intrinsics, ...
        annotations               (N,) object -> JSON {"text","start_idx","end_idx"}
        images.front_1            (T,) variable_length_bytes -> JPEG (640x480)
        obs_head_pose             (T,7)  xyz + quaternion, GLOBAL SLAM frame
        left/right.obs_keypoints  (T,63) = 21x3
        left/right.obs_wrist_pose (T,7)
        left/right.obs_ee_pose    (T,7)

Gotcha: `arr[i]` on a variable_length_bytes array returns an unsized object.
Use `arr[i:i+1][0]` — that is what read_frame() does.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr

DEFAULT_ROOT = Path("/mnt/workspace2/all_dataset/egoverse")


@dataclass
class Segment:
    """One sub-goal annotation (LEVEL 2)."""

    idx: int
    start: int
    end: int          # exclusive
    text: str

    @property
    def n_frames(self) -> int:
        return self.end - self.start

    def duration_s(self, fps: float = 30.0) -> float:
        return self.n_frames / fps

    def local_progress(self, frame: int) -> float:
        """0 at the segment's first frame, ->1 at its last."""
        if self.n_frames <= 1:
            return 1.0
        return float(np.clip((frame - self.start) / (self.n_frames - 1), 0.0, 1.0))


@dataclass
class Episode:
    episode_id: str
    path: Path
    attrs: dict
    segments: list[Segment] = field(default_factory=list)

    # ── clip-level metadata ───────────────────────────────────────────────────
    @property
    def n_frames(self) -> int:
        return int(self.attrs.get("total_frames") or 0)

    @property
    def fps(self) -> float:
        return float(self.attrs.get("fps") or 30.0)

    @property
    def global_task(self) -> str:
        """LEVEL 1 instruction. Prefer task_description (clean prose) over the slug."""
        d = (self.attrs.get("task_description") or "").strip()
        if d:
            return d
        return str(self.attrs.get("task_name") or "").replace("_", " ").strip()

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    # ── frame access ──────────────────────────────────────────────────────────
    def _group(self):
        return zarr.open(str(self.path), mode="r")

    def read_frame(self, frame: int) -> np.ndarray:
        """Decode one RGB frame (H, W, 3) uint8."""
        import simplejpeg

        arr = self._group()["images.front_1"]
        raw = arr[frame : frame + 1][0]          # scalar index returns an unsized object
        buf = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
        return simplejpeg.decode_jpeg(buf, colorspace="RGB")

    def segment_at(self, frame: int) -> Segment | None:
        for s in self.segments:
            if s.start <= frame < s.end:
                return s
        return None

    def covered_frames(self) -> np.ndarray:
        """Frames belonging to some segment (gap frames excluded)."""
        mask = np.zeros(self.n_frames, dtype=bool)
        for s in self.segments:
            mask[max(0, s.start) : min(self.n_frames, s.end)] = True
        return np.flatnonzero(mask)


def _parse_annotations(group, n_frames: int, min_seg_frames: int) -> tuple[list[Segment], dict]:
    """Parse + clean the annotations array. Returns (segments, drop-reason counts)."""
    stats = {"raw": 0, "unparsable": 0, "out_of_bounds": 0, "too_short": 0, "kept": 0}
    try:
        raw = group["annotations"][:]
    except KeyError:
        return [], stats

    parsed = []
    for r in raw:
        stats["raw"] += 1
        buf = r if isinstance(r, (bytes, bytearray)) else bytes(r)
        try:
            j = json.loads(buf)
            s, e, txt = int(j["start_idx"]), int(j["end_idx"]), str(j.get("text", "")).strip()
        except Exception:
            stats["unparsable"] += 1
            continue
        if e > n_frames or s < 0 or e <= s:      # 11 such segments across the 500
            s2, e2 = max(0, s), min(e, n_frames)
            if e2 <= s2:
                stats["out_of_bounds"] += 1
                continue
            stats["out_of_bounds"] += 1
            s, e = s2, e2
        if (e - s) < min_seg_frames:
            stats["too_short"] += 1
            continue
        parsed.append((s, e, txt))

    parsed.sort(key=lambda t: (t[0], t[1]))
    segs = [Segment(idx=i, start=s, end=e, text=t) for i, (s, e, t) in enumerate(parsed)]
    stats["kept"] = len(segs)
    return segs, stats


def load_episode(path: str | Path, min_seg_frames: int = 2) -> Episode:
    path = Path(path)
    attrs = json.load(open(path / "zarr.json")).get("attributes", {})
    ep = Episode(episode_id=path.name.replace(".zarr", ""), path=path, attrs=attrs)
    ep.segments, ep._seg_stats = _parse_annotations(  # type: ignore[attr-defined]
        zarr.open(str(path), mode="r"), ep.n_frames, min_seg_frames
    )
    return ep


def list_episodes(root: str | Path = DEFAULT_ROOT) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "zarr.json").exists())


def load_all(root: str | Path = DEFAULT_ROOT, limit: int | None = None,
             min_seg_frames: int = 2) -> list[Episode]:
    paths = list_episodes(root)
    if limit:
        paths = paths[:limit]
    return [load_episode(p, min_seg_frames=min_seg_frames) for p in paths]


def split_of(episode_id: str, val_frac: float = 0.1, test_frac: float = 0.1) -> str:
    """Deterministic per-EPISODE split (never split a trajectory across sets)."""
    import hashlib

    h = int(hashlib.md5(episode_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"
