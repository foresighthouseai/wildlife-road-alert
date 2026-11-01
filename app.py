"""Wildlife Road Alert: early driver's-view wildlife warning with SAM 3.1.

Run one video:  python app.py --video test_videos/clip.mp4
Run a folder:   python app.py --batch test_videos/
Image test:     python app.py --eval-images "Deers on roads.v4i.coco"
Showcase reel:  python app.py --showcase
Dashboard page: python app.py --dashboard
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import config

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
SUPPORTED_SUFFIXES = {".mp4", ".mov"}
LEVEL_ORDER = {"green": 0, "amber": 1, "red": 2}
LEVEL_COLORS = {"green": (113, 191, 47), "amber": (36, 165, 245),
                "red": (94, 77, 255)}  # BGR


class UserError(Exception):
    """An error the user can act on; shown as one plain-English line."""


@dataclass
class Detection:
    frame: int
    time_s: float
    track_id: str        # e.g. "deer-0" (word + SAM object id)
    label: str           # the animal word that was searched for
    obj_id: str
    confidence: float
    x1: int
    y1: int
    x2: int              # inclusive bottom-right, source pixels
    y2: int
    frame_w: int
    frame_h: int
    on_road: bool = False
    mask: Optional[bytes] = None       # row-major 0/1 raster, box-sized
    mask_w: int = 0
    mask_h: int = 0

    @property
    def feet(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, float(self.y2))

    @property
    def box_h(self) -> int:
        return self.y2 - self.y1 + 1


@dataclass
class TrackFrame:
    """Geometry and warning state for one track in one frame."""
    det: Detection
    where: str                       # "on_road" | "near_road" | "away"
    toward: bool
    ttc_s: Optional[float]           # seconds until the vehicle reaches it
    distance_m: Optional[float]
    level: str                       # this track's level: green|amber|red
    on_road_duration: float


@dataclass
class Alert:
    alert_no: int
    time_s: float            # when the alert (first red) fired
    entry_time_s: float      # when the track entered the road (or first near)
    track_id: str
    detected_as: str
    confidence: float
    frame: int = 0
    group_count: int = 1
    severity: int = 1
    assurance: str = "Low"
    mean_confidence: float = 0.0
    tracked_s: float = 0.0
    muse_species: str = ""
    muse_confidence: Optional[float] = None
    heading_toward_road: str = ""
    sign_text: str = config.DEFAULT_SIGN_TEXT
    muse_rejected: bool = False
    muse_latency_s: Optional[float] = None
    muse_cost_usd: Optional[float] = None
    first_warn_s: Optional[float] = None   # first amber or red frame
    ttc_at_alert: Optional[float] = None
    distance_at_alert: Optional[float] = None
    lead_time_s: Optional[float] = None    # warning -> vehicle reaches animal


SEVERITY_TEXT = {1: "moderate", 2: "high", 3: "very high"}
SPECIES_PLURAL = {"deer": "DEER", "moose": "MOOSE", "elk": "ELK",
                  "bear": "BEARS", "bison": "BISON", "cow": "COWS"}


def severity_of(label: str) -> int:
    return int(config.SPECIES_SEVERITY.get(label, 1))


def severity_text(severity: int) -> str:
    return SEVERITY_TEXT.get(int(severity), "moderate")


def assurance_level(mean_confidence: float, tracked_s: float,
                    muse_confirmed: Optional[bool]) -> str:
    """High = all three conditions, Medium = two, Low = one or none."""
    met = sum([mean_confidence >= config.ASSURANCE_HIGH_SCORE,
               tracked_s >= config.ASSURANCE_MIN_TRACK_S,
               muse_confirmed is True])
    return "High" if met >= 3 else ("Medium" if met == 2 else "Low")


def muse_cost_usd(model: str, prompt_tokens, completion_tokens) -> Optional[float]:
    prices = config.MUSE_PRICES.get(model)
    if prices is None or prompt_tokens is None or completion_tokens is None:
        return None
    return (float(prompt_tokens) * prices[0]
            + float(completion_tokens) * prices[1]) / 1_000_000.0


def severity_for_species_text(species: str) -> Optional[int]:
    """Severity for a free-text species (e.g. 'white-tailed deer' -> deer)."""
    text = species.lower()
    for word in ("moose", "bison", "cow", "elk", "bear", "deer"):
        if word in text:
            return int(config.SPECIES_SEVERITY.get(word, 1))
    return None


def effective_species(alert: Alert) -> str:
    """Muse's species when it confirms an animal at confidence >= 0.6."""
    if (alert.muse_species and alert.muse_confidence is not None
            and alert.muse_confidence >= 0.6 and not alert.muse_rejected):
        return alert.muse_species
    return alert.detected_as


def apply_muse_species(alert: Alert) -> None:
    """Label and severity follow Muse's confirmed species; SAM word stays."""
    if effective_species(alert) != alert.detected_as:
        severity = severity_for_species_text(alert.muse_species)
        if severity is not None:
            alert.severity = severity


def rename_track_to_species(track_frames: Dict[str, List[TrackFrame]],
                            details: Dict[str, Dict[str, object]],
                            alert: Alert) -> str:
    """Rename a track to Muse's confirmed species everywhere it is shown.

    `elk-0` confirmed as white-tailed deer becomes `deer-0`, labelled deer.
    SAM's word survives only in the alert's `detected_as`. Returns the
    (possibly unchanged) track id.
    """
    if effective_species(alert) == alert.detected_as:
        return alert.track_id
    key = species_height_key(alert.muse_species)
    if key is None:
        return alert.track_id
    old_id = alert.track_id
    suffix = old_id.rsplit("-", 1)[-1] if "-" in old_id else "0"
    new_id = "%s-%s" % (key, suffix)
    if new_id == old_id:
        return old_id
    tfs = track_frames.pop(old_id, [])
    for tf in tfs:
        tf.det.track_id = new_id
        tf.det.label = key
        tf.det.obj_id = suffix
    track_frames[new_id] = tfs
    if old_id in details:
        details[new_id] = details.pop(old_id)
    alert.track_id = new_id
    return new_id


def group_text(labels: Sequence[str]) -> str:
    """'3 DEER' for one species, '2 ANIMALS' for mixed, '' under two."""
    if len(labels) < 2:
        return ""
    if len(set(labels)) == 1:
        return "%d %s" % (len(labels), SPECIES_PLURAL.get(labels[0],
                                                          labels[0].upper() + "S"))
    return "%d ANIMALS" % len(labels)


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return "%02d:%04.1f" % (minutes, rest)


def point_in_polygon(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if (y1 > y) != (y2 > y):
            cross_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < cross_x:
                inside = not inside
    return inside


def focal_px(frame_h: float) -> float:
    return (frame_h / 2.0) / math.tan(math.radians(config.CAMERA_VFOV_DEG / 2.0))


def species_height_key(species: str) -> Optional[str]:
    """The SPECIES_HEIGHT_M key a species name contains ('white-tailed deer' -> 'deer')."""
    text = (species or "").lower()
    if text in config.SPECIES_HEIGHT_M:
        return text
    for key in config.SPECIES_HEIGHT_M:
        if key in text:
            return key
    return None


def estimate_distance_m(label: str, box_h_px: float, frame_h: float) -> Optional[float]:
    """Approximate distance from box height and a rough standing height."""
    if box_h_px <= 0 or frame_h <= 0:
        return None
    key = species_height_key(label)
    height_m = config.SPECIES_HEIGHT_M.get(key, 1.5) if key else 1.5
    return height_m * focal_px(frame_h) / box_h_px


def recompute_track_distances(track_frames_list: List[TrackFrame],
                              alert: Alert) -> bool:
    """Recompute a track's distance in every frame with its final species.

    When Muse confirms a species, distances use that species' height —
    including frames before the alert. Returns True when recomputed.
    """
    if effective_species(alert) == alert.detected_as:
        return False
    key = species_height_key(alert.muse_species)
    if key is None:
        return False
    for tf in track_frames_list:
        tf.distance_m = estimate_distance_m(key, tf.det.box_h, tf.det.frame_h)
        if tf.det.frame == alert.frame:
            alert.distance_at_alert = tf.distance_m
    return True


def estimate_ttc(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Time to reach from box heights: h / (dh/dt), fitted over the window.

    `points` are (time_s, box_height_px). Returns None unless the box is
    growing steadily (positive slope and a good linear fit).
    """
    if len(points) < 3:
        return None
    n = float(len(points))
    mean_t = sum(p[0] for p in points) / n
    mean_h = sum(p[1] for p in points) / n
    s_tt = sum((p[0] - mean_t) ** 2 for p in points)
    if s_tt <= 0:
        return None
    slope = sum((p[0] - mean_t) * (p[1] - mean_h) for p in points) / s_tt
    if slope <= 0:
        return None
    ss_res = sum((p[1] - (mean_h + slope * (p[0] - mean_t))) ** 2 for p in points)
    ss_tot = sum((p[1] - mean_h) ** 2 for p in points)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    if r2 < 0.7:
        return None
    return points[-1][1] / slope


class StateSmoother:
    """Up a level at once; down only after the lower level held for hold_s."""

    def __init__(self, hold_s: float = 1.0) -> None:
        self.hold_s = hold_s
        self.state = "green"
        self._pending: Optional[str] = None
        self._pending_since = 0.0

    def update(self, raw_state: str, time_s: float) -> str:
        if LEVEL_ORDER[raw_state] >= LEVEL_ORDER[self.state]:
            self.state = raw_state
            self._pending = None
        else:
            if self._pending != raw_state:
                self._pending, self._pending_since = raw_state, time_s
            elif time_s - self._pending_since >= self.hold_s - 1e-9:
                self.state = raw_state
                self._pending = None
        return self.state


def parse_muse_reply(text: str) -> Dict[str, object]:
    """First {...} object in the reply; safe defaults when parsing fails."""
    result: Dict[str, object] = {
        "is_animal": None, "species": "", "confidence": None,
        "heading_toward_road": "unclear", "sign_text": config.DEFAULT_SIGN_TEXT,
    }
    if not text:
        return result
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if match is None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return result
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return result
    if not isinstance(data, dict):
        return result
    if isinstance(data.get("is_animal"), bool):
        result["is_animal"] = data["is_animal"]
    if isinstance(data.get("species"), str) and data["species"].strip():
        result["species"] = data["species"].strip()
    try:
        if data.get("confidence") is not None:
            result["confidence"] = float(data["confidence"])
    except (TypeError, ValueError):
        pass
    if data.get("heading_toward_road") in ("yes", "no", "unclear"):
        result["heading_toward_road"] = data["heading_toward_road"]
    if isinstance(data.get("sign_text"), str) and data["sign_text"].strip():
        result["sign_text"] = data["sign_text"].strip().upper()
    return result


def expected_level(video_name: str) -> str:
    """Highest level a clip should reach, from its dashcam file name."""
    name = video_name.lower()
    if "no_animal" in name:
        return "green"
    if "beside_road" in name:
        return "amber"
    return "red"  # crossing / on_road


def should_alert_red(video_name: str) -> bool:
    return expected_level(video_name) == "red"


def verdict_for(video_name: str, alert_fired: bool) -> str:
    should = should_alert_red(video_name)
    if should and alert_fired:
        return "correct alert"
    if should and not alert_fired:
        return "missed"
    if not should and alert_fired:
        return "false alarm"
    return "correct silence"


def level_verdict_ok(video_name: str, highest_level: str, alert_fired: bool) -> bool:
    """Tick/cross for the results screen, using the level the clip reached."""
    want = expected_level(video_name)
    if want == "red":
        return bool(alert_fired)
    if want == "amber":
        return highest_level == "amber"
    return highest_level == "green"


def describe_clip(video_name: str) -> str:
    stem = Path(video_name).stem.lower()
    if "no_animal" in stem:
        return "Normal traffic, no animals"
    if "beside_road" in stem:
        return "Animal beside the road"
    if "crossing" in stem:
        return "Animal crossing the road"
    for word, label in (("elephant", "Elephants"), ("moose", "Moose"),
                        ("deer", "Deer"), ("elk", "Elk"), ("cow", "Cows")):
        if word in stem:
            return "%s on the road" % label
    words = [w for w in re.split(r"[_\-]+", stem) if w and not w.isdigit()]
    return " ".join(words).capitalize() or stem


def log_api_call(model: str, endpoint: str, input_desc: str,
                 response_id: str, duration_ms: float,
                 est_cost_usd: Optional[float], extra: str = "") -> None:
    """Append one proof-of-use line to output/api_calls.log (never the key)."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        cost = "" if est_cost_usd is None else "%.6f" % est_cost_usd
        line = ("%s model=%s endpoint=%s input=%s response_id=%s duration_ms=%d "
                "est_cost_usd=%s %s\n" % (stamp, model, endpoint, input_desc,
                                          response_id or "-", int(duration_ms),
                                          cost, extra)).strip() + "\n"
        with open(OUTPUT_DIR / "api_calls.log", "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass  # logging must never break a run


# ------------------------------------------------------------- SAM parsing --

def _sam_events_for_text(text: str):
    lane = {"item_id": "msg_saved", "output_index": 0, "content_index": 0}

    async def _gen():
        yield dict(type="response.output_text.delta", delta=text, **lane)
        yield dict(type="response.output_text.done", text=text, **lane)
        yield {"type": "response.completed", "response": {}}

    return _gen()


async def _parse_sam_async(text: str, word: str, media: str) -> List[Detection]:
    from meta_sam_parser import (
        SegmentationBoxRecord, SegmentationMaskRecord, decode_mask_to_raster,
        image_segmentation_format, parse_responses_stream,
        video_segmentation_format,
    )

    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    parsed = parse_responses_stream(_sam_events_for_text(text), fmt)
    result = await parsed.final_result()
    boxes: Dict[Tuple[int, str], object] = {}
    masks: Dict[Tuple[int, str], object] = {}
    for record in result.records:
        frame_index = record.frame.frame_index if record.frame is not None else 0
        key = (frame_index, str(record.object_id))
        if isinstance(record, SegmentationBoxRecord):
            boxes[key] = record
        elif isinstance(record, SegmentationMaskRecord):
            masks[key] = record
    detections: List[Detection] = []
    for (frame_index, obj_id), box in sorted(boxes.items()):
        confidence = box.confidence if box.confidence is not None else 0.0
        det = Detection(
            frame=frame_index, time_s=frame_index / config.PROCESS_FPS,
            track_id="%s-%s" % (word, obj_id), label=word, obj_id=str(obj_id),
            confidence=float(confidence), x1=int(box.left), y1=int(box.top),
            x2=int(box.right) - 1, y2=int(box.bottom) - 1, frame_w=0, frame_h=0)
        mask_record = masks.get((frame_index, obj_id))
        if mask_record is not None:
            det.mask = bytes(decode_mask_to_raster(mask_record.mask))
            det.mask_w = int(mask_record.mask.width)
            det.mask_h = int(mask_record.mask.height)
        detections.append(det)
    return detections


def parse_sam_output(text: str, word: str, fps: Optional[float] = None) -> List[Detection]:
    """Parse one saved/streamed SAM 3.1 video response for `word`."""
    if not text or not text.strip():
        return []
    try:
        detections = asyncio.run(_parse_sam_async(text, word, "video"))
    except ImportError:
        raise UserError("The SAM software is not installed - run: pip install -r requirements.txt (Python 3.10 or later), then try again.")
    if fps:
        for det in detections:
            det.time_s = det.frame / float(fps)
    return detections


def parse_sam_image_output(text: str, word: str) -> List[Detection]:
    """Parse one SAM 3.1 image response (frame 0) for `word`."""
    if not text or not text.strip():
        return []
    try:
        return asyncio.run(_parse_sam_async(text, word, "image"))
    except ImportError:
        raise UserError("The SAM software is not installed - run: pip install -r requirements.txt (Python 3.10 or later), then try again.")


def group_tracks(detections: Sequence[Detection]) -> Dict[str, List[Detection]]:
    tracks: Dict[str, List[Detection]] = {}
    for det in detections:
        tracks.setdefault(det.track_id, []).append(det)
    for track_id in tracks:
        tracks[track_id].sort(key=lambda d: (d.frame, d.time_s))
    return tracks


def _det_box(det: Detection):
    return (det.x1, det.y1, det.x2, det.y2)


def _relabel(det: Detection, track_id: str, label: str) -> Detection:
    return Detection(frame=det.frame, time_s=det.time_s, track_id=track_id,
                     label=label, obj_id=track_id.rsplit("-", 1)[-1],
                     confidence=det.confidence, x1=det.x1, y1=det.y1,
                     x2=det.x2, y2=det.y2, frame_w=det.frame_w,
                     frame_h=det.frame_h, on_road=det.on_road, mask=det.mask,
                     mask_w=det.mask_w, mask_h=det.mask_h)


def merge_duplicate_tracks(tracks: Dict[str, List[Detection]]):
    """One animal found by two words -> one track keeping every frame.

    Two tracks from different words merge when their boxes overlap with
    IoU >= DUPLICATE_IOU in at least half the frames they share. The merged
    track keeps every frame either word found the animal in (the more
    confident box in shared frames), starts at the earliest detection, and
    is labelled with the word that has the higher mean confidence.
    Returns (tracks, notes) where notes maps kept track_id -> other word.
    """
    notes: Dict[str, str] = {}
    result: Dict[str, List[Detection]] = dict(tracks)
    ids = sorted(tracks)
    dropped: set = set()
    mean_conf = {tid: (sum(d.confidence for d in dets) / len(dets) if dets else 0.0)
                 for tid, dets in tracks.items()}
    for i, first_id in enumerate(ids):
        if first_id in dropped:
            continue
        for second_id in ids[i + 1:]:
            if second_id in dropped or first_id in dropped:
                continue
            first = result.get(first_id) or []
            second = result.get(second_id) or []
            if not first or not second or first[0].label == second[0].label:
                continue
            by_first = {d.frame: d for d in first}
            by_second = {d.frame: d for d in second}
            shared = set(by_first) & set(by_second)
            if not shared:
                continue
            overlap = sum(1 for f in shared
                          if box_iou(_det_box(by_first[f]), _det_box(by_second[f]))
                          >= config.DUPLICATE_IOU)
            if overlap * 2 < len(shared):
                continue
            keep, drop = (first_id, second_id) if mean_conf[first_id] >= mean_conf[second_id] \
                else (second_id, first_id)
            keep_label = result[keep][0].label
            merged: List[Detection] = []
            for frame in sorted(set(by_first) | set(by_second)):
                if frame in by_first and frame in by_second:
                    chosen = by_first[frame] if by_first[frame].confidence >= by_second[frame].confidence \
                        else by_second[frame]
                elif frame in by_first:
                    chosen = by_first[frame]
                else:
                    chosen = by_second[frame]
                merged.append(_relabel(chosen, keep, keep_label))
            result[keep] = merged
            dropped.add(drop)
            notes.setdefault(keep, result[drop][0].label if result.get(drop) else "")
            del result[drop]
    return result, notes


def apply_group_hold(raw_levels: Sequence[str], near_flags: Sequence[bool],
                     fps: float, hold_s: Optional[float] = None):
    """Hold amber-or-above for GROUP_HOLD_SECONDS after animals leave.

    Returns (levels, hold_flags). The 1 s step-down smoother applies on top
    of these levels at render time.
    """
    hold = config.GROUP_HOLD_SECONDS if hold_s is None else float(hold_s)
    levels: List[str] = []
    holds: List[bool] = []
    last_near: Optional[float] = None
    warned = False
    for index, raw in enumerate(raw_levels):
        now = index / float(fps)
        if index < len(near_flags) and near_flags[index]:
            last_near = now
        if raw in ("amber", "red"):
            warned = True
        active = bool(warned and last_near is not None
                      and not (index < len(near_flags) and near_flags[index])
                      and now - last_near <= hold + 1e-9)
        holds.append(active)
        if active and raw == "green":
            levels.append("amber")
        else:
            levels.append(raw)
    return levels, holds


def api_seconds_for(clip_stem: str, log_path: Optional[Path] = None) -> float:
    """Summed API-call seconds logged for one clip (cache-first durations)."""
    path = log_path or (OUTPUT_DIR / "api_calls.log")
    if not path.exists():
        return 0.0
    total_ms = 0.0
    needle = "clip=%s" % clip_stem
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if needle not in line:
            continue
        match = re.search(r"duration_ms=(\d+)", line)
        if match:
            total_ms += float(match.group(1))
    return total_ms / 1000.0


# ------------------------------------------------------------- API helpers --

def require_api_key() -> str:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise UserError(
            "MODEL_API_KEY is not set - set it with: export MODEL_API_KEY=your-key "
            "(Mac Terminal) or set MODEL_API_KEY=your-key (Windows), then run again.")
    return key


def friendly_api_error(exc: Exception) -> UserError:
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return UserError("Your API key was rejected - check MODEL_API_KEY and billing, then try again.")
    if status == 404:
        return UserError("The model was not found - check the model names in config.py, then try again.")
    if status == 429:
        return UserError("The API is rate-limiting requests - wait a minute, then try again.")
    if status == 400:
        return UserError("The API rejected the request - check the video is MP4 or MOV, then try again.")
    if isinstance(status, int) and status >= 500:
        return UserError("The API is having trouble - wait a minute, then try again.")
    return UserError("Something went wrong calling the API - check your connection, then try again.")


def _status_of(exc: Exception) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def call_with_retry(func, *args, **kwargs):
    """Retry 429/5xx up to 3 times with backoff; never retry a 400."""
    delay = 1.0
    for attempt in range(4):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            status = _status_of(exc)
            if status != 429 and not (status is not None and status >= 500):
                raise
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2


async def _call_with_retry_async(coro_factory):
    delay = 1.0
    for attempt in range(4):
        try:
            return await coro_factory()
        except Exception as exc:
            status = _status_of(exc)
            if status != 429 and not (status is not None and status >= 500):
                raise
            if attempt == 3:
                raise
            await asyncio.sleep(delay)
            delay *= 2


def make_clients():
    """(sync, async) OpenAI clients for the Meta Model API. Key never logged."""
    key = require_api_key()
    try:
        from openai import AsyncOpenAI, OpenAI
    except ImportError:
        raise UserError("The app software is not installed - run: pip install -r requirements.txt, then try again.")
    return (OpenAI(base_url=config.API_BASE_URL, api_key=key),
            AsyncOpenAI(base_url=config.API_BASE_URL, api_key=key))


def resolve_sam_model(sync_client, clip_stem: str = "") -> str:
    started = time.monotonic()
    try:
        models = call_with_retry(sync_client.models.list)
    except Exception as exc:
        raise friendly_api_error(exc)
    log_api_call(config.SAM_MODEL, "models.list", "models clip=%s" % clip_stem, "",
                 (time.monotonic() - started) * 1000, 0.0)
    ids = [m.id for m in models.data]
    if config.SAM_MODEL in ids:
        return config.SAM_MODEL
    for model_id in ids:
        squashed = model_id.lower().replace("-", "").replace(".", "").replace("_", "")
        if "sam" in squashed and "31" in squashed:
            return model_id
    raise UserError("SAM 3.1 was not found for this API key - check billing, then try again.")


async def _tap_ids(events, holder: Dict[str, str]):
    async for event in events:
        response = getattr(event, "response", None)
        rid = getattr(response, "id", None) if response is not None else None
        if rid:
            holder["id"] = str(rid)
        yield event


async def _stream_sam(async_client, model: str, content: list,
                      media_format: str, input_desc: str,
                      est_cost: float) -> Tuple[str, str]:
    """One streamed SAM 3.1 call; returns (raw_text, response_id); logs it."""
    from meta_sam_parser import (
        image_segmentation_format, parse_responses_stream,
        video_segmentation_format,
    )

    started = time.monotonic()
    events = await async_client.responses.create(
        model=model,
        input=[{"type": "message", "role": "user", "content": content}],
        stream=True, metadata={"include_confidence": "true"})
    holder: Dict[str, str] = {}
    fmt = (video_segmentation_format() if media_format == "video"
           else image_segmentation_format())
    parsed = parse_responses_stream(_tap_ids(events, holder), fmt)
    result = await parsed.final_result()
    try:
        await events.close()
    except Exception:
        pass
    duration_ms = (time.monotonic() - started) * 1000
    log_api_call(model, "responses.create", input_desc, holder.get("id", ""),
                 duration_ms, est_cost)
    return result.raw_output or "", holder.get("id", "")


def call_sam_video(async_client, model: str, file_id: str, word: str,
                   n_frames: int, clip_stem: str = "") -> str:
    billed = max(50, n_frames)
    cost = billed / 1000.0 * 0.20
    try:
        text, _rid = asyncio.run(_call_with_retry_async(lambda: _stream_sam(
            async_client, model,
            [{"type": "input_text", "text": word},
             {"type": "input_video", "file_id": file_id}],
            "video", "video:%s:%dframes clip=%s" % (word, n_frames, clip_stem),
            cost)))
        return text
    except UserError:
        raise
    except Exception as exc:
        raise friendly_api_error(exc)


async def call_sam_image_async(async_client, model: str, data_url: str,
                               word: str) -> str:
    try:
        text, _rid = await _call_with_retry_async(lambda: _stream_sam(
            async_client, model,
            [{"type": "input_text", "text": word},
             {"type": "input_image", "image_url": data_url}],
            "image", "image:%s:1image" % word, 2.50 / 1000.0))
        return text
    except Exception as exc:
        raise friendly_api_error(exc)


def encode_jpeg_data_url(frame_bgr) -> str:
    import cv2

    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise UserError("Could not read a video frame - check the video file, then try again.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def image_file_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return "data:%s;base64,%s" % (
        mime, base64.b64encode(path.read_bytes()).decode("ascii"))


def check_alert_with_muse(sync_client, alert: Alert, det: Detection,
                          frame_bgr, road_mask, clip_stem: str = "") -> None:
    """One Muse call for one alert (never per frame). Fills the alert."""
    import cv2

    crop = crop_with_padding(frame_bgr, det)
    full = frame_bgr.copy()
    draw_road_outline(full, road_mask)
    cv2.rectangle(full, (det.x1, det.y1), (det.x2, det.y2), (0, 0, 255), 2)
    started = time.monotonic()
    try:
        reply = call_with_retry(
            sync_client.chat.completions.create,
            model=config.MUSE_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": config.MUSE_PROMPT},
                {"type": "image_url", "image_url": {"url": encode_jpeg_data_url(crop)}},
                {"type": "image_url", "image_url": {"url": encode_jpeg_data_url(full)}},
            ]}],
        )
    except Exception as exc:
        raise friendly_api_error(exc)
    alert.muse_latency_s = time.monotonic() - started
    usage = getattr(reply, "usage", None)
    tokens = ""
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    if usage is not None:
        tokens = "prompt_tokens=%s completion_tokens=%s" % (
            prompt_tokens, completion_tokens)
    alert.muse_cost_usd = muse_cost_usd(config.MUSE_MODEL, prompt_tokens,
                                        completion_tokens)
    log_api_call(config.MUSE_MODEL, "chat.completions.create",
                 "images:2 alert:%s clip=%s" % (alert.track_id, clip_stem),
                 str(getattr(reply, "id", "") or ""),
                 alert.muse_latency_s * 1000, alert.muse_cost_usd, tokens)
    text = reply.choices[0].message.content or ""
    parsed = parse_muse_reply(text)
    alert.muse_species = str(parsed["species"] or "")
    alert.muse_confidence = parsed["confidence"]  # type: ignore[assignment]
    alert.heading_toward_road = str(parsed["heading_toward_road"] or "unclear")
    alert.sign_text = str(parsed["sign_text"] or config.DEFAULT_SIGN_TEXT)
    alert.muse_rejected = parsed["is_animal"] is False


# ------------------------------------------------------------ video frames --

def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise UserError("ffmpeg is not installed - install it (Mac: brew install ffmpeg; Windows: winget install ffmpeg), then try again.")
    return path


def prepare_video(source: Path, prepared: Path) -> None:
    ffmpeg = require_ffmpeg()
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UserError("That video format is not supported - please use an MP4 or MOV file.")
    if not source.exists():
        raise UserError("That video file was not found - check the file name, then try again.")
    prepared.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(source), "-t", str(config.MAX_SECONDS),
           "-vf", "fps=%s,scale='min(%d,iw)':-2" % (config.PROCESS_FPS, config.MAX_WIDTH),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
           "-movflags", "+faststart", str(prepared)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0 or not prepared.exists():
        raise UserError("Could not read that video - please use an MP4 or MOV file.")


def prepare_display_video(source: Path, display: Path) -> None:
    """Full-resolution display copy: same frames, drawn on, never sent."""
    ffmpeg = require_ffmpeg()
    display.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(source), "-t", str(config.MAX_SECONDS),
           "-vf", "fps=%s,scale='min(%d,iw)':-2" % (config.PROCESS_FPS,
                                                    config.DISPLAY_MAX_WIDTH),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
           "-movflags", "+faststart", str(display)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0 or not display.exists():
        raise UserError("Could not read that video - please use an MP4 or MOV file.")


def scale_detection(det: Detection, from_w: int, from_h: int,
                    to_w: int, to_h: int) -> Detection:
    """Move a box from analysis-copy pixels to display-copy pixels."""
    if (from_w, from_h) == (to_w, to_h) or from_w <= 0 or from_h <= 0:
        det.frame_w, det.frame_h = to_w, to_h
        return det
    sx, sy = to_w / float(from_w), to_h / float(from_h)
    det.x1 = min(to_w - 1, max(0, int(round(det.x1 * sx))))
    det.x2 = min(to_w - 1, max(0, int(round(det.x2 * sx))))
    det.y1 = min(to_h - 1, max(0, int(round(det.y1 * sy))))
    det.y2 = min(to_h - 1, max(0, int(round(det.y2 * sy))))
    det.frame_w, det.frame_h = to_w, to_h
    return det


def video_size(video: Path) -> Tuple[int, int, int]:
    """(frame count, width, height) without decoding every frame."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return count, width, height


def load_frames(video: Path):
    import cv2

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise UserError("Could not read any frames from that video - please use an MP4 or MOV file.")
    height, width = frames[0].shape[:2]
    return frames, width, height


def crop_with_padding(frame_bgr, det: Detection, pad_ratio: float = 0.25):
    height, width = frame_bgr.shape[:2]
    box_w, box_h = det.x2 - det.x1 + 1, det.y2 - det.y1 + 1
    pad_x, pad_y = int(box_w * pad_ratio), int(box_h * pad_ratio)
    x1 = max(0, det.x1 - pad_x)
    y1 = max(0, det.y1 - pad_y)
    x2 = min(width - 1, det.x2 + pad_x)
    y2 = min(height - 1, det.y2 + pad_y)
    return frame_bgr[y1:y2 + 1, x1:x2 + 1].copy()


def mask_canvas(det: Detection, width: int, height: int):
    """Full-frame bool mask for one detection (raster placed at its box)."""
    import numpy as np

    canvas = np.zeros((height, width), dtype=bool)
    if det.mask is None or det.mask_w <= 0 or det.mask_h <= 0:
        canvas[det.y1:det.y2 + 1, det.x1:det.x2 + 1] = True
        return canvas
    import cv2

    raster = np.frombuffer(det.mask, dtype=np.uint8).reshape(det.mask_h, det.mask_w)
    box_w, box_h = det.x2 - det.x1 + 1, det.y2 - det.y1 + 1
    if (det.mask_w, det.mask_h) != (box_w, box_h):
        raster = cv2.resize(raster, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
    x1, y1 = min(max(det.x1, 0), width - 1), min(max(det.y1, 0), height - 1)
    x2, y2 = min(max(det.x2, 0), width - 1), min(max(det.y2, 0), height - 1)
    canvas[y1:y2 + 1, x1:x2 + 1] = raster[:y2 - y1 + 1, :x2 - x1 + 1] > 0
    return canvas


def build_road_masks(road_dets: Sequence[Detection], n_frames: int,
                     width: int, height: int):
    """Largest road object per frame; reuse last; ROAD_ZONE if none at all."""
    import numpy as np

    by_frame: Dict[int, List[Detection]] = {}
    for det in road_dets:
        by_frame.setdefault(det.frame, []).append(det)
    masks: Dict[int, object] = {}
    last = None
    any_road = False
    for index in range(n_frames):
        dets = by_frame.get(index, [])
        if dets:
            best = max(dets, key=lambda d: (d.x2 - d.x1 + 1) * (d.y2 - d.y1 + 1))
            last = mask_canvas(best, width, height)
            any_road = True
        if last is not None:
            masks[index] = last
    if not any_road:
        import cv2

        zone = np.zeros((height, width), dtype=np.uint8)
        pts = np.array([[int(x * width), int(y * height)]
                        for x, y in config.ROAD_ZONE], dtype=np.int32)
        cv2.fillPoly(zone, [pts], 1)
        fallback = zone > 0
        return {index: fallback for index in range(n_frames)}, True
    first = next(iter(masks.values())) if masks else None
    for index in range(n_frames):
        masks.setdefault(index, first)
    return masks, False


def roadside_band(road_mask, width: int):
    import cv2
    import numpy as np

    margin = max(1, int(round(config.ROADSIDE_MARGIN * width)))
    kernel = np.ones((3, 2 * margin + 1), dtype=np.uint8)
    return cv2.dilate(road_mask.astype(np.uint8), kernel) > 0


def road_centre_x(road_mask, row: int, width: int) -> float:
    import numpy as np

    row = min(max(row, 0), road_mask.shape[0] - 1)
    xs = np.nonzero(road_mask[row])[0]
    if len(xs):
        return float(xs.mean())
    xs_all = np.nonzero(road_mask)[1]
    return float(xs_all.mean()) if len(xs_all) else width / 2.0


# --------------------------------------------------------- driving geometry --

def analyse_tracks(tracks: Dict[str, List[Detection]], road_masks: Dict[int, object],
                   n_frames: int, width: int, height: int,
                   alert_min_seconds: Optional[float] = None):
    """Per-frame where/toward/TTC/distance/level for every track, plus alerts.

    Returns (track_frames, alerts, raw_levels, details) where details maps
    track_id -> first-seen/highest-level record for reports and the panel.
    """
    bands: Dict[int, object] = {}

    def band_for(frame_index: int):
        if frame_index not in bands:
            bands[frame_index] = roadside_band(road_masks[frame_index], width)
        return bands[frame_index]

    min_on_road = (config.ALERT_MIN_SECONDS if alert_min_seconds is None
                   else float(alert_min_seconds))
    track_frames: Dict[str, List[TrackFrame]] = {}
    alerts: List[Alert] = []
    details: Dict[str, Dict[str, object]] = {}
    level_by_frame: Dict[int, str] = {}
    for track_id in sorted(tracks):
        dets = tracks[track_id]
        frames_list: List[TrackFrame] = []
        run_start: Optional[float] = None
        first_warn: Optional[float] = None
        highest = "green"
        fired = False
        for pos, det in enumerate(dets):
            road = road_masks.get(det.frame)
            if road is None:
                road = road_masks[min(road_masks)] if road_masks else None
            feet_x, feet_y = det.feet
            fx = min(max(int(feet_x), 0), width - 1)
            fy = min(max(int(feet_y), 0), height - 1)
            if road is not None and bool(road[fy, fx]):
                where = "on_road"
            elif road is not None and bool(band_for(det.frame)[fy, fx]):
                where = "near_road"
            else:
                where = "away"
            det.on_road = where == "on_road"
            # Sideways speed over the last 0.5 s, toward the road centre.
            toward = False
            if road is not None:
                then = None
                for prev in dets[:pos + 1]:
                    if det.time_s - prev.time_s <= 0.5 + 1e-9:
                        then = prev
                        break
                if then is not None and det.time_s - then.time_s > 0.1:
                    vx = (feet_x - then.feet[0]) / (det.time_s - then.time_s)
                    centre = road_centre_x(road, fy, width)
                    thresh = 0.02 * width
                    toward = ((feet_x < centre and vx > thresh)
                              or (feet_x > centre and vx < -thresh))
            window = [(d.time_s, float(d.box_h)) for d in dets[:pos + 1]
                      if det.time_s - d.time_s <= 1.0 + 1e-9]
            ttc = estimate_ttc(window)
            distance = estimate_distance_m(det.label, det.box_h, height)
            if where == "on_road":
                if run_start is None:
                    run_start = det.time_s
                duration = det.time_s - run_start
            else:
                run_start = None
                duration = 0.0
            if ((where == "on_road" and duration >= min_on_road - 1e-9)
                    or (where in ("near_road", "on_road") and toward
                        and ttc is not None and ttc <= config.RED_TTC_SECONDS)):
                level = "red"
            elif where in ("near_road", "on_road") or toward:
                level = "amber"
            else:
                level = "green"
            if level in ("amber", "red") and first_warn is None:
                first_warn = det.time_s
            if LEVEL_ORDER[level] > LEVEL_ORDER[highest]:
                highest = level
            frames_list.append(TrackFrame(det, where, toward, ttc, distance,
                                          level, duration))
            if level != "green":
                current = level_by_frame.get(det.frame, "green")
                if LEVEL_ORDER[level] > LEVEL_ORDER[current]:
                    level_by_frame[det.frame] = level
            if level == "red" and not fired:
                entry = run_start if where == "on_road" and run_start is not None else det.time_s
                alert = Alert(alert_no=len(alerts) + 1, time_s=det.time_s,
                              entry_time_s=entry, track_id=track_id,
                              detected_as=det.label, confidence=det.confidence,
                              frame=det.frame, first_warn_s=first_warn,
                              severity=severity_of(det.label),
                              ttc_at_alert=ttc, distance_at_alert=distance)
                if ttc is not None and first_warn is not None:
                    alert.lead_time_s = (det.time_s + ttc) - first_warn
                alerts.append(alert)
                fired = True
        track_frames[track_id] = frames_list
        if dets:
            first = dets[0]
            first_tf = frames_list[0]
            centre0 = (road_centre_x(road_masks[first.frame], int(first.feet[1]), width)
                       if first.frame in road_masks else width / 2.0)
            side = "on road" if first_tf.where == "on_road" else (
                "left roadside" if first.feet[0] < centre0 else "right roadside")
            details[track_id] = {
                "first_seen_s": first.time_s,
                "first_distance_m": first_tf.distance_m,
                "first_box_h_px": first.box_h, "side": side,
                "highest_level": highest}
    alerts.sort(key=lambda a: (a.time_s, a.track_id))
    for index, alert in enumerate(alerts):
        alert.alert_no = index + 1
    visible: Dict[int, int] = {}
    for tfs in track_frames.values():
        for tf in tfs:
            visible[tf.det.frame] = visible.get(tf.det.frame, 0) + 1
    for alert in alerts:
        alert.group_count = visible.get(alert.frame, 1)
        dets = tracks.get(alert.track_id) or []
        if dets:
            alert.mean_confidence = sum(d.confidence for d in dets) / len(dets)
            alert.tracked_s = dets[-1].time_s - dets[0].time_s
    raw_levels = [level_by_frame.get(i, "green") for i in range(n_frames)]
    return track_frames, alerts, raw_levels, details


# --------------------------------------------------------------- rendering --

PANEL_W, PANEL_H = 1920, 1080
COL_BG = (24, 17, 11)
COL_PANEL = (36, 26, 17)
COL_BORDER = (54, 42, 31)
COL_GREEN = LEVEL_COLORS["green"]
COL_AMBER = LEVEL_COLORS["amber"]
COL_RED = LEVEL_COLORS["red"]
COL_WHITE = (240, 240, 240)
COL_GREY = (150, 150, 150)
COL_YELLOW = (0, 255, 255)
GREY_TINT = (160, 160, 160)


def _text(img, text, org, scale, color, thickness=2):
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _caption_text(img, text, org, px=40, color=(255, 255, 255)):
    """Unicode caption text (for \u2248) via PIL; falls back to Hershey."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = None
        for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                          "/Library/Fonts/Arial.ttf",
                          "DejaVuSans.ttf", "Arial.ttf"):
            try:
                font = ImageFont.truetype(candidate, int(px))
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        rgb = Image.fromarray(img[:, :, ::-1].copy())
        draw = ImageDraw.Draw(rgb)
        draw.text((int(org[0]), int(org[1])), text, font=font,
                  fill=(int(color[2]), int(color[1]), int(color[0])))
        img[:, :, :] = rgb.asarray()[:, :, ::-1]
    except Exception:
        _text(img, text.replace("\u2248", "~"), (org[0], org[1] + int(px)),
              px / 36.0, color, 2)


def draw_road_outline(frame_bgr, road_mask, band_mask=None):
    """Thin yellow road edges; faint dashed roadside band. Never filled."""
    import cv2

    if road_mask is not None:
        contours, _ = cv2.findContours(road_mask.astype("uint8"),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                cv2.polylines(frame_bgr, [contour], True, COL_YELLOW, 2)
    if band_mask is not None:
        contours, _ = cv2.findContours(band_mask.astype("uint8"),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) <= 200:
                continue
            pts = contour.reshape(-1, 2)
            for i in range(0, len(pts) - 8, 16):
                p1 = tuple(pts[i])
                p2 = tuple(pts[min(i + 8, len(pts) - 1)])
                cv2.line(frame_bgr, p1, p2, (0, 200, 200), 1)
    return frame_bgr


def format_label(tf: TrackFrame, label_override: str = "") -> str:
    dist = "%.0f m" % tf.distance_m if tf.distance_m is not None else "- m"
    ttc = "%.1f s" % tf.ttc_s if tf.ttc_s is not None else "-"
    return "%s #%s - %s - %s" % (label_override or tf.det.label,
                                 tf.det.obj_id, dist, ttc)


def _segment_hits_rect(p1, p2, rect) -> bool:
    """True when segment p1-p2 touches the (x1, y1, x2, y2) rectangle."""
    x1, y1, x2, y2 = rect
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    t_lo, t_hi = 0.0, 1.0
    for p, q in ((-dx, p1[0] - x1), (dx, x2 - p1[0]),
                 (-dy, p1[1] - y1), (dy, y2 - p1[1])):
        if abs(p) < 1e-9:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t_lo = max(t_lo, t)
        else:
            t_hi = min(t_hi, t)
        if t_lo > t_hi:
            return False
    return True


def annotate_frame(frame_bgr, tfs: Sequence[TrackFrame], road_mask,
                   band_mask, level: str, sign_text: Optional[str], time_s: float,
                   group_label: str = "", hold_active: bool = False,
                   label_map: Optional[Dict[str, str]] = None):
    import cv2
    import numpy as np

    frame = frame_bgr.copy()
    height, width = frame.shape[:2]
    draw_road_outline(frame, road_mask, band_mask)
    far: Optional[TrackFrame] = None
    label_rects: List[Tuple[int, int, int, int]] = []
    for tf in tfs:
        det = tf.det
        color = {"green": GREY_TINT, "amber": COL_AMBER, "red": (0, 0, 255)}[tf.level]
        if det.mask is not None and det.mask_w > 0 and det.mask_h > 0:
            raster = np.frombuffer(det.mask, dtype=np.uint8).reshape(det.mask_h, det.mask_w)
            box_w, box_h = det.x2 - det.x1 + 1, det.y2 - det.y1 + 1
            if (det.mask_w, det.mask_h) != (box_w, box_h):
                raster = cv2.resize(raster, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
            x1 = min(max(det.x1, 0), width - 1)
            y1 = min(max(det.y1, 0), height - 1)
            x2 = min(max(det.x2, 0), width - 1)
            y2 = min(max(det.y2, 0), height - 1)
            region = frame[y1:y2 + 1, x1:x2 + 1]
            tint = np.zeros_like(region)
            tint[:] = color
            sel = raster[:region.shape[0], :region.shape[1]] > 0
            region[sel] = (region[sel] * 0.55 + tint[sel] * 0.45).astype(np.uint8)
            frame[y1:y2 + 1, x1:x2 + 1] = region
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        override = (label_map or {}).get(det.track_id, "")
        label = format_label(tf, override)
        (tw, th), _base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.85, 2)
        # Above the ring as well as the box, on a dark background.
        ring_top = det.y1
        if det.box_h < 0.06 * height:
            ring_top = min(ring_top, (det.y1 + det.y2) // 2
                           - int(max(det.x2 - det.x1, det.y2 - det.y1) * 0.8))
        ly = max(72, ring_top - th - 12)
        lx = min(max(4, det.x1), max(4, width - tw - 12))
        overlay = frame.copy()
        cv2.rectangle(overlay, (lx - 4, ly - 4), (lx + tw + 8, ly + th + 8),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        _text(frame, label, (lx, ly + th), 0.85, color, 2)
        label_rects.append((lx - 4, ly - 4, lx + tw + 8, ly + th + 8))
        if det.box_h < 0.06 * height and far is None:
            far = tf
    if far is not None:
        det = far.det
        cx, cy = (det.x1 + det.x2) // 2, (det.y1 + det.y2) // 2
        radius = max(18, int(max(det.x2 - det.x1, det.y2 - det.y1) * 0.8))
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)
        crop = crop_with_padding(frame_bgr, det, 0.6)
        zoom = cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4),
                          interpolation=cv2.INTER_LINEAR)
        max_w, max_h = 360, 270
        if zoom.shape[1] > max_w or zoom.shape[0] > max_h:
            s = min(max_w / zoom.shape[1], max_h / zoom.shape[0])
            zoom = cv2.resize(zoom, (int(zoom.shape[1] * s), int(zoom.shape[0] * s)))
        zx, zy = width - zoom.shape[1] - 16, 72
        frame[zy:zy + zoom.shape[0], zx:zx + zoom.shape[1]] = zoom
        cv2.rectangle(frame, (zx, zy), (zx + zoom.shape[1], zy + zoom.shape[0]),
                      (255, 255, 255), 2)
        # Connector from a ring point whose path to the inset crosses no label.
        target = (zx, zy + zoom.shape[0])
        start = (cx + radius, cy)
        for angle_deg in (0, 45, 90, 135, 180, 225, 270, 315):
            rad = math.radians(angle_deg)
            cand = (int(cx + radius * math.cos(rad)),
                    int(cy + radius * math.sin(rad)))
            if not any(_segment_hits_rect(cand, target, r) for r in label_rects):
                start = cand
                break
        cv2.line(frame, start, target, (255, 255, 255), 1)
        dist = "~%.0f m" % far.distance_m if far.distance_m is not None else ""
        inset_label = "%s %s" % ((label_map or {}).get(det.track_id) or det.label, dist)
        (iw, ih), _b = cv2.getTextSize(inset_label, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.75, 2)
        label_y = zy + zoom.shape[0] + 6
        overlay = frame.copy()
        cv2.rectangle(overlay, (zx - 4, label_y), (zx + iw + 8, label_y + ih + 8),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        _text(frame, inset_label, (zx, label_y + ih + 2), 0.75,
              (255, 255, 255), 2)
    if level == "red" and sign_text:
        cv2.rectangle(frame, (0, 0), (width, 64), (0, 0, 200), -1)
        banner = sign_text
        if group_label:
            banner = "%s - %s" % (group_label, banner)
        _text(frame, banner, (12, 46), 1.15, (255, 255, 255), 3)
    elif level == "amber":
        cv2.rectangle(frame, (0, 0), (width, 64), COL_AMBER, -1)
        banner = "WILDLIFE AHEAD"
        if group_label:
            banner = "%s - %s" % (group_label, banner)
        if hold_active:
            banner = "%s - MORE MAY FOLLOW" % banner
        _text(frame, banner, (12, 46), 1.15, (0, 0, 0), 3)
    stamp = format_time(time_s)
    _text(frame, stamp, (10, height - 12), 0.6, (0, 0, 0), 3)
    _text(frame, stamp, (10, height - 12), 0.6, (255, 255, 255), 1)
    return frame


def write_h264(frames_bgr, out_path: Path, fps: float, width: int, height: int) -> None:
    ffmpeg = require_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (width, height), "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for frame in frames_bgr:
            proc.stdin.write(frame.tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    if proc.returncode != 0 or not out_path.exists():
        raise UserError("Could not write the output video - check ffmpeg is installed, then try again.")


def build_events(track_frames: Dict[str, List[TrackFrame]],
                 alerts: Sequence[Alert],
                 details: Dict[str, Dict[str, object]]) -> List[Tuple[float, str]]:
    events: List[Tuple[float, str]] = []
    alert_by_track = {a.track_id: a for a in alerts}
    for track_id in sorted(track_frames):
        tfs = track_frames[track_id]
        if not tfs:
            continue
        info = details.get(track_id, {})
        first = tfs[0]
        dist = first.distance_m
        events.append((first.det.time_s, "%s first seen, ~%s m, %d px tall, %s" % (
            track_id, ("%.0f" % dist) if dist is not None else "-",
            first.det.box_h, info.get("side", "roadside"))))
        alert_for_word = alert_by_track.get(track_id)
        if alert_for_word is not None and alert_for_word.detected_as != first.det.label:
            # Renamed to Muse's species: SAM's word gets this one line only.
            events.append((first.det.time_s, "%s also found by the word %s" % (
                track_id, alert_for_word.detected_as)))
        elif info.get("also_seen_as"):
            events.append((first.det.time_s, "%s also found by the word %s" % (
                track_id, info["also_seen_as"])))
        toward_seen = False
        for prev, cur in zip(tfs, tfs[1:]):
            if cur.toward and not toward_seen:
                events.append((cur.det.time_s, "%s moving toward road" % track_id))
                toward_seen = True
            if cur.where == "on_road" and prev.where != "on_road":
                events.append((cur.det.time_s, "%s entered road" % track_id))
        alert = alert_by_track.get(track_id)
        if alert is not None:
            at = next((tf for tf in tfs if tf.det.frame == alert.frame), None)
            dist_a = alert.distance_at_alert
            ttc_a = alert.ttc_at_alert
            events.append((alert.time_s, "SLOW DOWN, ~%s m, %s to reach" % (
                ("%.0f" % dist_a) if dist_a is not None else "-",
                ("%.1f s" % ttc_a) if ttc_a is not None else "-")))
            if alert.muse_species and alert.muse_latency_s is not None:
                heading = {"yes": "heading toward road",
                           "no": "heading away from road"}.get(
                               alert.heading_toward_road, "heading unclear")
                conf = ("%.2f" % alert.muse_confidence) if alert.muse_confidence is not None else "-"
                events.append((alert.time_s + alert.muse_latency_s,
                               "Muse: %s, %s, %s" % (alert.muse_species, conf, heading)))
    events.sort(key=lambda item: (item[0], item[1]))
    return events


def render_panel_frame(camera, level: str, species: str, ttc_s, distance_m,
                       sign_text: str, log_lines: Sequence[str],
                       counters: Dict[str, str], video_name: str, time_s: float,
                       lead_info: Optional[Dict[str, float]],
                       hold_active: bool = False, alert_info: str = ""):
    import cv2
    import numpy as np

    canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    canvas[:] = COL_BG
    cam_h, cam_w = camera.shape[:2]
    box_x, box_y, box_w, box_h = 40, 90, 1220, 880
    scale = min(box_w / cam_w, box_h / cam_h)
    new_w, new_h = max(1, int(cam_w * scale)), max(1, int(cam_h * scale))
    resized = cv2.resize(camera, (new_w, new_h), interpolation=cv2.INTER_AREA)
    off_x = box_x + (box_w - new_w) // 2
    off_y = box_y + (box_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), COL_BORDER, 2)
    _text(canvas, "CAMERA 01 - %s" % video_name, (40, 60), 0.8, COL_WHITE, 2)
    _text(canvas, format_time(time_s), (40, 1010), 0.7, COL_GREY, 2)

    rx, rw = 1300, 580
    color = LEVEL_COLORS[level]
    cv2.rectangle(canvas, (rx, 90), (rx + rw, 240), color, -1)
    if level == "green":
        lines = ["ROAD CLEAR"]
    elif level == "amber":
        lines = ["MORE MAY FOLLOW" if hold_active else "WILDLIFE AHEAD", species]
    else:
        lines = ["SLOW DOWN", species]
    _text(canvas, lines[0], (rx + 24, 150), 1.15, (255, 255, 255), 3)
    if len(lines) > 1 and lines[1]:
        _text(canvas, lines[1], (rx + 24, 205), 0.75, (255, 255, 255), 2)
    if alert_info:
        _text(canvas, alert_info, (rx + 24, 232), 0.55, (255, 255, 255), 2)
    ttc_txt = "%.1f s" % ttc_s if ttc_s is not None else "-"
    dist_txt = "~%.0f m" % distance_m if distance_m is not None else "-"
    _text(canvas, "Time to reach: %s" % ttc_txt, (rx + 24, 292), 0.95, COL_WHITE, 3)
    _text(canvas, "Distance: %s" % dist_txt, (rx + 24, 338), 0.8, COL_WHITE, 2)

    cv2.rectangle(canvas, (rx, 356), (rx + rw, 470), (0, 0, 0), -1)
    cv2.rectangle(canvas, (rx, 356), (rx + rw, 470), COL_BORDER, 2)
    _text(canvas, "ROAD SIGN", (rx + 24, 388), 0.6, COL_GREY, 2)
    if sign_text:
        _text(canvas, sign_text.upper(), (rx + 24, 440), 0.85, COL_AMBER, 3)

    cv2.rectangle(canvas, (rx, 486), (rx + rw, 800), COL_PANEL, -1)
    cv2.rectangle(canvas, (rx, 486), (rx + rw, 800), COL_BORDER, 2)
    _text(canvas, "EVENT LOG", (rx + 24, 524), 0.65, COL_WHITE, 2)
    y = 556
    wrapped: List[str] = []
    for line in log_lines:
        wrapped.extend(textwrap.wrap(str(line), width=46) or [""])
    for line in wrapped[:9]:
        _text(canvas, line, (rx + 24, y), 0.52, COL_WHITE, 1)
        y += 27

    cv2.rectangle(canvas, (rx, 816), (rx + rw, 1032), COL_PANEL, -1)
    cv2.rectangle(canvas, (rx, 816), (rx + rw, 1032), COL_BORDER, 2)
    y = 850
    for key in ("Animals tracked", "Alerts raised", "Frames analysed", "Looking for"):
        _text(canvas, "%s: %s" % (key, counters.get(key, "")), (rx + 24, y), 0.6, COL_WHITE, 2)
        y += 28
    if lead_info:
        _text(canvas, "Lead time: %.1f s" % lead_info["lead_s"], (rx + 24, y + 6), 0.6, COL_WHITE, 2)
        # Timeline scaled from 1 s before first seen to 1 s after reach.
        t_start = float(lead_info["seen_s"]) - 1.0
        t_end = max(float(lead_info["reach_s"]) + 1.0, t_start + 0.1)
        x1 = rx + 24
        x2 = rx + rw - 24
        yy = y + 66

        def _tx(t: float) -> int:
            return x1 + int((x2 - x1) * (t - t_start) / (t_end - t_start))

        cv2.line(canvas, (x1, yy), (x2, yy), COL_GREY, 2)
        placed: List[Tuple[int, int, int]] = []  # (row, left, right)
        for key, label in (("seen_s", "first seen"), ("warn_s", "warning"),
                           ("reach_s", "reaches animal")):
            mx = _tx(float(lead_info[key]))
            cv2.circle(canvas, (mx, yy), 6, COL_AMBER, -1)
            (lw, _lh), _b = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            lx = min(max(x1, mx - lw // 2), x2 - lw)
            row = 0
            while any(r == row and lx < right + 6 and lx + lw + 6 > left
                      for r, left, right in placed):
                row += 1
            placed.append((row, lx, lx + lw))
            _text(canvas, label, (lx, yy - 30 + row * 20), 0.42, COL_WHITE, 1)

    _text(canvas, "Recorded footage - simulated road sign - SAM 3.1 + Muse Spark",
          (40, 1050), 0.55, COL_GREY, 1)
    return canvas


def render_demo_and_panel(frames, track_frames, alerts, raw_levels, road_masks,
                          video_name: str, prompts: Sequence[str],
                          details, out_dir: Path) -> None:
    alert_by_track = {a.track_id: a for a in alerts}
    label_map = {a.track_id: effective_species(a) for a in alerts}
    events = build_events(track_frames, alerts, details)
    smoother = StateSmoother(hold_s=1.0)
    seen_tracks: set = set()
    demo_frames, panel_frames = [], []
    width = frames[0].shape[1]
    lead_info = None
    if alerts:
        first_alert = alerts[0]
        info = details.get(first_alert.track_id, {})
        if first_alert.lead_time_s is not None and first_alert.ttc_at_alert is not None:
            lead_info = {"seen_s": float(info.get("first_seen_s") or 0.0),
                         "warn_s": float(first_alert.first_warn_s or first_alert.time_s),
                         "reach_s": first_alert.time_s + first_alert.ttc_at_alert,
                         "lead_s": float(first_alert.lead_time_s)}
    by_frame: Dict[int, List[TrackFrame]] = {}
    for tfs in track_frames.values():
        for tf in tfs:
            by_frame.setdefault(tf.det.frame, []).append(tf)
    near_flags = [any(tf.where in ("on_road", "near_road")
                      for tf in by_frame.get(i, [])) for i in range(len(frames))]
    display_levels, hold_flags = apply_group_hold(raw_levels, near_flags,
                                                  config.PROCESS_FPS)
    for index, frame in enumerate(frames):
        now = index / float(config.PROCESS_FPS)
        tfs = sorted(by_frame.get(index, []), key=lambda t: t.det.track_id)
        for tf in tfs:
            seen_tracks.add(tf.det.track_id)
        level = smoother.update(display_levels[index] if index < len(display_levels) else "green", now)
        hold_active = bool(hold_flags[index]) if index < len(hold_flags) else False
        group_label = group_text([tf.det.label for tf in tfs])
        road = road_masks.get(index)
        band = roadside_band(road, width) if road is not None else None
        sign_text = None
        if level == "red":
            for tf in tfs:
                if tf.det.track_id in alert_by_track and tf.level == "red":
                    sign_text = alert_by_track[tf.det.track_id].sign_text
                    break
            if sign_text is None and alerts:
                sign_text = alerts[0].sign_text
        panel_sign = ""
        if level == "red" and sign_text:
            panel_sign = sign_text
        elif level == "amber":
            panel_sign = "CAUTION - WILDLIFE AHEAD"
        annotated = annotate_frame(frame, tfs, road, band, level, sign_text, now,
                                   group_label=group_label, hold_active=hold_active,
                                   label_map=label_map)
        demo_frames.append(annotated)
        species, ttc_s, dist_m = "", None, None
        alert_info = ""
        for tf in tfs:
            if tf.level == level or (level == "amber" and tf.level in ("amber", "red")):
                alert = alert_by_track.get(tf.det.track_id)
                species = effective_species(alert) if alert is not None else tf.det.label
                ttc_s, dist_m = tf.ttc_s, tf.distance_m
                if alert is not None:
                    alert_info = "Severity: %s - Assurance: %s" % (
                        severity_text(alert.severity), alert.assurance)
                break
        if not species and tfs:
            species = tfs[0].det.label
            ttc_s, dist_m = tfs[0].ttc_s, tfs[0].distance_m
        if group_label:
            species = group_label.title() if not species else group_label
        log_lines = ["%s %s" % (format_time(t), text)
                     for t, text in reversed(events) if t <= now + 1e-9]
        counters = {
            "Animals tracked": str(len(seen_tracks)),
            "Alerts raised": str(sum(1 for a in alerts if a.time_s <= now + 1e-9)),
            "Frames analysed": str(index + 1),
            "Looking for": ", ".join(prompts),
        }
        panel_frames.append(render_panel_frame(
            annotated, level, species, ttc_s, dist_m, panel_sign, log_lines,
            counters, video_name, now, lead_info,
            hold_active=hold_active, alert_info=alert_info))
    height = frames[0].shape[0]
    write_h264(iter(demo_frames), out_dir / "demo.mp4", config.PROCESS_FPS, width, height)
    write_h264(iter(panel_frames), out_dir / "demo_panel.mp4",
               config.PROCESS_FPS, PANEL_W, PANEL_H)


# ---------------------------------------------------------------- pipeline --

def write_csv(path: Path, header: Sequence[str], rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def process_video(video_path, animal_prompts=None, road_zone=None,
                  alert_min_seconds=None, min_score=None, progress=None) -> Dict[str, object]:
    def note(message: str) -> None:
        if progress:
            progress(message)

    source = Path(video_path)
    prompts = [w.strip().lower() for w in (animal_prompts or config.ANIMAL_PROMPTS) if w.strip()]
    min_seconds = config.ALERT_MIN_SECONDS if alert_min_seconds is None else float(alert_min_seconds)
    threshold = config.MIN_SCORE if min_score is None else float(min_score)
    out_dir = OUTPUT_DIR / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = out_dir / "prepared.mp4"

    note("Preparing video...")
    prepare_video(source, prepared)
    display = out_dir / "display.mp4"
    prepare_display_video(source, display)
    _acount, analysis_w, analysis_h = video_size(prepared)
    frames, width, height = load_frames(display)
    n_frames = len(frames)
    digest = hashlib.sha256(prepared.read_bytes()).hexdigest()
    cache_file = out_dir / "cache.json"
    cache: Dict[str, str] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}

    words = list(prompts) + [config.ROAD_PROMPT]
    raw_texts: Dict[str, str] = {}
    missing: List[str] = []
    for word in words:
        raw_path = out_dir / ("sam_raw_%s.txt" % word)
        if raw_path.exists() and cache.get(word) == digest:
            raw_texts[word] = raw_path.read_text(encoding="utf-8")
        else:
            missing.append(word)

    sam_model = "cached"
    sam_calls = 0
    if missing:
        note("Calling SAM 3.1 for: %s" % ", ".join(missing))
        sync_client, async_client = make_clients()
        sam_model = resolve_sam_model(sync_client, clip_stem=source.stem)
        started = time.monotonic()
        try:
            with open(prepared, "rb") as handle:
                uploaded = call_with_retry(sync_client.files.create,
                                           file=handle, purpose="user_data")
        except Exception as exc:
            raise friendly_api_error(exc)
        log_api_call(sam_model, "files.create",
                     "video:%s clip=%s" % (source.name, source.stem),
                     str(getattr(uploaded, "id", "") or ""),
                     (time.monotonic() - started) * 1000, 0.0)
        for word in missing:
            raw_texts[word] = call_sam_video(async_client, sam_model,
                                             uploaded.id, word, n_frames,
                                             clip_stem=source.stem)
            sam_calls += 1
            (out_dir / ("sam_raw_%s.txt" % word)).write_text(raw_texts[word], encoding="utf-8")
            cache[word] = digest
        cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    note("Reading detections...")
    detections: List[Detection] = []
    for word in prompts:
        for det in parse_sam_output(raw_texts.get(word, ""), word, fps=config.PROCESS_FPS):
            scale_detection(det, analysis_w, analysis_h, width, height)
            if det.confidence >= threshold:
                detections.append(det)
    road_dets = parse_sam_output(raw_texts.get(config.ROAD_PROMPT, ""),
                                 config.ROAD_PROMPT, fps=config.PROCESS_FPS)
    for det in road_dets:
        scale_detection(det, analysis_w, analysis_h, width, height)
    tracks = group_tracks(detections)
    tracks, dupe_notes = merge_duplicate_tracks(tracks)
    road_masks, used_fallback = build_road_masks(road_dets, n_frames, width, height)
    track_frames, alerts, raw_levels, details = analyse_tracks(
        tracks, road_masks, n_frames, width, height,
        alert_min_seconds=min_seconds)
    for kept_id, other_word in dupe_notes.items():
        if kept_id in details:
            details[kept_id]["also_seen_as"] = other_word

    muse_calls = 0
    muse_confirmed: Dict[str, Optional[bool]] = {}
    if alerts and config.USE_MUSE_CHECK:
        note("Checking %d alert(s) with Muse..." % min(len(alerts), config.MAX_MUSE_CALLS))
        sync_client, _async = make_clients()
        by_key = {(tf.det.track_id, tf.det.frame): tf.det
                  for tfs in track_frames.values() for tf in tfs}
        for alert in alerts[:config.MAX_MUSE_CALLS]:
            det = by_key.get((alert.track_id, alert.frame))
            if det is None:
                continue
            frame = frames[min(alert.frame, n_frames - 1)]
            check_alert_with_muse(sync_client, alert, det, frame,
                                  road_masks.get(alert.frame),
                                  clip_stem=source.stem)
            muse_calls += 1
            apply_muse_species(alert)
            if recompute_track_distances(track_frames.get(alert.track_id, []), alert):
                tfs = track_frames.get(alert.track_id, [])
                if tfs and alert.track_id in details:
                    details[alert.track_id]["first_distance_m"] = tfs[0].distance_m
            old_id = alert.track_id
            new_id = rename_track_to_species(track_frames, details, alert)
            if new_id != old_id:
                for other in alerts:
                    if other.track_id == old_id:
                        other.track_id = new_id
            muse_confirmed[alert.track_id] = not alert.muse_rejected and bool(
                alert.muse_species or alert.muse_confidence is not None)
    for alert in alerts:
        called = alert.track_id in muse_confirmed
        confirmed = muse_confirmed.get(alert.track_id) if called else None
        # Muse confirmation counts only when it said this is an animal.
        if called:
            confirmed = confirmed and not alert.muse_rejected
        alert.assurance = assurance_level(alert.mean_confidence,
                                          alert.tracked_s, confirmed)

    flat = [tf.det for track_id in sorted(track_frames) for tf in track_frames[track_id]]
    write_csv(out_dir / "detections.csv",
              ["time_s", "frame", "track_id", "label", "confidence", "on_road",
               "x1", "y1", "x2", "y2"],
              [["%.3f" % d.time_s, d.frame, d.track_id, d.label, "%.4f" % d.confidence,
                str(bool(d.on_road)).lower(), d.x1, d.y1, d.x2, d.y2] for d in flat])
    write_csv(out_dir / "alerts.csv",
              ["alert_no", "time_s", "track_id", "detected_as", "confidence",
               "group_count", "severity", "assurance",
               "muse_species", "muse_confidence", "heading_toward_road",
               "sign_text", "muse_rejected"],
              [[a.alert_no, "%.3f" % a.time_s, a.track_id, a.detected_as,
                "%.4f" % a.confidence, a.group_count,
                severity_text(a.severity), a.assurance, a.muse_species,
                "" if a.muse_confidence is None else "%.4f" % a.muse_confidence,
                a.heading_toward_road, a.sign_text, str(bool(a.muse_rejected)).lower()]
               for a in alerts])

    note("Rendering videos...")
    render_demo_and_panel(frames, track_frames, alerts, raw_levels, road_masks,
                          source.name, prompts, details, out_dir)
    if "crossing_night" in source.stem.lower() and alerts:
        first_alert = alerts[0]
        track_info = details.get(first_alert.track_id, {})
        first_dist = track_info.get("first_distance_m")
        seen_s = track_info.get("first_seen_s")
        delay = (first_alert.time_s - float(seen_s)) if seen_s is not None else None
        seconds_left = (first_alert.distance_at_alert / (100.0 / 3.6)
                        if first_alert.distance_at_alert is not None else None)
        caption_text = showcase_caption(
            float(first_dist) if first_dist is not None else None, delay,
            seconds_left, physics_card()["stopping_dry_m"])
        write_clean_copies(display, out_dir / "demo.mp4", source,
                           first_alert.frame, caption_text, out_dir)
    ttr_info: Dict[str, object] = {}
    impact_s = load_impact_times().get(source.name)
    if "collision" in source.name.lower() and impact_s is not None:
        check = ttr_check(track_frames, alerts, float(impact_s))
        write_ttr_check_image(check, out_dir / "ttr_check.png")
        caption = "Flagged %s s before impact - estimate within %s%% of the real impact time" % (
            ("%.1f" % check["lead_time_s"]) if check["lead_time_s"] is not None else "-",
            ("%.0f" % check["mae_pct"]) if check["mae_pct"] is not None else "-")
        write_openings(display, out_dir / "demo.mp4", float(impact_s),
                       out_dir, caption)
        ttr_info = {"impact_s": float(impact_s), "mae_s": check["mae_s"],
                    "mae_pct": check["mae_pct"], "lead_time_s": check["lead_time_s"]}
    # Billed frames count every SAM request for this video (road included),
    # cached ones too: they were billed when first made. Never report 0
    # for a video SAM has processed.
    requests_total = len(words)
    frames_billed = requests_total * max(50, n_frames)
    sam_cost = frames_billed / 1000.0 * 0.20
    muse_cost = sum(a.muse_cost_usd or 0.0 for a in alerts)
    api_seconds = api_seconds_for(source.stem)
    video_seconds = n_frames / float(config.PROCESS_FPS)
    realtime_factor = (api_seconds / video_seconds) if video_seconds > 0 else 0.0
    highest = "green"
    for level in raw_levels:
        if LEVEL_ORDER[level] > LEVEL_ORDER[highest]:
            highest = level
    run_info = {
        "video": source.name, "settings": {
            "animal_prompts": prompts, "road_prompt": config.ROAD_PROMPT,
            "road_zone_fallback": used_fallback,
            "max_seconds": config.MAX_SECONDS, "process_fps": config.PROCESS_FPS,
            "max_width": config.MAX_WIDTH,
            "display_max_width": config.DISPLAY_MAX_WIDTH,
            "alert_min_seconds": min_seconds,
            "red_ttc_seconds": config.RED_TTC_SECONDS, "min_score": threshold,
            "use_muse_check": config.USE_MUSE_CHECK, "muse_model": config.MUSE_MODEL,
            "max_muse_calls": config.MAX_MUSE_CALLS},
        "sam_model": sam_model, "muse_model": config.MUSE_MODEL,
        "api_calls": sam_calls + muse_calls, "sam_calls": sam_calls,
        "sam_calls_total": requests_total, "muse_calls": muse_calls,
        "frames_analysed": n_frames, "frames_billed": frames_billed,
        "sam_cost_usd": round(sam_cost, 6),
        "muse_cost_usd": round(muse_cost, 6),
        "estimated_cost_usd": round(sam_cost + muse_cost, 6),
        "api_seconds": round(api_seconds, 3),
        "realtime_factor": round(realtime_factor, 3),
        "highest_level": highest,
        "tracks_detail": details,
        "alerts_detail": [{
            "track_id": a.track_id, "time_s": a.time_s,
            "first_warn_s": a.first_warn_s, "lead_time_s": a.lead_time_s,
            "ttc_at_alert": a.ttc_at_alert, "distance_at_alert": a.distance_at_alert,
            "group_count": a.group_count, "severity": a.severity,
            "severity_text": severity_text(a.severity), "assurance": a.assurance,
        } for a in alerts],
        "ttr_check": ttr_info,
    }
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    note("Done: %d animal(s), %d alert(s)." % (len(tracks), len(alerts)))
    return {"out_dir": out_dir, "tracks": tracks, "alerts": alerts,
            "details": details, "highest_level": highest,
            "frames_analysed": n_frames, "frames_billed": frames_billed,
            "sam_calls": sam_calls, "muse_calls": muse_calls,
            "api_seconds": run_info["api_seconds"],
            "realtime_factor": run_info["realtime_factor"],
            "estimated_cost_usd": run_info["estimated_cost_usd"]}


def run_batch(folder) -> List[Dict[str, object]]:
    folder = Path(folder)
    videos = sorted(p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not videos:
        raise UserError("No MP4 or MOV clips were found in that folder - add clips, then try again.")
    rows: List[Dict[str, object]] = []
    for video in videos:
        print("Running %s ..." % video.name)
        result = process_video(video)
        alerts = result["alerts"]  # type: ignore[assignment]
        first = alerts[0] if alerts else None
        species = ""
        if first is not None:
            species = first.muse_species or first.detected_as
        details = result["details"]  # type: ignore[assignment]
        first_seen = min((d["first_seen_s"] for d in details.values()), default=None)
        lead = first.lead_time_s if first is not None else None
        rows.append({
            "video": video.name, "animals_found": len(result["tracks"]),
            "alerts": len(alerts),
            "first_alert_s": "" if first is None else "%.3f" % first.time_s,
            "species": species, "frames_billed": result["frames_billed"],
            "frames_analysed": result["frames_analysed"],
            "alert_delay_s": "" if first is None else first.time_s - first.entry_time_s,
            "highest_level": result["highest_level"],
            "first_seen_s": "" if first_seen is None else "%.3f" % float(first_seen),
            "lead_time_s": "" if lead is None else "%.3f" % float(lead),
            "realtime_factor": result.get("realtime_factor", ""),
            "estimated_cost_usd": result.get("estimated_cost_usd", ""),
        })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "summary.csv",
              ["video", "animals_found", "alerts", "first_alert_s", "species",
               "frames_billed"],
              [[r["video"], r["animals_found"], r["alerts"], r["first_alert_s"],
                r["species"], r["frames_billed"]] for r in rows])
    write_results_image(rows)
    rebuild_sightings()
    return rows


def alert_delays_from_files(stem: str) -> List[float]:
    out_dir = OUTPUT_DIR / stem
    det_file, alert_file = out_dir / "detections.csv", out_dir / "alerts.csv"
    if not det_file.exists() or not alert_file.exists():
        return []
    with open(det_file, newline="", encoding="utf-8") as handle:
        dets = list(csv.DictReader(handle))
    with open(alert_file, newline="", encoding="utf-8") as handle:
        alert_rows = list(csv.DictReader(handle))
    delays: List[float] = []
    for alert_row in alert_rows:
        track_id = alert_row["track_id"]
        alert_time = float(alert_row["time_s"])
        entry: Optional[float] = None
        track_dets = sorted(
            (d for d in dets if d["track_id"] == track_id),
            key=lambda d: int(d["frame"]))
        for det in track_dets:
            on_road = str(det["on_road"]).lower() == "true"
            if on_road and entry is None:
                entry = float(det["time_s"])
            elif not on_road:
                entry = None
            if entry is not None and abs(float(det["time_s"]) - alert_time) < 1e-6:
                delays.append(alert_time - entry)
                break
    return delays


def collect_result_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    summary = OUTPUT_DIR / "summary.csv"
    if summary.exists():
        with open(summary, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    for row in rows:
        stem = Path(str(row["video"])).stem
        run_file = OUTPUT_DIR / stem / "run.json"
        row.setdefault("highest_level", "green")
        row.setdefault("first_seen_s", "")
        row.setdefault("lead_time_s", "")
        row["frames_analysed"] = ""
        row["alert_delay_s"] = ""
        if run_file.exists():
            try:
                info = json.loads(run_file.read_text(encoding="utf-8"))
                row["frames_analysed"] = info.get("frames_analysed", "")
                row["estimated_cost_usd"] = info.get("estimated_cost_usd", "")
                row["realtime_factor"] = info.get("realtime_factor", "")
                row["highest_level"] = info.get("highest_level", "green")
                det = info.get("tracks_detail", {}) or {}
                seen = [v.get("first_seen_s") for v in det.values()
                        if isinstance(v, dict) and v.get("first_seen_s") is not None]
                if seen:
                    row["first_seen_s"] = "%.3f" % min(float(s) for s in seen)
                alert_det = info.get("alerts_detail", []) or []
                leads = [a.get("lead_time_s") for a in alert_det
                         if isinstance(a, dict) and a.get("lead_time_s") is not None]
                if leads:
                    row["lead_time_s"] = "%.3f" % float(leads[0])
            except ValueError:
                pass
        alerts_file = OUTPUT_DIR / stem / "alerts.csv"
        if alerts_file.exists():
            with open(alerts_file, newline="", encoding="utf-8") as handle:
                row["alert_rows"] = list(csv.DictReader(handle))
        delays = alert_delays_from_files(stem)
        if delays:
            row["alert_delay_s"] = sum(delays) / len(delays)
    return rows


def result_stats(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    should = [r for r in rows if should_alert_red(str(r["video"]))]
    caught = [r for r in should if int(r.get("alerts") or 0) > 0]
    false_alarms = [r for r in rows
                    if not should_alert_red(str(r["video"])) and int(r.get("alerts") or 0) > 0]
    delays = [float(r["alert_delay_s"]) for r in rows
              if r.get("alert_delay_s") not in ("", None)]
    leads = [float(r["lead_time_s"]) for r in rows
             if r.get("lead_time_s") not in ("", None)]
    frames = sum(int(r["frames_analysed"]) for r in rows
                 if str(r.get("frames_analysed", "")).isdigit())
    billed = sum(int(r["frames_billed"]) for r in rows
                 if str(r.get("frames_billed", "")).isdigit())
    costs = [float(r["estimated_cost_usd"]) for r in rows
             if r.get("estimated_cost_usd") not in ("", None)]
    total_cost = round(sum(costs), 6) if costs else round(billed / 1000.0 * 0.20, 4)
    return {
        "catch_rate": (len(caught) / len(should)) if should else 0.0,
        "should_alert": len(should), "caught": len(caught),
        "false_alarms": len(false_alarms),
        "avg_delay_s": (sum(delays) / len(delays)) if delays else None,
        "avg_lead_s": (sum(leads) / len(leads)) if leads else None,
        "total_frames": frames, "total_billed": billed,
        "total_cost_usd": total_cost,
    }


def write_results_image(rows: Sequence[Dict[str, object]]) -> Path:
    import cv2
    import numpy as np

    stats = result_stats(rows)
    img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    img[:] = COL_BG
    _text(img, "Wildlife Road Alert - Results", (60, 90), 1.2, COL_WHITE, 3)
    y = 170
    for x, head in ((60, "Clip"), (700, "Expected"), (920, "Highest"),
                    (1080, "Alert"), (1200, "First alert"), (1400, "Species"),
                    (1640, "Verdict")):
        _text(img, head, (x, y), 0.65, COL_GREY, 2)
    y += 30
    cv2.line(img, (60, y), (1860, y), COL_BORDER, 2)
    for row in rows:
        y += 52
        video = str(row["video"])
        fired = int(row.get("alerts") or 0) > 0
        highest = str(row.get("highest_level") or "green")
        ok = level_verdict_ok(video, highest, fired)
        _text(img, video[:34], (60, y), 0.6, COL_WHITE, 2)
        _text(img, expected_level(video), (700, y), 0.6, COL_WHITE, 2)
        _text(img, highest, (920, y), 0.6, COL_WHITE, 2)
        _text(img, "yes" if fired else "no", (1080, y), 0.6, COL_WHITE, 2)
        _text(img, str(row.get("first_alert_s") or "-"), (1200, y), 0.6, COL_WHITE, 2)
        _text(img, str(row.get("species") or "-")[:16], (1400, y), 0.6, COL_WHITE, 2)
        if ok:
            cv2.line(img, (1640, y - 6), (1652, y + 6), COL_GREEN, 4)
            cv2.line(img, (1652, y + 6), (1672, y - 16), COL_GREEN, 4)
        else:
            cv2.line(img, (1640, y - 14), (1664, y + 10), COL_RED, 4)
            cv2.line(img, (1664, y - 14), (1640, y + 10), COL_RED, 4)
        _text(img, verdict_for(video, fired), (1690, y), 0.55,
              COL_GREEN if ok else COL_RED, 2)
    y += 80
    avg = stats["avg_delay_s"]
    lead = stats["avg_lead_s"]
    lines = [
        "Catch rate: %d/%d (%.0f%%)" % (stats["caught"], stats["should_alert"],
                                        100.0 * float(stats["catch_rate"])),
        "False alarms: %d" % stats["false_alarms"],
        "Average alert delay: %s" % (("%.2f s" % avg) if avg is not None else "n/a"),
        "Average lead time: %s" % (("%.2f s" % lead) if lead is not None else "n/a"),
        "Total frames analysed: %d" % stats["total_frames"],
        "Total frames billed: %d" % stats["total_billed"],
        "Total API cost (SAM + Muse): $%.4f" % float(stats["total_cost_usd"]),
    ]
    for line in lines:
        _text(img, line, (60, y), 0.72, COL_WHITE, 2)
        y += 42
    _text(img, "Recorded footage - simulated road sign - SAM 3.1 + Muse Spark",
          (60, 1030), 0.55, COL_GREY, 1)
    out = OUTPUT_DIR / "results.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), img):
        raise UserError("Could not write the results image - try again.")
    return out


def _title_card(text: str, seconds: float, path: Path) -> None:
    import numpy as np

    frames = int(round(seconds * config.PROCESS_FPS))
    img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    img[:] = COL_BG
    _text(img, text, (120, 540), 1.4, COL_WHITE, 3)
    write_h264(iter([img] * frames), path, config.PROCESS_FPS, PANEL_W, PANEL_H)


def _showcase_order(panel: Path) -> Tuple[int, int, str]:
    name = panel.parent.name.lower()
    if "no_animal" in name:
        group = 0
    elif "beside_road" in name:
        group = 1
    else:
        group = 2
    light = 2 if "night" in name else (1 if "dusk" in name else 0)
    return (group, light, panel.parent.name)


def build_showcase() -> Path:
    ffmpeg = require_ffmpeg()
    active = {p.stem for p in (BASE_DIR / "test_videos").glob("*.mp4")}
    panels = [p for p in sorted(OUTPUT_DIR.glob("*/demo_panel.mp4"))
              if p.parent.name in active]
    panels.sort(key=_showcase_order)
    if not panels:
        raise UserError("No demo panels were found - run: python app.py --batch test_videos/ first.")
    rows = collect_result_rows()
    if not (OUTPUT_DIR / "results.png").exists() and rows:
        write_results_image(rows)
    tmp = OUTPUT_DIR / "_showcase_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    parts: List[Path] = []
    for index, panel in enumerate(panels):
        card = tmp / ("card_%02d.mp4" % index)
        _title_card("Scene %d - %s" % (index + 1, describe_clip(panel.parent.name + ".mp4")),
                    2.0, card)
        parts.append(card)
        if "collision" in panel.parent.name.lower():
            parts.append(_collision_panel(panel, tmp / ("held_%02d.mp4" % index)))
        elif "crossing_night" in panel.parent.name.lower():
            parts.append(_night_panel(panel, tmp / ("held_%02d.mp4" % index)))
        else:
            parts.append(_held_panel(panel, tmp / ("held_%02d.mp4" % index)))
    results_png = OUTPUT_DIR / "results.png"
    if results_png.exists():
        import cv2

        img = cv2.imread(str(results_png))
        hold = tmp / "results_hold.mp4"
        write_h264(iter([img] * int(6 * config.PROCESS_FPS)), hold,
                   config.PROCESS_FPS, PANEL_W, PANEL_H)
        parts.append(hold)
    list_file = tmp / "list.txt"
    list_file.write_text("".join("file '%s'\n" % p.resolve() for p in parts),
                         encoding="utf-8")
    out = OUTPUT_DIR / "showcase.mp4"
    done = subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i",
                           str(list_file), "-c:v", "libx264", "-pix_fmt",
                           "yuv420p", "-movflags", "+faststart", str(out)],
                          capture_output=True, text=True)
    if done.returncode != 0 or not out.exists():
        raise UserError("Could not build the showcase reel - check ffmpeg is installed, then try again.")
    return out


def _held_panel(panel: Path, out_path: Path) -> Path:
    """Panel with a 1.5 s hold at the first sighting, caption included."""
    import cv2

    run_file = panel.parent / "run.json"
    info = {}
    if run_file.exists():
        try:
            info = json.loads(run_file.read_text(encoding="utf-8"))
        except ValueError:
            info = {}
    details = info.get("tracks_detail", {}) or {}
    if not details:
        return panel
    first_track = min(details.values(),
                      key=lambda v: float(v.get("first_seen_s") or 0))
    seen_s = float(first_track.get("first_seen_s") or 0)
    seen_frame = int(seen_s * config.PROCESS_FPS)
    cap = cv2.VideoCapture(str(panel))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames or seen_frame >= len(frames):
        return panel
    held = frames[seen_frame].copy()
    dist = first_track.get("first_distance_m")
    alerts_detail = info.get("alerts_detail", []) or []
    lead_txt = ""
    if alerts_detail and alerts_detail[0].get("lead_time_s") is not None:
        lead_txt = " - %.1f s before reaching it" % float(alerts_detail[0]["lead_time_s"])
    caption = "Spotted at ~%s m%s" % (
        ("%.0f" % float(dist)) if dist is not None else "-", lead_txt)
    _text(held, caption, (60, 120), 1.0, COL_WHITE, 3)
    hold_n = int(1.5 * config.PROCESS_FPS)
    out_frames = frames[:seen_frame] + [held] * hold_n + frames[seen_frame:]
    write_h264(iter(out_frames), out_path, config.PROCESS_FPS, PANEL_W, PANEL_H)
    return out_path


def _night_panel(panel: Path, out_path: Path) -> Path:
    """Freeze on the first red frame for 2 s with the showcase caption."""
    import cv2

    caption, freeze_frame = night_caption_for(panel.parent)
    if freeze_frame is None:
        return _held_panel(panel, out_path)
    cap = cv2.VideoCapture(str(panel))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames or freeze_frame >= len(frames):
        return panel
    held = frames[freeze_frame].copy()
    _caption_text(held, caption, (60, 88), px=40)
    hold_n = int(2 * config.PROCESS_FPS)
    out_frames = frames[:freeze_frame] + [held] * hold_n + frames[freeze_frame:]
    write_h264(iter(out_frames), out_path, config.PROCESS_FPS, PANEL_W, PANEL_H)
    return out_path


def _collision_panel(panel: Path, out_path: Path) -> Path:
    """Freeze 0.5 s before impact on the red alert; never show the impact."""
    import cv2

    video_name = panel.parent.name + ".mp4"
    impact_s = load_impact_times().get(video_name)
    if impact_s is None:
        return _held_panel(panel, out_path)
    run_file = panel.parent / "run.json"
    info = {}
    if run_file.exists():
        try:
            info = json.loads(run_file.read_text(encoding="utf-8"))
        except ValueError:
            info = {}
    check = info.get("ttr_check", {}) or {}
    caption = "Flagged %s s before impact - estimate within %s%% of the real impact time" % (
        ("%.1f" % check["lead_time_s"]) if check.get("lead_time_s") is not None else "-",
        ("%.0f" % check["mae_pct"]) if check.get("mae_pct") is not None else "-")
    cap = cv2.VideoCapture(str(panel))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    cut = int((float(impact_s) - 0.5) * config.PROCESS_FPS)
    cut = max(1, min(cut, len(frames)))
    span = frames[:cut]
    last = span[-1].copy()
    _text(last, caption, (60, 120), 1.0, COL_WHITE, 3)
    span = span + [last] * int(3 * config.PROCESS_FPS)
    write_h264(iter(span), out_path, config.PROCESS_FPS, PANEL_W, PANEL_H)
    return out_path


# --------------------------------------------------------------- dashboard --

def load_sample_network() -> Dict[str, object]:
    path = BASE_DIR / "sample_network.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = default_sample_network()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def default_sample_network() -> Dict[str, object]:
    roads = ["PTH 1", "PTH 12", "PTH 44", "PTH 52", "PTH 59"]
    cameras = []
    for i in range(26):
        state = "nearby" if i in (3, 11) else "clear"
        cameras.append({
            "id": "CAM-%02d" % (i + 1), "road": roads[i % len(roads)],
            "x": round(8 + (i * 37) % 84, 1), "y": round(12 + (i * 29) % 76, 1),
            "state": state})
    return {
        "cameras": cameras,
        "hotspots": [
            {"segment": "PTH 12 near Steinbach", "count": 14},
            {"segment": "PTH 44 near Beausejour", "count": 9},
            {"segment": "PTH 59 near Birds Hill", "count": 7},
            {"segment": "PTH 1 near Oakbank", "count": 5},
            {"segment": "PTH 52 near Mitchell", "count": 3}],
        "network_today": {"cameras_online": 24, "alerts_raised": 6,
                          "false_alarms_marked": 1},
    }


def dashboard_real_data() -> Dict[str, object]:
    rows = collect_result_rows()
    stats = result_stats(rows)
    best: Optional[Dict[str, object]] = None
    tiles: List[Dict[str, object]] = []
    active_stems = {p.stem for p in (BASE_DIR / "test_videos").glob("*.mp4")}
    for row in rows:
        stem = Path(str(row["video"])).stem
        if stem not in active_stems:
            continue
        out_dir = OUTPUT_DIR / stem
        highest = str(row.get("highest_level") or "green")
        state = {"red": "alert", "amber": "nearby"}.get(highest, "clear")
        if (out_dir / "demo.mp4").exists():
            tiles.append({"stem": stem, "video": str(row["video"]), "state": state,
                          "demo": "%s/demo.mp4" % stem,
                          "panel": "%s/demo_panel.mp4" % stem})
        for alert_row in row.get("alert_rows", []) or []:
            try:
                conf = float(alert_row.get("muse_confidence") or 0)
            except (TypeError, ValueError):
                conf = 0.0
            candidate = {"stem": stem, "video": str(row["video"]),
                         "alert": alert_row, "confidence": conf, "row": row}
            if best is None or conf > float(best["confidence"]):
                best = candidate
    log_lines: List[str] = []
    if best is not None:
        stem = str(best["stem"])
        alert_row = best["alert"]  # type: ignore[assignment]
        alert_time = float(alert_row["time_s"])
        delays = alert_delays_from_files(stem)
        delay = delays[0] if delays else 0.0
        log_lines.append("%s ALERT raised, %.1f s after entry" % (
            format_time(alert_time), delay))
        if alert_row.get("muse_species"):
            try:
                conf_txt = "%.2f" % float(alert_row.get("muse_confidence") or 0)
            except (TypeError, ValueError):
                conf_txt = "-"
            log_lines.append("%s Muse: %s, %s, heading %s" % (
                format_time(alert_time), alert_row["muse_species"], conf_txt,
                {"yes": "toward road", "no": "away from road"}.get(
                    alert_row.get("heading_toward_road", ""), "unclear")))
    return {"rows": rows, "stats": stats, "active": best,
            "tiles": tiles[:4], "log": list(reversed(log_lines))[:8]}


def build_dashboard() -> Path:
    import html as html_lib

    sample = load_sample_network()
    cameras = sample.get("cameras", [])  # type: ignore[attr-defined]
    hotspots = sample.get("hotspots", [])  # type: ignore[attr-defined]
    network = sample.get("network_today", {})  # type: ignore[attr-defined]
    real = dashboard_real_data()
    stats = real["stats"]  # type: ignore[assignment]
    active = real["active"]
    counts = {"alert": 0, "nearby": 0, "clear": 0}
    for cam in cameras:
        key = str(cam.get("state", "clear"))
        counts[key] = counts.get(key, 0) + 1
    if active is not None:
        counts["alert"] = max(counts["alert"], 1)
    dots = ""
    for cam in cameras:
        color = {"alert": "#FF4D5E", "nearby": "#F5A524"}.get(str(cam.get("state")), "#2FBF71")
        pulse = ' class="pulse"' if cam.get("state") == "alert" else ""
        cx, cy = float(cam["x"]) * 6.4, float(cam["y"]) * 3.6
        if cam.get("source") == "roadside_camera":
            dots += ('<rect x="%.1f" y="%.1f" width="9" height="9" fill="%s"%s>'
                     '<title>%s %s (roadside, sample)</title></rect>'
                     % (cx - 4.5, cy - 4.5, color, pulse,
                        html_lib.escape(str(cam["id"])), html_lib.escape(str(cam["road"]))))
        else:
            dots += ('<circle cx="%.1f" cy="%.1f" r="1.6" fill="%s"%s><title>%s %s</title></circle>'
                     % (cx, cy, color, pulse,
                        html_lib.escape(str(cam["id"])), html_lib.escape(str(cam["road"]))))
    if active is not None:
        alert_row = active["alert"]  # type: ignore[assignment]
        delays = alert_delays_from_files(str(active["stem"]))
        delay_text = ("%.1f s after entering the road" % delays[0]) if delays else "just after entering the road"
        cam = camera_for_clip(str(active["stem"]), sample) or {}
        road_text = str(cam.get("road") or "PTH 12")
        cam_text = ("%s - %s km %s" % (
            str(cam.get("id", "CAM 07")).replace("-", " "), road_text,
            ("%g" % float(cam["km"])) if cam.get("km") is not None else "18"))
        try:
            conf_text = "%.2f" % float(alert_row.get("muse_confidence") or 0)
        except (TypeError, ValueError):
            conf_text = "-"
        heading_text = {"yes": "heading toward road",
                        "no": "heading away from road"}.get(
                            str(alert_row.get("heading_toward_road") or ""), "heading unclear")
        active_html = (
            '<div class="card red"><h2>Active alert</h2>'
            "<p><b>Road:</b> %s &nbsp; <b>Camera:</b> %s</p>"
            "<p>Raised %s.</p><p>Muse: %s (%s), %s.</p>"
            "<p>Group: %s | Severity: %s | Assurance: %s</p>"
            '<div class="sign">%s</div>'
            '<p><button>Open camera</button> <button>Mark false alarm</button></p></div>'
            % (html_lib.escape(road_text),
               html_lib.escape(cam_text), delay_text,
               html_lib.escape(str(alert_row.get("muse_species") or alert_row.get("detected_as", ""))),
               html_lib.escape(conf_text),
               html_lib.escape(heading_text),
               html_lib.escape(str(alert_row.get("group_count") or "1")),
               html_lib.escape(str(alert_row.get("severity") or "")),
               html_lib.escape(str(alert_row.get("assurance") or "")),
               html_lib.escape(str(alert_row.get("sign_text") or config.DEFAULT_SIGN_TEXT))))
    else:
        active_html = ('<div class="card"><h2>Active alert</h2>'
                       "<p>No alert recorded yet - run the app on the test clips first.</p></div>")
    log_html = "".join("<div>%s</div>" % html_lib.escape(line) for line in real["log"]) \
        or "<div>No events yet.</div>"
    tiles_html = ""
    for tile in real["tiles"]:  # type: ignore[attr-defined]
        badge = {"alert": "ALERT", "nearby": "NEARBY"}.get(str(tile["state"]), "CLEAR")
        tiles_html += ('<a class="tile %s" href="%s"><video src="%s" autoplay muted loop playsinline></video>'
                       "<span>%s - %s</span></a>" % (
                           tile["state"], html_lib.escape(str(tile["panel"])),
                           html_lib.escape(str(tile["demo"])), badge,
                           html_lib.escape(str(tile["video"]))))
    if not tiles_html:
        tiles_html = "<p>No camera videos yet - run the app on the test clips first.</p>"
    max_count = max([int(h["count"]) for h in hotspots] or [1])
    hotspot_html = "".join(
        '<div class="hot"><span>%s</span><i style="width:%d%%"></i><b>%d</b></div>'
        % (html_lib.escape(str(h["segment"])), int(100 * int(h["count"]) / max_count),
           int(h["count"])) for h in hotspots)
    sightings = load_sightings()
    approved_n = len(waze_incidents(sightings, now_epoch=demo_now_epoch(sightings)))
    waiting_n = sum(1 for s in sightings if not s.get("approved") and not s.get("sample"))
    phys = physics_card()
    physics_html = ("At 100 km/h: %.1f m/s | Stopping (dry): %.1f m | "
                    "Stopping (snow): %.1f m | Low beams: %.0f m"
                    % (phys["m_per_s_at_100kmh"], phys["stopping_dry_m"],
                       phys["stopping_snow_m"], phys["low_beam_m"]))
    waze_html = ("Waze partner feed (CIFS): %d approved incidents ready - "
                 "%d waiting for operator approval - not live until Waze approves the feed"
                 % (approved_n, waiting_n))
    warned_html = ""
    if active is not None:
        match = next((s for s in sightings
                      if str(s.get("clip")) == str(active["video"])), None)
        if match is not None:
            warned = warned_vehicles(match, sample.get("vehicles", []))  # type: ignore[attr-defined]
            warned_html = "<br>".join(html_lib.escape(str(w["text"])) for w in warned) \
                or "No fleet vehicles in range."
    briefing_path = OUTPUT_DIR / "briefing.md"
    briefing_text = (briefing_path.read_text(encoding="utf-8").strip()
                     if briefing_path.exists() else "No briefing yet - run: python app.py --briefing")
    import html as _h

    briefing_html = ("<b>Tonight's briefing - Muse Wonder - draft for operator review</b><br>%s"
                     % _h.escape(briefing_text).replace("\n", "<br>"))
    replay_data = {"briefing": briefing_text}
    if active is not None:
        match = next((s for s in sightings
                      if str(s.get("clip")) == str(active["video"])), None)
        if match is not None:
            warned = warned_vehicles(match, sample.get("vehicles", []))  # type: ignore[attr-defined]
            replay_data.update({
                "species": match.get("species"), "road": match.get("road"),
                "km": match.get("km"), "group": match.get("group_count"),
                "demo": "%s/demo.mp4" % Path(str(active["video"])).stem,
                "warned": [str(w["text"]) for w in warned],
                "phone": "Wildlife ahead - %s %s on %s near km %s. More may follow. Slow down." % (
                    match.get("group_count"), match.get("species"),
                    match.get("road"), match.get("km")),
                "approved": bool(match.get("approved")),
            })
    page = _dashboard_html(counts, dots, active_html, log_html, tiles_html,
                           hotspot_html, network, stats["avg_delay_s"],
                           physics_html, waze_html, warned_html,
                           briefing_html, json.dumps(replay_data))
    out = OUTPUT_DIR / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def _dashboard_html(counts, dots, active_html, log_html, tiles_html,
                    hotspot_html, network, avg_delay,
                    physics_html: str = "", waze_html: str = "",
                    warned_html: str = "", briefing_html: str = "",
                    replay_json: str = "{}") -> str:
    delay = ("%.2f s" % avg_delay) if avg_delay is not None else "n/a"
    page = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Wildlife Road Alert - Operations Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>body{background:#0B1118;color:#E8EEF4;font-family:'IBM Plex Sans',sans-serif;margin:0;padding:18px}
.mono{font-family:'IBM Plex Mono',monospace}.top{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.chip{border-radius:16px;padding:4px 12px;font-weight:700}.red-b{background:#D7263D}.amber-b{background:#F5A524;color:#111}.green-b{background:#2FBF71;color:#111}
.tag{border:1px dashed #F5A524;color:#F5A524;padding:4px 10px;border-radius:6px}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:14px;margin-top:14px}.panel{background:#111A24;border:1px solid #1F2A36;border-radius:10px;padding:14px}
.card{background:#111A24;border:1px solid #1F2A36;border-radius:10px;padding:14px;margin-bottom:14px}.card.red{border-color:#FF4D5E}
.sign{background:#000;color:#F5A524;font-weight:700;letter-spacing:1px;padding:12px;border-radius:8px;text-align:center}
button{background:#1F2A36;color:#E8EEF4;border:1px solid #2FBF71;border-radius:6px;padding:8px 12px;cursor:pointer}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}.tile{position:relative;border:3px solid #2FBF71;border-radius:8px;overflow:hidden;color:#fff;text-decoration:none}
.tile.alert{border-color:#FF4D5E}.tile.nearby{border-color:#F5A524}.tile video{width:100%;height:150px;object-fit:cover;display:block;background:#000}
.tile span{position:absolute;left:8px;bottom:8px;background:rgba(0,0,0,.7);padding:3px 8px;border-radius:5px}
.bottom{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:14px}.hot i{display:inline-block;height:12px;background:#F5A524;border-radius:3px;margin:0 8px;vertical-align:middle}
.pulse{animation:p 1.2s infinite}@keyframes p{0%{opacity:1}50%{opacity:.25}100%{opacity:1}}footer{color:#8A9BA8;margin-top:14px;font-size:13px}</style></head><body>
<div class="top"><h1 style="margin:0">Wildlife Road Alert</h1><span>Network - Southern Manitoba</span>
<span class="chip red-b">__N_ALERT__ active alerts</span><span class="chip amber-b">__N_NEARBY__ animals nearby</span><span class="chip green-b">__N_CLEAR__ roads clear</span>
<span class="mono" id="clock"></span><span class="tag">CONCEPT - SAMPLE DATA</span></div>
<div class="grid"><div class="panel"><h2 style="margin-top:0">Southern Manitoba</h2>
<svg viewBox="0 0 640 360" style="width:100%;background:#0E1620;border-radius:8px">
<path d="M330 10 L390 10 L380 130 L340 120 Z" fill="#274B63"/><text x="336" y="70" fill="#BFD9EA" font-size="13">Lake Winnipeg</text>
<rect x="330" y="120" width="300" height="230" fill="#F5A524" opacity="0.13"/><text x="470" y="150" fill="#F5A524" font-size="13">Eastman</text>
<line x1="40" y1="250" x2="620" y2="250" stroke="#5B7186" stroke-width="5"/><text x="60" y="240" fill="#E8EEF4" font-size="12">PTH 1</text>
<line x1="300" y1="330" x2="470" y2="130" stroke="#5B7186" stroke-width="4"/><text x="438" y="222" fill="#E8EEF4" font-size="12">PTH 12</text>
<line x1="300" y1="230" x2="560" y2="170" stroke="#5B7186" stroke-width="4"/><text x="500" y="185" fill="#E8EEF4" font-size="12">PTH 44</text>
<line x1="330" y1="330" x2="520" y2="320" stroke="#5B7186" stroke-width="4"/><text x="392" y="350" fill="#E8EEF4" font-size="12">PTH 52</text>
<line x1="360" y1="340" x2="380" y2="110" stroke="#5B7186" stroke-width="4"/><text x="390" y="140" fill="#E8EEF4" font-size="12">PTH 59</text>
<circle cx="300" cy="250" r="5" fill="#fff"/><text x="312" y="255" fill="#fff" font-size="13">Winnipeg</text>
<circle cx="455" cy="300" r="5" fill="#fff"/><text x="467" y="305" fill="#fff" font-size="13">Steinbach</text>
__DOTS__</svg><p>Legend: <span style="color:#FF4D5E">&#9679;</span> alert &nbsp;<span style="color:#F5A524">&#9679;</span> nearby &nbsp;<span style="color:#2FBF71">&#9679;</span> clear &nbsp;&#9632; roadside thermal camera (sample)</p></div>
<div><div>__ACTIVE__</div><div class="panel"><h2 style="margin-top:0">Event log</h2><div class="mono" style="font-size:13px;line-height:1.7">__LOG__</div></div></div></div>
<div class="tiles">__TILES__</div>
<div class="bottom"><div class="panel"><h3>Driver warning</h3><p>Wildlife ahead - Deer on PTH 12, 2 km ahead. Slow down.</p></div>
<div class="panel"><h3>Hotspots this week</h3>__HOTSPOTS__</div>
<div class="panel"><h3>Network today</h3><p>Cameras online: __ONLINE__<br>Alerts raised: __RAISED__<br>Average alert delay: __DELAY__<br>False alarms marked: __FALSE__</p></div></div>
<div class="bottom"><div class="panel"><h3>Physics</h3><p>__PHYSICS__</p></div>
<div class="panel"><h3>Warned vehicles</h3><p>__WARNED__</p></div>
<div class="panel"><h3>Waze feed</h3><p>__WAZE__</p></div></div>
<div class="panel" style="margin-top:14px"><p>__BRIEFING__</p>
<p><button id="replayBtn">Replay</button></p>
<div id="replay" style="display:none">
<video id="replayVideo" style="width:640px;max-width:100%" muted loop playsinline></video>
<div id="replayPin" style="color:#FF4D5E;font-weight:700"></div>
<div id="replayWarned" class="mono" style="font-size:13px;line-height:1.7"></div>
<div id="replayPhone" style="display:none;background:#000;border:1px solid #2FBF71;border-radius:10px;padding:12px;max-width:420px"></div>
<div id="replayOp" style="display:none;color:#2FBF71;font-weight:700"></div>
<div id="replayBrief" style="display:none"></div>
</div></div>
<footer>Concept screen with sample data - detection by SAM 3.1, checks by Muse Spark (Meta).</footer>
<script>function tick(){document.getElementById('clock').textContent=new Date().toLocaleString()}tick();setInterval(tick,1000)
var REPLAY=__REPLAY_JSON__;
document.getElementById('replayBtn').onclick=function(){
var box=document.getElementById('replay');box.style.display='block';
var v=document.getElementById('replayVideo');if(REPLAY.demo){v.src=REPLAY.demo;v.play();}
document.getElementById('replayPin').textContent='Sighting: '+(REPLAY.species||'wildlife')+' on '+(REPLAY.road||'')+' near km '+(REPLAY.km||'');
var w=document.getElementById('replayWarned');w.innerHTML='';
(REPLAY.warned||[]).forEach(function(t,i){setTimeout(function(){var d=document.createElement('div');d.textContent=t;w.appendChild(d);},4000+i*1500);});
setTimeout(function(){var p=document.getElementById('replayPhone');p.style.display='block';p.textContent=REPLAY.phone||'';},12000);
setTimeout(function(){var o=document.getElementById('replayOp');o.style.display='block';o.textContent=REPLAY.approved?'Operator approved - sent to Waze feed':'Waiting for operator approval';},15000);
setTimeout(function(){var b=document.getElementById('replayBrief');b.style.display='block';b.textContent=REPLAY.briefing||'';},17000);
};</script>
</body></html>"""
    for token, value in (
            ("__N_ALERT__", str(counts["alert"])), ("__N_NEARBY__", str(counts["nearby"])),
            ("__N_CLEAR__", str(counts["clear"])), ("__DOTS__", dots),
            ("__ACTIVE__", active_html), ("__LOG__", log_html),
            ("__TILES__", tiles_html), ("__HOTSPOTS__", hotspot_html),
            ("__ONLINE__", str(network.get("cameras_online", "-"))),
            ("__RAISED__", str(network.get("alerts_raised", "-"))),
            ("__DELAY__", delay), ("__PHYSICS__", physics_html),
            ("__WAZE__", waze_html), ("__WARNED__", warned_html),
            ("__BRIEFING__", briefing_html), ("__REPLAY_JSON__", replay_json),
            ("__FALSE__", str(network.get("false_alarms_marked", "-")))):
        page = page.replace(token, value)
    return page


# ------------------------------------------------- sightings and network --

def stopping_distance_m(speed_kmh: float, decel: float) -> float:
    speed = speed_kmh / 3.6
    return speed * config.REACTION_S + speed * speed / (2.0 * decel)


def physics_card() -> Dict[str, float]:
    speed = 100.0 / 3.6
    return {"m_per_s_at_100kmh": round(speed, 1),
            "stopping_dry_m": round(stopping_distance_m(100.0, config.DECEL_DRY), 1),
            "stopping_snow_m": round(stopping_distance_m(100.0, config.DECEL_SNOW), 1),
            "low_beam_m": float(config.LOW_BEAM_M)}


def warned_vehicles(sighting: Dict[str, object],
                    vehicles: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Vehicles on the same road heading toward the sighting, within range."""
    warned: List[Dict[str, object]] = []
    s_km = float(sighting.get("km") or 0)
    for vehicle in vehicles:
        if str(vehicle.get("road")) != str(sighting.get("road")):
            continue
        v_km = float(vehicle.get("km") or 0)
        direction = str(vehicle.get("direction") or "")
        sign = 1.0 if direction in ("east", "north", "eastbound", "northbound") else -1.0
        ahead = (s_km - v_km) * sign
        if ahead <= 0 or ahead > config.SIGHTING_RADIUS_KM:
            continue
        speed = float(vehicle.get("speed_kmh") or 90)
        warned.append({"id": vehicle.get("id"), "distance_km": round(ahead, 2),
                       "minutes": round(ahead / speed * 60.0, 2),
                       "text": "%s - %.1f km away - arrives in %.1f min - warned" % (
                           vehicle.get("id"), ahead, ahead / speed * 60.0)})
    warned.sort(key=lambda w: float(w["distance_km"]))
    return warned


def load_sightings() -> List[Dict[str, object]]:
    path = OUTPUT_DIR / "sightings.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except ValueError:
            return []
    return []


def save_sightings(sightings: List[Dict[str, object]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "sightings.json"
    path.write_text(json.dumps(sightings, indent=2), encoding="utf-8")
    return path


def camera_for_clip(stem: str, network: Dict[str, object]) -> Optional[Dict[str, object]]:
    for cam in network.get("cameras", []):  # type: ignore[attr-defined]
        if cam.get("clip") == stem:
            return cam
    return None


def rebuild_sightings() -> List[Dict[str, object]]:
    """Sightings from real clip alerts plus the roadside sample sightings."""
    network = load_sample_network()
    previous_list = load_sightings()
    previous = {str(s.get("id")): s for s in previous_list}
    sightings: List[Dict[str, object]] = [
        s for s in previous_list if s.get("source") == "voice"]
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for row in collect_result_rows():
        stem = Path(str(row["video"])).stem
        cam = camera_for_clip(stem, network) or {}
        for alert_row in row.get("alert_rows", []) or []:
            sid = "%s-%s" % (stem, alert_row.get("alert_no"))
            old = previous.get(sid, {})
            muse_species = alert_row.get("muse_species") or ""
            try:
                muse_conf = float(alert_row.get("muse_confidence") or 0)
            except (TypeError, ValueError):
                muse_conf = 0.0
            rejected = str(alert_row.get("muse_rejected")).lower() == "true"
            species = (muse_species if muse_species and muse_conf >= 0.6
                       and not rejected else alert_row.get("detected_as"))
            sightings.append({
                "id": sid, "time": float(alert_row.get("time_s") or 0),
                "created_at": old.get("created_at", now_iso),
                "source": "camera",
                "species": species,
                "group_count": int(alert_row.get("group_count") or 1),
                "severity": {"moderate": 1, "high": 2, "very high": 3}.get(
                    str(alert_row.get("severity") or "moderate"), 1),
                "assurance": alert_row.get("assurance") or "Low",
                "road": cam.get("road", ""), "km": cam.get("km"),
                "side": "", "x": cam.get("x"), "y": cam.get("y"),
                "lat": cam.get("lat"), "lon": cam.get("lon"),
                "confidence": alert_row.get("muse_confidence") or alert_row.get("confidence"),
                "approved": bool(old.get("approved", False)),
                "approved_at": old.get("approved_at"),
                "sample": False, "clip": str(row["video"]),
            })
    for cam in network.get("cameras", []):  # type: ignore[attr-defined]
        if cam.get("source") == "roadside_camera":
            for sample_sighting in cam.get("sample_sightings", []):
                sid = str(sample_sighting.get("id"))
                old = previous.get(sid, {})
                sighting = dict(sample_sighting)
                sighting.update({
                    "created_at": old.get("created_at", now_iso),
                    "source": "roadside_camera", "sample": True,
                    "road": cam.get("road", ""), "km": cam.get("km"),
                    "x": cam.get("x"), "y": cam.get("y"),
                    "lat": cam.get("lat"), "lon": cam.get("lon"),
                    "approved": bool(old.get("approved", False)),
                    "approved_at": old.get("approved_at"), "clip": ""})
                sightings.append(sighting)
    sightings.sort(key=lambda s: (-int(s.get("severity") or 1), str(s.get("id"))))
    save_sightings(sightings)
    return sightings


def approve_sighting(sighting_id: str) -> Dict[str, object]:
    sightings = load_sightings()
    for sighting in sightings:
        if str(sighting.get("id")) == sighting_id:
            sighting["approved"] = True
            sighting["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            save_sightings(sightings)
            return sighting
    raise UserError("That sighting was not found - check the sighting id, then try again.")


def demo_now_epoch(sightings: Sequence[Dict[str, object]]) -> float:
    """The 'now' for demo screens: DEMO_NOW setting, else real time."""
    import datetime

    setting = str(getattr(config, "DEMO_NOW", "") or "").strip()
    if not setting:
        return time.time()
    if setting.lower() == "latest":
        stamps = []
        for sighting in sightings:
            if sighting.get("source") != "camera":
                continue
            try:
                stamps.append(datetime.datetime.strptime(
                    str(sighting.get("created_at")), "%Y-%m-%dT%H:%M:%S%z").timestamp())
            except (ValueError, TypeError):
                continue
        return (max(stamps) + 120.0) if stamps else time.time()
    try:
        return datetime.datetime.strptime(setting, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError:
        return time.time()


def waze_incidents(sightings: Sequence[Dict[str, object]],
                   now_epoch: Optional[float] = None) -> List[Dict[str, object]]:
    """CIFS incidents: active, operator-approved, never sample sightings."""
    import datetime

    now = time.time() if now_epoch is None else now_epoch
    incidents: List[Dict[str, object]] = []
    for sighting in sightings:
        if not sighting.get("approved") or sighting.get("sample"):
            continue
        created = sighting.get("created_at") or ""
        try:
            start = datetime.datetime.strptime(
                str(created), "%Y-%m-%dT%H:%M:%S%z").timestamp()
        except ValueError:
            start = now
        if now - start > config.SIGHTING_TTL_MIN * 60:
            continue
        lat, lon = sighting.get("lat"), sighting.get("lon")
        if lat is None or lon is None:
            continue
        start_dt = datetime.datetime.fromtimestamp(start).astimezone()
        end_dt = start_dt + datetime.timedelta(minutes=config.SIGHTING_TTL_MIN)
        species = str(sighting.get("species") or "Wildlife")
        incidents.append({"incident": {
            "id": str(sighting.get("id")),
            "type": "HAZARD", "subtype": "HAZARD_ON_ROAD",
            "polyline": "%s %s" % (lat, lon),
            "street": str(sighting.get("road") or ""),
            "starttime": start_dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "endtime": end_dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "description": "%s reported on the road (Wildlife Road Alert)"
                           % species.capitalize(),
            "direction": "BOTH_DIRECTIONS",
        }})
    return incidents


def build_waze_feed() -> Path:
    sightings = load_sightings()
    incidents = waze_incidents(sightings, now_epoch=demo_now_epoch(sightings))
    out = OUTPUT_DIR / "waze_feed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"incidents": incidents}, indent=2), encoding="utf-8")
    return out


# ------------------------------------------------------- voice and briefing --

def audio_to_pcm(path: Path) -> Tuple[bytes, float]:
    """Convert any audio file to raw PCM16 mono 16 kHz with ffmpeg."""
    ffmpeg = require_ffmpeg()
    probe = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe.stderr)
    seconds = 0.0
    if match:
        seconds = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    done = subprocess.run([ffmpeg, "-y", "-i", str(path), "-ac", "1", "-ar", "16000",
                           "-f", "s16le", "-"], capture_output=True)
    if done.returncode != 0 or not done.stdout:
        raise UserError("Could not read that audio file - try an MP3, WAV or M4A recording.")
    if seconds <= 0:
        seconds = len(done.stdout) / 2.0 / 16000.0
    return done.stdout, seconds


async def _transcribe_async(pcm: bytes, seconds: float):
    import websockets

    key = require_api_key()
    partials: List[str] = []
    final_text = ""
    started = time.monotonic()
    async with websockets.connect("wss://api.meta.ai/v1/asr/realtime",
                                  max_size=None) as ws:
        await ws.send(json.dumps({
            "authorization": {"accessToken": "Bearer %s" % key},
            "audioEncoding": "PCM_16KHZ", "model": config.VOICE_MODEL,
            "mode": "PUSH_TO_TALK", "partialMode": "CUMULATIVE"}))
        chunk = 3200  # 100 ms of 16 kHz mono PCM16

        async def sender() -> None:
            for i in range(0, len(pcm), chunk):
                await ws.send(pcm[i:i + chunk])
                await asyncio.sleep(0.05)
            await ws.send(json.dumps({"type": "endStream"}))

        send_task = asyncio.create_task(sender())
        try:
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    break
                except Exception:
                    break  # server closing the socket ends the session
                if isinstance(message, bytes):
                    continue
                try:
                    event = json.loads(message)
                except ValueError:
                    continue
                etype = str(event.get("type") or "")
                if etype == "error":
                    raise UserError("Transcription failed - check MODEL_API_KEY, then try again.")
                if etype == "transcript":
                    text = str(event.get("text") or event.get("transcript") or "")
                    if text:
                        if event.get("isFinal") is True or event.get("final") is True:
                            final_text = text
                        partials.append(text)
                if etype == "speechComplete" and partials:
                    final_text = partials[-1]
        finally:
            send_task.cancel()
    if not final_text and partials:
        final_text = partials[-1]
    log_api_call(config.VOICE_MODEL, "asr/realtime",
                 "audio:%.1fs clip=voice" % seconds, "",
                 (time.monotonic() - started) * 1000, None,
                 "audio_seconds=%.2f price=unknown" % seconds)
    return final_text, partials


def transcribe_audio(path: Path):
    pcm, seconds = audio_to_pcm(Path(path))
    try:
        transcript, partials = asyncio.run(_transcribe_async(pcm, seconds))
    except UserError:
        raise
    except Exception:
        raise UserError("Transcription failed - check your connection and MODEL_API_KEY, then try again.")
    if not transcript:
        raise UserError("No speech was found in that recording - try a clearer recording.")
    return transcript, partials, seconds


VOICE_PARSE_PROMPT = (
    "Extract a wildlife road report as JSON. Reply with only a JSON object: "
    '{"species": null, "road": null, "km": null, "side": null, '
    '"behaviour": null, "confidence": null}. '
    "Use null for anything not said. Never guess a location. Report:\n"
)


def parse_voice_report(transcript: str, sync_client) -> Dict[str, object]:
    started = time.monotonic()
    try:
        reply = call_with_retry(
            sync_client.chat.completions.create,
            model=config.MUSE_MODEL,
            messages=[{"role": "user", "content": VOICE_PARSE_PROMPT + transcript}])
    except Exception as exc:
        raise friendly_api_error(exc)
    usage = getattr(reply, "usage", None)
    cost = muse_cost_usd(config.MUSE_MODEL,
                         getattr(usage, "prompt_tokens", None) if usage else None,
                         getattr(usage, "completion_tokens", None) if usage else None)
    log_api_call(config.MUSE_MODEL, "chat.completions.create",
                 "text:voice-report clip=voice", str(getattr(reply, "id", "") or ""),
                 (time.monotonic() - started) * 1000, cost)
    text = reply.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    data: Dict[str, object] = {}
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                data = parsed
        except ValueError:
            data = {}
    km = data.get("km")
    try:
        km = float(km) if km is not None else None
    except (TypeError, ValueError):
        km = None
    conf = data.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    return {"species": data.get("species"), "road": data.get("road"), "km": km,
            "side": data.get("side"), "behaviour": data.get("behaviour"),
            "confidence": conf}


def run_voice_report(path, progress=None) -> Dict[str, object]:
    def note(message: str) -> None:
        if progress:
            progress(message)

    source = Path(path)
    if not source.exists():
        raise UserError("That recording was not found - check the file name, then try again.")
    note("Transcribing...")
    transcript, _partials, seconds = transcribe_audio(source)
    note("Reading the report...")
    sync_client, _async = make_clients()
    parsed = parse_voice_report(transcript, sync_client)
    network = load_sample_network()
    needs_location = not parsed.get("road") or parsed.get("km") is None
    position: Dict[str, object] = {}
    if not needs_location:
        def road_key(value: object) -> str:
            digits = re.sub(r"\D", "", str(value or ""))
            return digits or str(value or "").strip().lower()

        same_road = [c for c in network.get("cameras", [])  # type: ignore[attr-defined]
                     if road_key(c.get("road")) == road_key(parsed.get("road"))]
        if same_road:
            nearest = min(same_road, key=lambda c: abs(float(c.get("km") or 0) - float(parsed["km"])))  # type: ignore[arg-type]
            position = {"x": nearest.get("x"), "y": nearest.get("y"),
                        "lat": nearest.get("lat"), "lon": nearest.get("lon")}
    species = str(parsed.get("species") or "unknown")
    sighting: Dict[str, object] = {
        "id": "voice-%s" % source.stem,
        "time": 0, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "voice", "species": species, "group_count": 1,
        "severity": severity_of(species.lower()), "assurance": "Low",
        "road": parsed.get("road") or "", "km": parsed.get("km"),
        "side": parsed.get("side") or "",
        "confidence": parsed.get("confidence"),
        "approved": False, "approved_at": None, "sample": False, "clip": "",
        "transcript": transcript, "audio_seconds": round(seconds, 2),
        "needs_location": bool(needs_location),
    }
    sighting.update(position)
    sightings = [s for s in load_sightings() if str(s.get("id")) != sighting["id"]]
    sightings.append(sighting)
    sightings.sort(key=lambda s: (-int(s.get("severity") or 1), str(s.get("id"))))
    save_sightings(sightings)
    return {"transcript": transcript, "parsed": parsed, "sighting": sighting}


def run_briefing() -> Path:
    sync_client, _async = make_clients()
    sightings = load_sightings()
    network = load_sample_network()
    facts = {
        "time_of_day": time.strftime("%H:%M"),
        "sightings": [{k: s.get(k) for k in
                       ("id", "species", "road", "km", "severity", "assurance",
                        "source", "group_count", "approved", "sample")}
                      for s in sightings],
        "hotspots": network.get("hotspots", []),
        "network_today": network.get("network_today", {}),
    }
    prompt = (
        "Write a briefing of at most 120 words for a road authority or fleet "
        "manager: the three highest-risk stretches tonight, why, and one "
        "recommended action for each. Use only the facts given below and "
        "invent no numbers, roads or places. If there is little data, say so.\n"
        "Facts (JSON):\n" + json.dumps(facts))
    started = time.monotonic()
    try:
        reply = call_with_retry(sync_client.chat.completions.create,
                                model=config.BRIEFING_MODEL,
                                messages=[{"role": "user", "content": prompt}])
    except Exception as exc:
        raise friendly_api_error(exc)
    usage = getattr(reply, "usage", None)
    cost = muse_cost_usd(config.BRIEFING_MODEL,
                         getattr(usage, "prompt_tokens", None) if usage else None,
                         getattr(usage, "completion_tokens", None) if usage else None)
    log_api_call(config.BRIEFING_MODEL, "chat.completions.create",
                 "text:briefing clip=briefing", str(getattr(reply, "id", "") or ""),
                 (time.monotonic() - started) * 1000, cost)
    text = (reply.choices[0].message.content or "").strip()
    out = OUTPUT_DIR / "briefing.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    return out



# ------------------------------------------------------ collision checks --

def load_impact_times() -> Dict[str, float]:
    path = BASE_DIR / "test_videos" / "impact_times.json"
    if path.exists():
        try:
            return {str(k): float(v) for k, v in
                    json.loads(path.read_text(encoding="utf-8")).items()}
        except (ValueError, TypeError):
            return {}
    return {}


def ttr_check(track_frames: Dict[str, List[TrackFrame]], alerts: Sequence[Alert],
              impact_s: float) -> Dict[str, object]:
    """Estimated vs true time-to-reach for a collision clip."""
    points: List[Tuple[float, float, float]] = []
    for tfs in track_frames.values():
        for tf in tfs:
            if tf.ttc_s is not None and tf.det.time_s < impact_s:
                points.append((tf.det.time_s, tf.ttc_s, impact_s - tf.det.time_s))
    points.sort()
    errors = [abs(est - true) for _t, est, true in points]
    mae = sum(errors) / len(errors) if errors else None
    mae_pct = None
    if points and mae is not None:
        mean_true = sum(p[2] for p in points) / len(points)
        mae_pct = 100.0 * mae / mean_true if mean_true > 0 else None
    first_warn = None
    for alert in alerts:
        if alert.first_warn_s is not None:
            first_warn = alert.first_warn_s if first_warn is None else min(first_warn, alert.first_warn_s)
    return {"impact_s": impact_s, "points": points, "mae_s": mae,
            "mae_pct": mae_pct,
            "lead_time_s": (impact_s - first_warn) if first_warn is not None else None}


def write_ttr_check_image(check: Dict[str, object], out_path: Path) -> None:
    import cv2
    import numpy as np

    img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    img[:] = COL_BG
    _text(img, "Time to reach: estimated vs true (collision clip)", (60, 80), 1.1, COL_WHITE, 3)
    points = check["points"]  # type: ignore[assignment]
    mae, pct, lead = check["mae_s"], check["mae_pct"], check["lead_time_s"]
    _text(img, "Mean absolute error: %s s (%s%%) | Warning lead time: %s s" % (
        ("%.2f" % mae) if mae is not None else "n/a",
        ("%.1f" % pct) if pct is not None else "n/a",
        ("%.1f" % lead) if lead is not None else "n/a"), (60, 125), 0.7, COL_GREY, 2)
    if points:
        x0, y0, x1, y1 = 100, 900, 1820, 220
        t_max = max(p[0] for p in points) or 1.0
        v_max = max(max(p[1], p[2]) for p in points) or 1.0
        cv2.line(img, (x0, y0), (x1, y0), COL_GREY, 2)
        cv2.line(img, (x0, y0), (x0, y1), COL_GREY, 2)
        for series, color, name in ((1, COL_AMBER, "estimated"), (2, COL_GREEN, "true")):
            prev = None
            for t, est, true in points:
                value = est if series == 1 else true
                px = x0 + int((x1 - x0) * t / t_max)
                py = y0 - int((y0 - y1) * value / v_max)
                if prev is not None:
                    cv2.line(img, prev, (px, py), color, 2)
                prev = (px, py)
            _text(img, name, (x0 + (0 if series == 1 else 200), y1 - 24), 0.65, color, 2)
        if lead is not None:
            warn_t = float(check["impact_s"]) - float(lead)  # type: ignore[arg-type]
            px = x0 + int((x1 - x0) * warn_t / t_max)
            cv2.line(img, (px, y0), (px, y1), COL_RED, 2)
            _text(img, "first warning", (px + 8, y1 + 20), 0.6, COL_RED, 2)
    cv2.imwrite(str(out_path), img)


def has_audio_stream(source: Path) -> bool:
    ffmpeg = require_ffmpeg()
    probe = subprocess.run([ffmpeg, "-i", str(source)], capture_output=True, text=True)
    return "Audio:" in probe.stderr


def write_h264_audio(frames_bgr, out_path: Path, fps: float, width: int,
                     height: int, audio_source: Optional[Path] = None,
                     audio_start: float = 0.0) -> None:
    """H.264 video; with the source clip's sound when it has any."""
    ffmpeg = require_ffmpeg()
    frames_list = list(frames_bgr)
    duration = len(frames_list) / float(fps)
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (width, height), "-r", str(fps), "-i", "-"]
    if audio_source is not None and has_audio_stream(audio_source):
        cmd += ["-ss", "%.3f" % audio_start, "-i", str(audio_source),
                "-map", "0:v", "-map", "1:a", "-c:a", "aac",
                "-af", "apad", "-t", "%.3f" % duration]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for frame in frames_list:
            proc.stdin.write(frame.tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    if proc.returncode != 0 or not out_path.exists():
        raise UserError("Could not write the output video - check ffmpeg is installed, then try again.")


def showcase_caption(first_distance, delay_s, seconds_left, stopping_m) -> str:
    return ("Spotted at \u2248 %s m - red %s s later - "
            "at 100 km/h that leaves %s s; stopping needs \u2248 %s m" % (
                ("%.0f" % first_distance) if first_distance is not None else "-",
                ("%.1f" % delay_s) if delay_s is not None else "-",
                ("%.1f" % seconds_left) if seconds_left is not None else "-",
                ("%.0f" % stopping_m) if stopping_m is not None else "-"))


def night_caption_for(out_dir: Path) -> Tuple[str, Optional[int]]:
    """Showcase caption and first-red frame for a crossing clip, from run.json."""
    info: Dict[str, object] = {}
    run_file = out_dir / "run.json"
    if run_file.exists():
        try:
            info = json.loads(run_file.read_text(encoding="utf-8"))
        except ValueError:
            info = {}
    alerts_detail = info.get("alerts_detail", []) or []
    details = info.get("tracks_detail", {}) or {}
    if not alerts_detail:
        return showcase_caption(None, None, None, physics_card()["stopping_dry_m"]), None
    first = alerts_detail[0]  # type: ignore[index]
    track = details.get(first.get("track_id"), {}) if isinstance(first, dict) else {}
    first_seen = track.get("first_seen_s")
    first_dist = track.get("first_distance_m")
    alert_t = float(first.get("time_s") or 0)  # type: ignore[attr-defined]
    delay = alert_t - float(first_seen) if first_seen is not None else None
    dist_at = first.get("distance_at_alert")  # type: ignore[attr-defined]
    seconds_left = (float(dist_at) / (100.0 / 3.6)) if dist_at is not None else None
    caption = showcase_caption(
        float(first_dist) if first_dist is not None else None, delay,
        seconds_left, physics_card()["stopping_dry_m"])
    return caption, int(alert_t * config.PROCESS_FPS)


def draw_caption_band(frame, caption: str) -> None:
    """Caption in its own dark band directly below the 64 px banner."""
    import cv2

    width = frame.shape[1]
    cv2.rectangle(frame, (0, 64), (width, 128), (10, 10, 10), -1)
    _caption_text(frame, caption, (24, 80), px=40)


def video_fps(video: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 1.0 else float(config.PROCESS_FPS)


def write_clean_copies(prepared: Path, demo: Path, source: Path,
                       freeze_frame: int, caption: str, out_dir: Path) -> None:
    """opening/ending plain+alert files around the freeze frame.

    All four run at the original clip's own frame rate at 1920x1080: the
    plain files straight from the original video, the alert files with
    each original frame carrying the overlay of its nearest analysed
    (demo) frame.
    """
    import cv2

    plain_frames, width, height = load_frames(source)
    if (width, height) != (1920, 1080):
        plain_frames = [cv2.resize(f, (1920, 1080)) for f in plain_frames]
        width, height = 1920, 1080
    demo_frames, _w, _h = load_frames(demo)
    fps = video_fps(source)
    freeze_s = freeze_frame / float(config.PROCESS_FPS)
    freeze = max(1, min(int(round(freeze_s * fps)), len(plain_frames) - 1))

    def alert_frame(index: int):
        demo_idx = min(int(round(index / fps * float(config.PROCESS_FPS))),
                       len(demo_frames) - 1)
        return demo_frames[demo_idx]

    alert_frames = [alert_frame(i) for i in range(len(plain_frames))]
    opening_plain = plain_frames[:freeze + 1] + [plain_frames[freeze]] * int(2 * fps)
    write_h264_audio(iter(opening_plain), out_dir / "opening_plain.mp4", fps,
                     width, height, audio_source=source, audio_start=0.0)
    held = alert_frames[freeze].copy()
    draw_caption_band(held, caption)
    opening_alert = alert_frames[:freeze + 1] + [held] * int(3 * fps)
    write_h264(iter(opening_alert), out_dir / "opening_alert.mp4", fps, width, height)
    write_h264_audio(iter(plain_frames[freeze:]), out_dir / "ending_plain.mp4",
                     fps, width, height, audio_source=source,
                     audio_start=freeze_s)
    write_h264(iter(alert_frames[freeze:]), out_dir / "ending_alert.mp4",
               fps, width, height)


def write_openings(prepared: Path, demo: Path, impact_s: float,
                   out_dir: Path, caption: str) -> None:
    plain_frames, width, height = load_frames(prepared)
    demo_frames, _w, _h = load_frames(demo)
    cut = int((impact_s - 0.5) * config.PROCESS_FPS)
    cut = max(1, min(cut, len(plain_frames), len(demo_frames)))
    plain_span = plain_frames[:cut]
    plain_span = plain_span + [plain_span[-1]] * int(2 * config.PROCESS_FPS)
    write_h264(iter(plain_span), out_dir / "opening_plain.mp4",
               config.PROCESS_FPS, width, height)
    demo_span = demo_frames[:cut]
    last = demo_span[-1].copy()
    draw_caption_band(last, caption)
    demo_span = demo_span + [last] * int(3 * config.PROCESS_FPS)
    write_h264(iter(demo_span), out_dir / "opening_alert.mp4",
               config.PROCESS_FPS, width, height)


# ------------------------------------------------------------ thermal try --

def draw_try_overlay(frame_bgr, dets: Sequence[Detection]):
    import cv2
    import numpy as np

    out = frame_bgr.copy()
    for det in dets:
        if det.mask is not None and det.mask_w > 0 and det.mask_h > 0:
            raster = np.frombuffer(det.mask, dtype=np.uint8).reshape(det.mask_h, det.mask_w)
            box_w, box_h = det.x2 - det.x1 + 1, det.y2 - det.y1 + 1
            if (det.mask_w, det.mask_h) != (box_w, box_h):
                raster = cv2.resize(raster, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
            region = out[det.y1:det.y2 + 1, det.x1:det.x2 + 1]
            tint = np.zeros_like(region)
            tint[:] = (0, 200, 255)
            sel = raster[:region.shape[0], :region.shape[1]] > 0
            region[sel] = (region[sel] * 0.5 + tint[sel] * 0.5).astype(np.uint8)
            out[det.y1:det.y2 + 1, det.x1:det.x2 + 1] = region
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), (0, 200, 255), 2)
        _text(out, "%s %.2f" % (det.label, det.confidence),
              (det.x1, max(14, det.y1 - 6)), 0.55, (0, 200, 255), 2)
    return out


async def _try_fetch(async_client, model, images, raw_dir: Path, word: str):
    semaphore = asyncio.Semaphore(4)

    async def one(path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        raw_path = raw_dir / ("%s_%s.txt" % (digest, word))
        if raw_path.exists():
            return
        async with semaphore:
            text, _rid = await _call_with_retry_async(lambda: _stream_sam(
                async_client, model,
                [{"type": "input_text", "text": word},
                 {"type": "input_image", "image_url": image_file_data_url(path)}],
                "image", "image:%s:1image clip=try" % word, 2.50 / 1000.0))
        raw_path.write_text(text, encoding="utf-8")

    await asyncio.gather(*[one(p) for p in images])


def run_try_images(folder, prompt_word: str, progress=None) -> List[Dict[str, object]]:
    def note(message: str) -> None:
        if progress:
            progress(message)

    import cv2

    root = Path(folder)
    if not root.exists():
        raise UserError("That image folder was not found - check the name, then try again.")
    images = sorted(p for p in root.iterdir()
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise UserError("No images were found in that folder - add JPG or PNG images, then try again.")
    word = (prompt_word or "deer").strip().lower()
    out_dir = OUTPUT_DIR / "try"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [p for p in images
               if not (raw_dir / ("%s_%s.txt" % (
                   hashlib.sha256(p.read_bytes()).hexdigest(), word))).exists()]
    if missing:
        note("Calling SAM 3.1 on %d images..." % len(missing))
        sync_client, async_client = make_clients()
        model = resolve_sam_model(sync_client, clip_stem="try")
        asyncio.run(_try_fetch(async_client, model, images, raw_dir, word))
    rows: List[Dict[str, object]] = []
    for path in images:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        dets = parse_sam_image_output(
            (raw_dir / ("%s_%s.txt" % (digest, word))).read_text(encoding="utf-8"), word)
        found = [d for d in dets if d.confidence >= config.MIN_SCORE]
        frame = cv2.imread(str(path))
        if frame is not None:
            cv2.imwrite(str(out_dir / ("%s.png" % path.stem)),
                        draw_try_overlay(frame, found))
        rows.append({"image": path.name, "objects_found": len(found),
                     "highest_confidence": max((d.confidence for d in found),
                                               default=0.0),
                     "overlay": "output/try/%s.png" % path.stem})
        note("%s: %d found" % (path.name, len(found)))
    write_csv(out_dir / "summary.csv",
              ["image", "objects_found", "highest_confidence"],
              [[r["image"], r["objects_found"], "%.4f" % float(r["highest_confidence"])]
               for r in rows])
    return rows


# ---------------------------------------------------------- dataset eval --

def load_dataset(root: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for split in ("train", "valid", "test"):
        ann_file = root / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        data = json.loads(ann_file.read_text(encoding="utf-8"))
        by_image: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for ann in data.get("annotations", []):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            by_image.setdefault(int(ann["image_id"]), []).append(
                (int(x), int(y), int(x + w - 1), int(y + h - 1)))
        for image in data.get("images", []):
            items.append({
                "split": split, "path": root / split / image["file_name"],
                "name": "%s/%s" % (split, image["file_name"]),
                "w": int(image["width"]), "h": int(image["height"]),
                "humans": by_image.get(int(image["id"]), []),
            })
    return items


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


async def _eval_fetch(async_client, model, items, raw_dir: Path, note):
    semaphore = asyncio.Semaphore(4)
    done = [0]

    async def one(item) -> None:
        digest = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        raw_path = raw_dir / ("%s.txt" % digest)
        item["digest"] = digest
        if raw_path.exists():
            return
        async with semaphore:
            text = await call_sam_image_async(
                async_client, model, image_file_data_url(Path(item["path"])), "deer")
        raw_path.write_text(text, encoding="utf-8")
        done[0] += 1
        note("SAM 3.1 image %d done..." % done[0])

    await asyncio.gather(*[one(item) for item in items])


def run_eval(dataset_dir, progress=None) -> Dict[str, object]:
    def note(message: str) -> None:
        if progress:
            progress(message)

    import cv2

    root = Path(dataset_dir)
    if not root.exists():
        raise UserError("That dataset folder was not found - check the name, then try again.")
    items = load_dataset(root)
    if not items:
        raise UserError("No dataset images were found in that folder - check the name, then try again.")
    eval_dir = OUTPUT_DIR / "eval"
    raw_dir = eval_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [it for it in items
               if not (raw_dir / ("%s.txt" % hashlib.sha256(
                   Path(it["path"]).read_bytes()).hexdigest())).exists()]
    sam_model = "cached"
    sam_calls = 0
    if missing:
        note("Calling SAM 3.1 on %d new images..." % len(missing))
        sync_client, async_client = make_clients()
        sam_model = resolve_sam_model(sync_client)
        asyncio.run(_eval_fetch(async_client, sam_model, items, raw_dir, note))
        sam_calls = len(missing)

    note("Matching boxes...")
    rows: List[Dict[str, object]] = []
    total_sam = 0
    matched_sam = 0
    false_dets = 0
    for item in items:
        digest = item.get("digest") or hashlib.sha256(
            Path(item["path"]).read_bytes()).hexdigest()
        raw_path = raw_dir / ("%s.txt" % digest)
        dets = parse_sam_image_output(raw_path.read_text(encoding="utf-8"), "deer")
        img = cv2.imread(str(item["path"]))
        brightness = 0.0
        if img is not None:
            brightness = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
        light = "night" if brightness < 70 else "day"
        humans = item["humans"]  # type: ignore[assignment]
        group = "group" if len(humans) >= 2 else "single"
        sam_boxes = [(d.x1, d.y1, d.x2, d.y2, d.confidence) for d in dets
                     if d.confidence >= config.MIN_SCORE]
        total_sam += len(sam_boxes)
        used = [False] * len(sam_boxes)
        order = sorted(range(len(sam_boxes)), key=lambda i: -sam_boxes[i][4])
        found_flags = [False] * len(humans)
        for si in order:
            best_hi, best_iou = -1, 0.0
            for hi, human in enumerate(humans):
                if found_flags[hi]:
                    continue
                value = box_iou(sam_boxes[si][:4], human)
                if value > best_iou:
                    best_hi, best_iou = hi, value
            if best_hi >= 0 and best_iou >= 0.5:
                found_flags[best_hi] = True
                used[si] = True
                matched_sam += 1
        false_dets += used.count(False)
        item["sam_boxes"] = sam_boxes  # type: ignore[assignment]
        item["found_flags"] = found_flags  # type: ignore[assignment]
        for hi, human in enumerate(humans):
            best_iou, best_conf = 0.0, 0.0
            for box in sam_boxes:
                value = box_iou(box[:4], human)
                if value > best_iou:
                    best_iou, best_conf = value, box[4]
            frac = (human[3] - human[1] + 1) / float(item["h"])  # type: ignore[arg-type]
            size = "small" if frac < 0.10 else ("medium" if frac <= 0.30 else "large")
            rows.append({"image": item["name"], "split": item["split"],
                         "brightness": light, "size": size, "group": group,
                         "tags": "%s;%s;%s" % (light, size, group),
                         "found": found_flags[hi],
                         "best_iou": round(best_iou, 4), "sam_confidence": best_conf})

    def rate(found: int, total: int) -> Optional[float]:
        return round(found / total, 4) if total else None

    def group_stats(key: str, value: str) -> Dict[str, object]:
        sel = [r for r in rows if r[key] == value]
        found = sum(1 for r in sel if r["found"])
        return {"found": found, "total": len(sel), "rate": rate(found, len(sel))}

    summary = {
        "images": len(items), "human_boxes": len(rows),
        "found": sum(1 for r in rows if r["found"]),
        "sam_model": sam_model, "min_score": config.MIN_SCORE,
        "sam_calls": sam_calls,
        "estimated_cost_usd": round(len(items) * 2.50 / 1000.0, 4),
        "precision": {"matched_sam_boxes": matched_sam, "sam_boxes": total_sam,
                      "precision": rate(matched_sam, total_sam)},
        "false_detections": false_dets,
        "false_per_image": round(false_dets / len(items), 4) if items else 0,
        "by_tag": {
            "day": group_stats("brightness", "day"),
            "night": group_stats("brightness", "night"),
            "small": group_stats("size", "small"),
            "medium": group_stats("size", "medium"),
            "large": group_stats("size", "large"),
            "single": group_stats("group", "single"),
            "group": group_stats("group", "group"),
        },
    }
    summary["detection_rate"] = rate(summary["found"], len(rows))  # type: ignore[arg-type]
    write_csv(eval_dir / "eval_results.csv",
              ["image", "tags", "found", "best_iou", "sam_confidence"],
              [[r["image"], r["tags"], str(bool(r["found"])).lower(),
                "%.4f" % float(r["best_iou"]), "%.4f" % float(r["sam_confidence"])]
               for r in rows])
    (eval_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (eval_dir / "run.json").write_text(json.dumps({
        "dataset": str(root), "sam_model": sam_model,
        "muse_model": config.MUSE_MODEL, "images": len(items),
        "sam_calls": sam_calls, "frames_billed": 0,
        "estimated_cost_usd": summary["estimated_cost_usd"],
    }, indent=2), encoding="utf-8")
    write_eval_chart(summary, eval_dir / "eval_chart.png")
    write_eval_examples(items, eval_dir / "eval_examples.png")
    note("Eval done: %s of %s deer found." % (summary["found"], len(rows)))
    return summary


def write_eval_chart(summary: Dict[str, object], out_path: Path) -> None:
    import cv2
    import numpy as np

    img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    img[:] = COL_BG
    _text(img, "SAM 3.1 deer detection - Deers on roads dataset", (60, 80), 1.1, COL_WHITE, 3)
    overall = summary.get("detection_rate")
    _text(img, "Overall: %s (%s/%s deer found)" % (
        ("%.0f%%" % (100 * overall)) if overall is not None else "n/a",
        summary["found"], summary["human_boxes"]), (60, 125), 0.7, COL_GREY, 2)
    by_tag = summary["by_tag"]  # type: ignore[assignment]
    groups = [("Day vs night", ["day", "night"]),
              ("Size (far = small)", ["small", "medium", "large"]),
              ("Single vs group", ["single", "group"])]
    labels = {"day": "Day", "night": "Night", "small": "Small",
              "medium": "Medium", "large": "Large", "single": "Single",
              "group": "Group"}
    x = 80
    base_y, max_h = 880, 560
    for title, keys in groups:
        _text(img, title, (x, 200), 0.75, COL_WHITE, 2)
        for key in keys:
            stat = by_tag[key]  # type: ignore[index]
            rate_v = stat["rate"] if stat["rate"] is not None else 0
            bar_h = int(max_h * float(rate_v))
            color = COL_GREEN if float(rate_v) >= 0.5 else COL_AMBER
            cv2.rectangle(img, (x, base_y - bar_h), (x + 130, base_y), color, -1)
            _text(img, "%.0f%% (n=%d)" % (100 * float(rate_v), stat["total"]),
                  (x, base_y - bar_h - 16), 0.6, COL_WHITE, 2)
            _text(img, labels[key], (x, base_y + 36), 0.65, COL_WHITE, 2)
            x += 170
        x += 90
    precision = summary["precision"]  # type: ignore[assignment]
    _text(img, "Precision: %s | False detections per image: %s" % (
        ("%.0f%%" % (100 * precision["precision"]))
        if precision["precision"] is not None else "n/a",
        summary["false_per_image"]), (60, 960), 0.7, COL_WHITE, 2)
    _text(img, config.DATASET_CREDIT, (60, 1000), 0.55, COL_GREY, 1)
    _text(img, config.DATASET_NOTE, (60, 1032), 0.55, COL_GREY, 1)
    cv2.imwrite(str(out_path), img)


def write_eval_examples(items, out_path: Path) -> None:
    import cv2
    import numpy as np

    def has_found(item, want_small_night: bool) -> bool:
        return any(item.get("found_flags") or [])

    found_items = [it for it in items if any(it.get("found_flags") or [])]
    missed_items = [it for it in items
                    if it["humans"] and not any(it.get("found_flags") or [])]
    # Prefer small/night among the found: re-check tags via brightness/size.
    def is_small_night(item) -> bool:
        img = cv2.imread(str(item["path"]))
        night = img is not None and float(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()) < 70
        small = any((h[3] - h[1] + 1) / float(item["h"]) < 0.10
                    for h in item["humans"])
        return bool(night or small)

    found_items.sort(key=lambda it: (0 if is_small_night(it) else 1, str(it["name"])))
    chosen = found_items[:6] + missed_items[:6]
    cols, rows_n = 4, 3
    cell_w, cell_h = PANEL_W // cols, (PANEL_H - 80) // rows_n
    canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    canvas[:] = COL_BG
    for idx, item in enumerate(chosen):
        img = cv2.imread(str(item["path"]))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        scale = min((cell_w - 16) / iw, (cell_h - 46) / ih)
        resized = cv2.resize(img, (int(iw * scale), int(ih * scale)))
        ox = (idx % cols) * cell_w + 8
        oy = (idx // cols) * cell_h + 30
        canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        found_flags = item.get("found_flags") or []
        for hi, human in enumerate(item["humans"]):
            color = (255, 255, 255)
            cv2.rectangle(canvas,
                          (ox + int(human[0] * scale), oy + int(human[1] * scale)),
                          (ox + int(human[2] * scale), oy + int(human[3] * scale)),
                          color, 2)
        for box in item.get("sam_boxes") or []:
            cv2.rectangle(canvas,
                          (ox + int(box[0] * scale), oy + int(box[1] * scale)),
                          (ox + int(box[2] * scale), oy + int(box[3] * scale)),
                          (0, 200, 255), 2)
        verdict = "FOUND" if any(found_flags) else "MISSED"
        _text(canvas, "%s - %s" % (verdict, Path(str(item["path"])).name[:26]),
              (ox, oy - 8), 0.5, COL_GREEN if verdict == "FOUND" else COL_RED, 2)
    _text(canvas, "White = human box, colour = SAM 3.1 box. %s" % config.DATASET_CREDIT,
          (40, PANEL_H - 42), 0.55, COL_GREY, 1)
    _text(canvas, config.DATASET_NOTE, (40, PANEL_H - 14), 0.55, COL_GREY, 1)
    cv2.imwrite(str(out_path), canvas)


def print_report(rows: Sequence[Dict[str, object]]) -> None:
    stats = result_stats(rows)
    print("\n%-34s %-9s %-6s %-11s %-18s %s" % ("video", "expected", "alert",
                                                "first_alert", "species", "verdict"))
    for row in rows:
        fired = int(row.get("alerts") or 0) > 0
        print("%-34s %-9s %-6s %-11s %-18s %s" % (
            str(row["video"])[:34], expected_level(str(row["video"])),
            "yes" if fired else "no", str(row.get("first_alert_s") or "-"),
            str(row.get("species") or "-")[:18], verdict_for(str(row["video"]), fired)))
        print("  first seen: %s s | highest: %s | lead time: %s s | realtime_factor: %s" % (
            row.get("first_seen_s") or "-", row.get("highest_level") or "-",
            row.get("lead_time_s") or "-", row.get("realtime_factor") or "-"))
    avg = stats["avg_delay_s"]
    lead = stats["avg_lead_s"]
    print("\nCatch rate: %d/%d (%.0f%%)" % (stats["caught"], stats["should_alert"],
                                             100.0 * float(stats["catch_rate"])))
    print("False alarms: %d" % stats["false_alarms"])
    print("Average alert delay: %s" % (("%.2f s" % avg) if avg is not None else "n/a"))
    print("Average lead time: %s" % (("%.2f s" % lead) if lead is not None else "n/a"))
    print("Total frames analysed: %d" % stats["total_frames"])
    print("Total frames billed: %d" % stats["total_billed"])
    print("Total API cost (SAM + Muse): $%.4f" % float(stats["total_cost_usd"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Wildlife Road Alert")
    parser.add_argument("--video", help="run one video file")
    parser.add_argument("--batch", help="run every clip in this folder")
    parser.add_argument("--eval-images", dest="eval_images",
                        help="run the deer image dataset evaluation")
    parser.add_argument("--try-images", dest="try_images",
                        help="run SAM 3.1 on a folder of images (thermal try-out)")
    parser.add_argument("--prompt", default="deer",
                        help="animal word for --try-images (default deer)")
    parser.add_argument("--voice-report", dest="voice_report",
                        help="transcribe a voice report into a sighting")
    parser.add_argument("--briefing", action="store_true",
                        help="write output/briefing.md with Muse Wonder")
    parser.add_argument("--approve", help="approve a sighting id for the Waze feed")
    parser.add_argument("--waze-feed", action="store_true",
                        help="build output/waze_feed.json (no API calls)")
    parser.add_argument("--showcase", action="store_true",
                        help="build output/showcase.mp4 (no API calls)")
    parser.add_argument("--dashboard", action="store_true",
                        help="build output/dashboard.html (no API calls)")
    args = parser.parse_args(argv)
    try:
        if args.video:
            result = process_video(args.video)
            print("Done. Outputs in: %s" % result["out_dir"])
        elif args.batch:
            rows = run_batch(args.batch)
            print_report(rows)
            print("Summary: %s" % (OUTPUT_DIR / "summary.csv"))
            print("Results image: %s" % (OUTPUT_DIR / "results.png"))
        elif args.eval_images:
            summary = run_eval(args.eval_images)
            print(json.dumps(summary, indent=2))
            print("Chart: %s" % (OUTPUT_DIR / "eval" / "eval_chart.png"))
            print("Examples: %s" % (OUTPUT_DIR / "eval" / "eval_examples.png"))
        elif args.try_images:
            rows = run_try_images(args.try_images, args.prompt)
            for row in rows:
                print("%s: %s found (best %.2f) -> %s" % (
                    row["image"], row["objects_found"],
                    float(row["highest_confidence"]), row["overlay"]))
        elif args.voice_report:
            result = run_voice_report(args.voice_report)
            print("Transcript: %s" % result["transcript"])
            print("Sighting: %s" % json.dumps(result["sighting"], indent=2))
        elif args.briefing:
            print("Briefing: %s" % run_briefing())
        elif args.approve:
            sighting = approve_sighting(args.approve)
            print("Approved sighting %s for the Waze feed." % sighting["id"])
        elif args.waze_feed:
            path = build_waze_feed()
            count = len(json.loads(path.read_text(encoding="utf-8"))["incidents"])
            print("Waze feed: %s (%d approved incidents)" % (path, count))
        elif args.showcase:
            print("Showcase: %s" % build_showcase())
        elif args.dashboard:
            print("Dashboard: %s" % build_dashboard())
        else:
            parser.print_help()
            return 2
    except UserError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())














