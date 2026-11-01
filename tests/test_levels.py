"""Warning levels with fake tracks and a fake road mask. No API calls."""
import numpy as np

import app

FPS = 12
W, H = 640, 360


def road_mask():
    """Trapezoid road: rows 180-360, widening toward the bottom."""
    mask = np.zeros((H, W), dtype=bool)
    for y in range(180, H):
        half = int(40 + (y - 180) * 1.2)
        mask[y, W // 2 - half:W // 2 + half] = True
    return {f: mask for f in range(400)}


def make_det(frame, feet_x, feet_y, track_id="deer-0"):
    return app.Detection(
        frame=frame, time_s=frame / FPS, track_id=track_id, label="deer",
        obj_id="0", confidence=0.9, x1=feet_x - 20, y1=feet_y - 50,
        x2=feet_x + 20, y2=feet_y, frame_w=W, frame_h=H)


def analyse(dets, track_id="deer-0"):
    masks = road_mask()
    return app.analyse_tracks({track_id: dets}, masks, 400, W, H)


def test_no_animal_is_green():
    _tf, alerts, raw, _d = analyse([])
    assert alerts == []
    assert set(raw) == {"green"}


def test_roadside_band_is_amber():
    # Road edge at y=300 is x=136; the band reaches 96 px further left.
    dets = [make_det(f, 100, 300) for f in range(24)]
    _tf, alerts, raw, _d = analyse(dets)
    assert alerts == []
    assert "amber" in raw and "red" not in raw


def test_on_road_point_three_seconds_is_red_one_alert():
    dets = [make_det(f, 320, 300) for f in range(24)]
    _tf, alerts, raw, _d = analyse(dets)
    assert len(alerts) == 1
    assert abs(alerts[0].time_s - (4 / FPS)) < 1e-9  # 0.3 s after frame 0
    assert "red" in raw


def test_away_and_still_never_raises():
    dets = [make_det(f, 30, 100) for f in range(48)]
    _tf, alerts, raw, _d = analyse(dets)
    assert alerts == []
    assert set(raw) == {"green"}


def test_step_down_only_after_one_second():
    smoother = app.StateSmoother(hold_s=1.0)
    assert smoother.update("green", 0.0) == "green"
    assert smoother.update("amber", 1.0) == "amber"
    assert smoother.update("red", 2.0) == "red"
    assert smoother.update("amber", 3.0) == "red"
    assert smoother.update("amber", 3.9) == "red"
    assert smoother.update("amber", 4.0) == "amber"
    assert smoother.update("green", 4.5) == "amber"
    assert smoother.update("green", 5.5) == "green"
