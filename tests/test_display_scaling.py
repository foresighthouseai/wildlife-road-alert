"""Display scaling: analysis boxes land correctly on the display copy."""
import app


def make_det(x1, y1, x2, y2, w, h):
    return app.Detection(frame=0, time_s=0.0, track_id="deer-0", label="deer",
                         obj_id="0", confidence=0.9, x1=x1, y1=y1, x2=x2,
                         y2=y2, frame_w=w, frame_h=h)


def test_box_scales_960_to_1920():
    det = make_det(100, 50, 199, 149, 960, 540)
    app.scale_detection(det, 960, 540, 1920, 1080)
    assert (det.x1, det.y1, det.x2, det.y2) == (200, 100, 398, 298)
    assert (det.frame_w, det.frame_h) == (1920, 1080)
    assert det.feet == (299.0, 298.0)


def test_distance_survives_scaling():
    small = make_det(0, 0, 40, 99, 960, 540)    # 100 px tall at 540
    big = make_det(0, 0, 40, 99, 960, 540)      # same animal, then scaled
    app.scale_detection(big, 960, 540, 1920, 1080)
    assert big.box_h in (199, 200)  # inclusive corners lose at most 1 px
    d_small = app.estimate_distance_m("deer", small.box_h, 540)
    d_big = app.estimate_distance_m("deer", big.box_h, 1080)
    assert d_small is not None and d_big is not None
    assert abs(d_big - d_small) / d_small < 0.02


def test_same_size_is_unchanged():
    det = make_det(10, 20, 30, 40, 1920, 1080)
    app.scale_detection(det, 1920, 1080, 1920, 1080)
    assert (det.x1, det.y1, det.x2, det.y2) == (10, 20, 30, 40)
