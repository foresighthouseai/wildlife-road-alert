"""Time-to-reach and distance maths. No API calls."""
import app

FPS = 12


def test_ttc_constant_speed_approach():
    # Vehicle approaches at constant speed: height = k / distance.
    k, dist0, speed = 20000.0, 200.0, 10.0
    points = []
    for f in range(0, 49):
        t = f / FPS
        points.append((t, k / (dist0 - speed * t)))
    window = [p for p in points if points[-1][0] - p[0] <= 1.0]
    ttc = app.estimate_ttc(window)
    true_ttc = (dist0 - speed * points[-1][0]) / speed
    assert ttc is not None
    assert abs(ttc - true_ttc) / true_ttc < 0.10


def test_ttc_not_growing_is_none():
    flat = [(f / FPS, 100.0) for f in range(13)]
    assert app.estimate_ttc(flat) is None
    shrinking = [(f / FPS, 100.0 - f) for f in range(13)]
    assert app.estimate_ttc(shrinking) is None
    assert app.estimate_ttc([(0.0, 10.0), (0.1, 11.0)]) is None


def test_distance_deer_100px_in_1080_frame():
    dist = app.estimate_distance_m("deer", 100, 1080)
    assert dist is not None
    assert abs(dist - 14.5) / 14.5 < 0.02
