"""Group hold, duplicates, severity/assurance, approval, speed. No API calls."""
import json

import numpy as np

import app

FPS = 12
W, H = 640, 360


def road_mask_dict():
    mask = np.zeros((H, W), dtype=bool)
    for y in range(180, H):
        half = int(40 + (y - 180) * 1.2)
        mask[y, W // 2 - half:W // 2 + half] = True
    return {f: mask for f in range(400)}


def make_det(frame, feet_x, feet_y, track_id="deer-0", label="deer", conf=0.9):
    return app.Detection(
        frame=frame, time_s=frame / FPS, track_id=track_id, label=label,
        obj_id=track_id.split("-")[-1], confidence=conf,
        x1=feet_x - 20, y1=feet_y - 50, x2=feet_x + 20, y2=feet_y,
        frame_w=W, frame_h=H)


def test_group_hold_stays_amber_then_steps_down():
    # Near the road for 1 s (frames 0-11), then far away for good.
    near = [make_det(f, 100, 300) for f in range(12)]
    away = [make_det(f, 30, 100) for f in range(12, 200)]
    _tf, alerts, raw, _d = app.analyse_tracks(
        {"deer-0": near + away}, road_mask_dict(), 200, W, H)
    assert alerts == []
    near_flags = [False] * 200
    for f in range(12):
        near_flags[f] = True
    levels, holds = app.apply_group_hold(raw, near_flags, FPS)
    leave_s = 12 / FPS  # 1.0 s
    assert levels[int((leave_s + 4.9) * FPS)] == "amber"
    assert holds[int((leave_s + 4.9) * FPS)] is True
    assert levels[int((leave_s + 5.6) * FPS)] == "green"


def test_group_count_two_tracks_one_frame():
    masks = road_mask_dict()
    dets_a = [make_det(f, 300, 300, "deer-0", "deer") for f in range(12)]
    dets_b = [make_det(f, 340, 300, "deer-1", "deer") for f in range(12)]
    track_frames, _a, _r, _d = app.analyse_tracks(
        {"deer-0": dets_a, "deer-1": dets_b}, masks, 12, W, H)
    labels = [tf.det.label for tfs in track_frames.values() for tf in tfs
              if tf.det.frame == 0]
    assert app.group_text(labels) == "2 DEER"
    assert app.group_text(["deer", "moose"]) == "2 ANIMALS"
    assert app.group_text(["deer"]) == ""


def test_duplicates_merge_keeping_more_confident_word():
    deer = [make_det(f, 320, 300, "deer-0", "deer", conf=0.6) for f in range(12)]
    elk = [make_det(f, 320, 300, "elk-0", "elk", conf=0.9) for f in range(12)]
    kept, notes = app.merge_duplicate_tracks({"deer-0": deer, "elk-0": elk})
    assert sorted(kept) == ["elk-0"]
    assert notes["elk-0"] == "deer"


def test_merged_track_starts_at_earlier_first_frame():
    # Deer found it from frame 0; elk only from frame 6. Same boxes.
    deer = [make_det(f, 320, 300, "deer-0", "deer", conf=0.6) for f in range(12)]
    elk = [make_det(f, 320, 300, "elk-0", "elk", conf=0.9) for f in range(6, 18)]
    kept, notes = app.merge_duplicate_tracks({"deer-0": deer, "elk-0": elk})
    assert sorted(kept) == ["elk-0"]  # higher mean confidence labels it
    merged = kept["elk-0"]
    assert merged[0].frame == 0       # earliest frame of either word
    assert merged[-1].frame == 17
    assert len(merged) == 18          # every frame of both tracks
    assert all(d.label == "elk" for d in merged)
    assert notes["elk-0"] == "deer"


def test_duplicates_separate_animals_are_kept():
    deer = [make_det(f, 150, 300, "deer-0", "deer") for f in range(12)]
    elk = [make_det(f, 500, 300, "elk-0", "elk") for f in range(12)]
    kept, notes = app.merge_duplicate_tracks({"deer-0": deer, "elk-0": elk})
    assert sorted(kept) == ["deer-0", "elk-0"]
    assert notes == {}


def test_severity_and_assurance():
    assert app.severity_of("moose") == 3
    assert app.severity_text(3) == "very high"
    assert app.severity_of("horse") == 1  # unknown word
    assert app.assurance_level(0.8, 1.5, True) == "High"
    assert app.assurance_level(0.8, 1.5, None) == "Medium"  # no Muse call
    assert app.assurance_level(0.8, 1.5, False) == "Medium"
    assert app.assurance_level(0.5, 0.2, False) == "Low"


def test_muse_species_wins_for_label_and_severity():
    alert = app.Alert(alert_no=1, time_s=1.0, entry_time_s=0.5,
                      track_id="elk-0", detected_as="elk", confidence=0.71,
                      severity=app.severity_of("elk"))
    alert.muse_species = "white-tailed deer"
    alert.muse_confidence = 0.92
    alert.muse_rejected = False
    assert app.effective_species(alert) == "white-tailed deer"
    app.apply_muse_species(alert)
    assert alert.severity == 1  # deer, not elk's 2
    assert alert.detected_as == "elk"  # SAM's word is kept


def test_muse_species_needs_confidence_and_confirmation():
    low = app.Alert(alert_no=1, time_s=1.0, entry_time_s=0.5,
                    track_id="elk-0", detected_as="elk", confidence=0.71,
                    severity=app.severity_of("elk"))
    low.muse_species = "white-tailed deer"
    low.muse_confidence = 0.5  # under 0.6: SAM's word stays
    assert app.effective_species(low) == "elk"
    app.apply_muse_species(low)
    assert low.severity == 2
    rejected = app.Alert(alert_no=1, time_s=1.0, entry_time_s=0.5,
                         track_id="elk-0", detected_as="elk", confidence=0.71,
                         severity=app.severity_of("elk"))
    rejected.muse_species = "white-tailed deer"
    rejected.muse_confidence = 0.92
    rejected.muse_rejected = True  # Muse said not an animal
    assert app.effective_species(rejected) == "elk"


def test_confirmed_species_recomputes_every_frame_distance():
    # Track found as elk (2.0 m); Muse confirms white-tailed deer (1.4 m).
    dets = [make_det(f, 320, 300, "elk-0", "elk") for f in range(12)]
    tfs = [app.TrackFrame(det=d, where="on_road", toward=False, ttc_s=None,
                          distance_m=app.estimate_distance_m("elk", d.box_h, H),
                          level="red", on_road_duration=0.5) for d in dets]
    elk_dist = tfs[0].distance_m
    alert = app.Alert(alert_no=1, time_s=dets[-1].time_s, entry_time_s=0.0,
                      track_id="elk-0", detected_as="elk", confidence=0.71,
                      severity=app.severity_of("elk"), frame=dets[-1].frame,
                      distance_at_alert=tfs[-1].distance_m)
    alert.muse_species = "white-tailed deer"
    alert.muse_confidence = 0.92
    assert app.species_height_key("white-tailed deer") == "deer"
    assert app.recompute_track_distances(tfs, alert) is True
    deer_dist = app.estimate_distance_m("deer", dets[0].box_h, H)
    # Every frame, including frames before the alert, uses the deer height.
    assert all(tf.distance_m == deer_dist for tf in tfs)
    assert tfs[0].distance_m < elk_dist
    assert abs(tfs[0].distance_m / elk_dist - 1.4 / 2.0) < 1e-9
    assert alert.distance_at_alert == deer_dist
    # Under 0.6, SAM's word (and its height) stays.
    low = app.Alert(alert_no=1, time_s=1.0, entry_time_s=0.0,
                    track_id="elk-0", detected_as="elk", confidence=0.71)
    low.muse_species = "white-tailed deer"
    low.muse_confidence = 0.5
    assert app.recompute_track_distances(tfs, low) is False


def test_confirmed_species_renames_track_everywhere():
    dets = [make_det(f, 320, 300, "elk-0", "elk") for f in range(6)]
    tfs = [app.TrackFrame(det=d, where="on_road", toward=False, ttc_s=None,
                          distance_m=40.0, level="red", on_road_duration=0.5)
           for d in dets]
    track_frames = {"elk-0": tfs}
    details = {"elk-0": {"first_seen_s": 0.0}}
    alert = app.Alert(alert_no=1, time_s=0.5, entry_time_s=0.0,
                      track_id="elk-0", detected_as="elk", confidence=0.8)
    alert.muse_species = "white-tailed deer"
    alert.muse_confidence = 0.9
    new_id = app.rename_track_to_species(track_frames, details, alert)
    assert new_id == "deer-0"
    assert sorted(track_frames) == ["deer-0"]
    assert sorted(details) == ["deer-0"]
    assert all(tf.det.track_id == "deer-0" and tf.det.label == "deer"
               for tf in track_frames["deer-0"])
    assert alert.track_id == "deer-0"
    assert alert.detected_as == "elk"  # SAM's word survives only here


def test_muse_prices():
    cost = app.muse_cost_usd("muse-spark-1.2", 1_000_000, 1_000_000)
    assert abs(cost - (1.25 + 4.25)) < 1e-9
    wonder = app.muse_cost_usd("muse-wonder-1.0", 500_000, 100_000)
    assert abs(wonder - (2.0 + 2.0)) < 1e-9
    assert app.muse_cost_usd("unknown-model", 10, 10) is None


def test_waze_feed_approval_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    sightings = [
        {"id": "a-1", "approved": False, "sample": False, "lat": 49.9,
         "lon": -97.1, "road": "PTH 12", "species": "deer",
         "created_at": "2999-01-01T00:00:00+0000"},
        {"id": "rs-1", "approved": True, "sample": True, "lat": 49.8,
         "lon": -96.5, "road": "PTH 1", "species": "deer",
         "created_at": "2999-01-01T00:00:00+0000"},
    ]
    (tmp_path / "sightings.json").write_text(json.dumps(sightings))
    assert app.waze_incidents(app.load_sightings(), now_epoch=4102444800) == []
    app.approve_sighting("a-1")
    incidents = app.waze_incidents(app.load_sightings(), now_epoch=4102444800)
    assert len(incidents) == 1
    incident = incidents[0]["incident"]
    assert incident["type"] == "HAZARD"
    assert incident["subtype"] == "HAZARD_ON_ROAD"
    assert "Deer reported on the road" in incident["description"]


def test_demo_now_uses_latest_camera_sighting_plus_two_minutes():
    sightings = [
        {"source": "camera", "created_at": "2026-10-31T19:53:23-0500"},
        {"source": "voice", "created_at": "2026-10-31T23:59:59-0500"},
        {"source": "camera", "created_at": "2026-10-31T19:50:00-0500"},
    ]
    import datetime

    expect = datetime.datetime.strptime(
        "2026-10-31T19:53:23-0500", "%Y-%m-%dT%H:%M:%S%z").timestamp() + 120
    assert abs(app.demo_now_epoch(sightings) - expect) < 1
    # And with that "now", the expired-in-real-time sighting is active again.
    sighting = {"id": "x-1", "source": "camera", "approved": True,
                "sample": False, "lat": 49.8,
                "lon": -96.6, "road": "PTH 12", "species": "deer",
                "created_at": "2026-10-31T19:53:23-0500"}
    assert len(app.waze_incidents([sighting],
                                  now_epoch=app.demo_now_epoch([sighting]))) == 1


def test_realtime_factor_from_fake_log(tmp_path):
    log = tmp_path / "api_calls.log"
    log.write_text(
        "t model=sam-3.1 endpoint=responses.create input=video:deer:240frames clip=clipA duration_ms=24000 est_cost_usd=0.1\n"
        "t model=muse-spark-1.2 endpoint=chat.completions.create input=images:2 clip=clipA duration_ms=1000 est_cost_usd=0.001\n"
        "t model=sam-3.1 endpoint=responses.create input=video:deer:240frames clip=other duration_ms=99000 est_cost_usd=0.1\n")
    seconds = app.api_seconds_for("clipA", log)
    assert abs(seconds - 25.0) < 1e-9
    video_seconds = 240 / FPS
    assert abs(seconds / video_seconds - 1.25) < 1e-9


def test_physics_and_warned_vehicles():
    card = app.physics_card()
    assert abs(card["m_per_s_at_100kmh"] - 27.8) < 0.1
    # Hand check: 27.78*1.5 + 27.78^2/(2*6.9) = 41.7 + 55.9 = 97.6 m dry.
    assert abs(card["stopping_dry_m"] - 97.6) < 1.5
    sighting = {"road": "PTH 12", "km": 18.0}
    vehicles = [
        {"id": "Truck 12", "road": "PTH 12", "km": 13.9, "direction": "east",
         "speed_kmh": 95},
        {"id": "Van 3", "road": "PTH 12", "km": 40.0, "direction": "east",
         "speed_kmh": 90},
        {"id": "Truck 9", "road": "PTH 1", "km": 15.0, "direction": "east",
         "speed_kmh": 90},
    ]
    warned = app.warned_vehicles(sighting, vehicles)
    assert [w["id"] for w in warned] == ["Truck 12"]
    assert abs(warned[0]["distance_km"] - 4.1) < 0.01
    assert abs(warned[0]["minutes"] - 4.1 / 95 * 60) < 0.05
