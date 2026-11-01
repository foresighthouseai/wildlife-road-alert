"""SAM 3.1 output parsing with the real meta-sam-parser. No API calls.

The example line is frame 0 of the docs' video example (two objects,
320x334 source) from "Reading segmentation output".
"""
import pytest

pytest.importorskip("meta_sam_parser")

import app  # noqa: E402

EXAMPLE_LINE = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>,"
    "1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>"
)


def test_docs_example_parses_into_two_tracks_with_correct_boxes():
    detections = app.parse_sam_output(EXAMPLE_LINE, "deer")
    tracks = app.group_tracks(detections)
    assert sorted(tracks) == ["deer-0", "deer-1"]
    first = tracks["deer-0"][0]
    assert (first.x1, first.y1, first.x2, first.y2) == (211, 228, 270, 254)
    assert first.frame == 0
    second = tracks["deer-1"][0]
    assert (second.x1, second.y1, second.x2, second.y2) == (155, 228, 202, 254)


def test_docs_example_masks_are_box_sized():
    detections = app.parse_sam_output(EXAMPLE_LINE, "deer")
    by_id = {d.track_id: d for d in detections}
    assert (by_id["deer-0"].mask_w, by_id["deer-0"].mask_h) == (60, 27)
    assert (by_id["deer-1"].mask_w, by_id["deer-1"].mask_h) == (48, 27)
    assert by_id["deer-0"].mask is not None
    assert len(by_id["deer-0"].mask) == 60 * 27


def test_empty_output_gives_no_detections():
    assert app.parse_sam_output("", "deer") == []
    assert app.parse_sam_output("   \n", "deer") == []
