"""Muse reply parsing. No API calls."""
import app
import config

GOOD = ('{"is_animal": true, "species": "white-tailed deer", "confidence": 0.93, '
        '"heading_toward_road": "yes", "sign_text": "DEER ON ROAD - SLOW DOWN"}')


def test_plain_json():
    parsed = app.parse_muse_reply(GOOD)
    assert parsed["is_animal"] is True
    assert parsed["species"] == "white-tailed deer"
    assert parsed["confidence"] == 0.93
    assert parsed["heading_toward_road"] == "yes"
    assert parsed["sign_text"] == "DEER ON ROAD - SLOW DOWN"


def test_json_wrapped_in_prose_and_code_fences():
    reply = "Here is my check:\n```json\n" + GOOD + "\n```\nHope that helps."
    parsed = app.parse_muse_reply(reply)
    assert parsed["species"] == "white-tailed deer"
    assert parsed["sign_text"] == "DEER ON ROAD - SLOW DOWN"


def test_broken_reply_falls_back_to_default_sign():
    for bad in ("", "no json here", "{not valid json", "[]"):
        parsed = app.parse_muse_reply(bad)
        assert parsed["sign_text"] == config.DEFAULT_SIGN_TEXT
        assert parsed["is_animal"] is None


def test_rejected_animal_is_kept():
    parsed = app.parse_muse_reply(
        '{"is_animal": false, "species": "", "confidence": 0.2, '
        '"heading_toward_road": "unclear", "sign_text": "WILDLIFE ON ROAD - SLOW DOWN"}')
    assert parsed["is_animal"] is False
