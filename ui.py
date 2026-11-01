"""Local web page for Wildlife Road Alert. Start with: python ui.py"""
from __future__ import annotations

import csv
from pathlib import Path

import gradio as gr

import app as pipeline
import config

PRESET_NAMES = list(config.ROAD_ZONE_PRESETS)


def _zone_preview(video_path, preset_name):
    """First frame with the fallback zone drawn, shown before a run."""
    if not video_path:
        return None
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        height, width = frame.shape[:2]
        zone = config.ROAD_ZONE_PRESETS.get(preset_name, config.ROAD_ZONE)
        pts = np.array([[int(x * width), int(y * height)] for x, y in zone],
                       dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _run(video_path, preset_name, words_text, alert_seconds, min_score):
    if not video_path:
        yield "Please choose a video first.", None, None, [], []
        return
    words = [w.strip() for w in words_text.replace(";", ",").split(",") if w.strip()]
    zone = config.ROAD_ZONE_PRESETS.get(preset_name, config.ROAD_ZONE)
    messages = []

    def progress(message):
        messages.append(message)

    try:
        result = pipeline.process_video(
            video_path, animal_prompts=words, road_zone=zone,
            alert_min_seconds=float(alert_seconds), min_score=float(min_score),
            progress=progress)
    except pipeline.UserError as exc:
        yield str(exc), None, None, [], []
        return
    out_dir = Path(result["out_dir"])
    rows = []
    alerts_file = out_dir / "alerts.csv"
    if alerts_file.exists():
        with open(alerts_file, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    status = messages[-1] if messages else "Done."
    files = [str(out_dir / name) for name in
             ("demo.mp4", "demo_panel.mp4", "alerts.csv", "detections.csv", "run.json")
             if (out_dir / name).exists()]
    yield (status, str(out_dir / "demo.mp4"), str(out_dir / "demo_panel.mp4"),
           rows, files)


def _results():
    rows = pipeline.collect_result_rows()
    table = [["video", "expected", "highest", "alert", "first alert (s)",
              "species", "verdict", "first seen (s)", "lead time (s)"]]
    for row in rows:
        fired = int(row.get("alerts") or 0) > 0
        table.append([row["video"], pipeline.expected_level(str(row["video"])),
                      row.get("highest_level") or "-", "yes" if fired else "no",
                      row.get("first_alert_s") or "-", row.get("species") or "-",
                      pipeline.verdict_for(str(row["video"]), fired),
                      row.get("first_seen_s") or "-", row.get("lead_time_s") or "-"])
    stats = pipeline.result_stats(rows)
    avg = stats["avg_delay_s"]
    text = ("Catch rate: %d/%d (%.0f%%) | False alarms: %d | "
            "Average alert delay: %s | Total frames: %d | SAM cost: $%.2f"
            % (stats["caught"], stats["should_alert"], 100.0 * float(stats["catch_rate"]),
               stats["false_alarms"], ("%.2f s" % avg) if avg is not None else "n/a",
               stats["total_frames"], float(stats["total_cost_usd"])))
    image = pipeline.OUTPUT_DIR / "results.png"
    return table, text, str(image) if image.exists() else None


def _sightings():
    items = pipeline.load_sightings()
    table = [["id", "species", "road", "severity", "assurance", "approved", "sample"]]
    for s in items:
        table.append([s.get("id"), s.get("species"), s.get("road"),
                      s.get("severity"), s.get("assurance"),
                      str(bool(s.get("approved"))), str(bool(s.get("sample")))])
    ids = [str(s.get("id")) for s in items if not s.get("approved")]
    return table, ids


def _approve(sighting_id):
    if not sighting_id:
        return "Pick a sighting first."
    try:
        pipeline.approve_sighting(str(sighting_id))
        return "Approved %s for the Waze feed." % sighting_id
    except pipeline.UserError as exc:
        return str(exc)


def _voice(audio_path):
    if not audio_path:
        yield "Please record or upload a voice report first.", ""
        return
    messages = []

    def progress(message):
        messages.append(message)

    try:
        result = pipeline.run_voice_report(audio_path, progress=progress)
    except pipeline.UserError as exc:
        yield str(exc), ""
        return
    import json as json_lib

    yield result["transcript"], json_lib.dumps(result["sighting"], indent=2)


def _model_test(run_it):
    import json as json_lib

    eval_dir = pipeline.OUTPUT_DIR / "eval"
    if run_it:
        try:
            pipeline.run_eval(pipeline.BASE_DIR / "Deers on roads.v4i.coco")
        except pipeline.UserError as exc:
            return str(exc), None, None
    summary_file = eval_dir / "eval_summary.json"
    if not summary_file.exists():
        return "No model test yet - tick Run and press the button.", None, None
    data = json_lib.loads(summary_file.read_text(encoding="utf-8"))
    text = ("Overall: %s%% (%s/%s) | day %s | night %s | small %s | medium %s | "
            "large %s | single %s | group %s | precision %s | false/image %s" % (
                round(100 * (data.get("detection_rate") or 0)),
                data.get("found"), data.get("human_boxes"),
                data["by_tag"]["day"]["rate"], data["by_tag"]["night"]["rate"],
                data["by_tag"]["small"]["rate"], data["by_tag"]["medium"]["rate"],
                data["by_tag"]["large"]["rate"], data["by_tag"]["single"]["rate"],
                data["by_tag"]["group"]["rate"],
                data["precision"]["precision"], data.get("false_per_image")))
    chart = eval_dir / "eval_chart.png"
    examples = eval_dir / "eval_examples.png"
    return (text, str(chart) if chart.exists() else None,
            str(examples) if examples.exists() else None)


def _showcase():
    try:
        return "Showcase ready.", str(pipeline.build_showcase())
    except pipeline.UserError as exc:
        return str(exc), None


def _dashboard():
    try:
        path = pipeline.build_dashboard()
        return path.read_text(encoding="utf-8"), str(path)
    except pipeline.UserError as exc:
        return str(exc), None


def build_page() -> gr.Blocks:
    with gr.Blocks(title="Wildlife Road Alert") as page:
        gr.Markdown("# Wildlife Road Alert")
        with gr.Tabs():
            with gr.Tab("Run"):
                video = gr.Video(label="Road video")
                preset = gr.Dropdown(PRESET_NAMES, value="Dashcam",
                                     label="Road-zone preset")
                preview = gr.Image(label="First frame with road zone (check before running)")
                words = gr.Textbox(
                    value=", ".join(config.ANIMAL_PROMPTS),
                    label="Look for (animal words, comma-separated; new words call SAM 3.1, cached words do not)")
                alert_seconds = gr.Number(value=config.ALERT_MIN_SECONDS,
                                          label="Seconds on road before alert")
                min_score = gr.Slider(0.0, 1.0, value=config.MIN_SCORE,
                                      label="Minimum detection confidence")
                run_button = gr.Button("Run", variant="primary")
                status = gr.Textbox(label="Progress", interactive=False)
                demo = gr.Video(label="Annotated video (demo.mp4)")
                panel = gr.Video(label="Demo screen (demo_panel.mp4)")
                alerts_table = gr.Dataframe(label="Alerts", interactive=False)
                downloads = gr.File(label="Download outputs", file_count="multiple")
                video.change(_zone_preview, [video, preset], preview)
                preset.change(_zone_preview, [video, preset], preview)
                run_button.click(_run, [video, preset, words, alert_seconds, min_score],
                                 [status, demo, panel, alerts_table, downloads])
            with gr.Tab("Results"):
                refresh = gr.Button("Refresh results")
                results_table = gr.Dataframe(label="Results per clip", interactive=False)
                results_text = gr.Textbox(label="Totals", interactive=False)
                results_image = gr.Image(label="Results screen (results.png)")
                refresh.click(_results, [], [results_table, results_text, results_image])
            with gr.Tab("Demo"):
                showcase_button = gr.Button("Build showcase reel (no API calls)")
                showcase_status = gr.Textbox(label="Status", interactive=False)
                showcase_video = gr.Video(label="Showcase (output/showcase.mp4)")
                showcase_button.click(_showcase, [], [showcase_status, showcase_video])
            with gr.Tab("Dashboard"):
                dashboard_button = gr.Button("Build dashboard (no API calls)")
                dashboard_html = gr.HTML(label="Operations dashboard")
                dashboard_file = gr.File(label="Download dashboard.html")
                dashboard_button.click(_dashboard, [], [dashboard_html, dashboard_file])
            with gr.Tab("Sightings"):
                sight_refresh = gr.Button("Refresh sightings")
                sight_table = gr.Dataframe(label="Sightings (sorted by severity)",
                                           interactive=False)
                sight_pick = gr.Dropdown(label="Sighting waiting for approval")
                approve_button = gr.Button("Approve for Waze")
                approve_status = gr.Textbox(label="Status", interactive=False)
                sight_refresh.click(_sightings, [], [sight_table, sight_pick])
                approve_button.click(_approve, [sight_pick], [approve_status])
            with gr.Tab("Report by voice"):
                voice_audio = gr.Audio(label="Record or upload a wildlife report",
                                       type="filepath")
                voice_button = gr.Button("Send voice report")
                voice_transcript = gr.Textbox(label="Transcript", interactive=False)
                voice_sighting = gr.Textbox(label="Sighting created", interactive=False)
                voice_button.click(_voice, [voice_audio],
                                   [voice_transcript, voice_sighting])
            with gr.Tab("Model test"):
                run_check = gr.Checkbox(label="Run the image test first (calls SAM 3.1)",
                                        value=False)
                test_button = gr.Button("Show / run model test")
                test_text = gr.Textbox(label="Summary", interactive=False)
                test_chart = gr.Image(label="Detection chart (eval_chart.png)")
                test_examples = gr.Image(label="Examples (eval_examples.png)")
                test_button.click(_model_test, [run_check],
                                  [test_text, test_chart, test_examples])
    return page


if __name__ == "__main__":
    build_page().launch()
