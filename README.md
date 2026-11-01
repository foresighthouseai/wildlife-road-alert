# Wildlife Road Alert

Wildlife Road Alert watches dashcam video for deer, moose, elk, bear and
bison near the road and warns the driver before the vehicle reaches the
animal. SAM 3.1 finds and tracks each animal and segments the road surface
from a 960 px analysis copy (boxes and masks are scaled up to the
full-resolution display copy); plain geometry — on the road, in the
roadside band, moving toward the road, box growth as time-to-reach, and
distance from box height and a per-species standing height — drives three
warning levels (green, amber, red), with a group count, a "MORE MAY
FOLLOW" hold, severity by species and an assurance level. Every red alert
is checked by Muse Spark, which can confirm the species (its species
then renames the track and sets severity and distance, with SAM's word
kept in `detected_as`), write the road-sign text, or mark the alert
rejected. Confirmed alerts become sightings on a small network map;
vehicles behind the sighting are warned, an operator can approve a
sighting into a Waze-ready feed (`output/waze_feed.json`), drivers can
file voice reports that Muse Voice Transcribe turns into sightings, and
Muse Wonder writes a text briefing from the day's sightings and
hotspots. Built for the Meta Global AI Hackathon 2026.

## Setup

1. Install Python 3.10+ and ffmpeg (Mac: `brew install python ffmpeg`).
2. In this folder:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set your Meta Model API key (never commit it):

   ```bash
   export MODEL_API_KEY=your-key    # Windows: set MODEL_API_KEY=your-key
   ```

## Getting the data

The clips and datasets are not in this repository (see `.gitignore`).

- **Night dashcam clip:** Pond5 item **141272328** (commercial licence),
  saved as `test_videos/dashcam_deer_crossing_night_01.mp4`. The traffic
  clip `test_videos/traffic_no_animal_01.mp4` is a free-licence Pexels
  clip. Clip naming and expected results: `test_videos/CLIP_LIST.md`.
- **Deer photo dataset:** "Deers on roads" by Scalersai, Roboflow
  Universe, CC BY 4.0 — https://universe.roboflow.com/scalersai/deers-on-roads
  Export in COCO format into `Deers on roads.v4i.coco/`.
- **Thermal photos:** only the licence-verified images listed in
  `thermal_samples/SOURCES.md`; download them from the pages listed
  there into `thermal_samples/`.
- **Voice samples:** short driver recordings (`.mp3`) in a local
  `voice_samples/` folder; they are not shared.

## Running

| Command | What it does |
| --- | --- |
| `python app.py --video test_videos/<clip>.mp4` | Run one clip: SAM tracking, decisions, Muse check, `demo.mp4`, `demo_panel.mp4`, clean opening/ending copies, `alerts.csv`, `run.json` in `output/<clip>/` |
| `python app.py --batch test_videos/` | Run every clip in the folder and write `output/summary.csv` plus `output/results.png` |
| `python app.py --eval-images "Deers on roads.v4i.coco"` | Measure SAM 3.1 against the human deer boxes; writes `output/eval/` (chart, examples grid, `eval_summary.json`) |
| `python app.py --try-images thermal_samples/ --prompt deer` | Visual try-out of one animal word on a folder of still images; overlays in `output/try/` |
| `python app.py --voice-report "voice_samples/report.mp3"` | Transcribe a driver voice report into a sighting |
| `python app.py --briefing` | Write `output/briefing.md` from sightings and hotspots (Muse Wonder) |
| `python app.py --approve <sighting-id>` | Approve a sighting for the Waze feed |
| `python app.py --waze-feed` | Write `output/waze_feed.json` from approved sightings (no API calls) |
| `python app.py --showcase` | Build `output/showcase.mp4` from the clip panels (no API calls) |
| `python app.py --dashboard` | Build `output/dashboard.html` (no API calls) |
| `python ui.py` | Local web UI |
| `pytest` | Automated tests (no API calls) |

Re-running a video reuses the cached SAM responses in
`output/<clip>/sam_raw_*.txt`, so only changed steps cost API calls.

## Measured results

- **Photo dataset** (`output/eval/eval_summary.json`, chart and examples
  in `docs/results/`): SAM 3.1 found 487 of 508 human deer boxes
  (95.9%) at IoU ≥ 0.5, precision 89.2%, 0.21 false detections per image,
  on 282 images stretched from 416×416. Cost $0.705.
- **Dashcam clips** (`output/summary.csv`, `docs/results/results.png`):
  the night crossing clip raised a correct red alert and the traffic
  clip stayed green — catch rate 1/1, no false alarms. On the night
  clip the deer was first detected at 15.0 s at ≈ 44 m, already on the
  road; red fired 0.6 s later at ≈ 29.6 m, which a vehicle at 100 km/h
  covers in ≈ 1.1 s against a ≈ 98 m dry stopping distance. Muse
  confirmed the species and the track was renamed from SAM's word to the
  confirmed species.
- **Thermal try-out** (`thermal_samples/SOURCES.md`, overlays in
  `docs/results/`): SAM 3.1 found the animals in the full-size thermal
  scene photos; the tiny 128×128 BAMBI crops were too small to find.

## Credits

- SAM 3.1, Muse Spark, Muse Voice Transcribe and Muse Wonder: Meta
  Model API.
- Deer photos: "Deers on roads" (Scalersai, Roboflow Universe), CC BY 4.0.
- Thermal photos: Arno / Coen (CC BY-SA 3.0), Zaereth (CC BY-SA 4.0) and
  Mark Taylor (CC BY 2.0) via Wikimedia Commons; BAMBI Re-ID (Coucke,
  Schedl, Praschl, Maschek et al., Zenodo 15532500), CC BY 4.0. See
  `thermal_samples/SOURCES.md` and `docs/results/CREDITS.md`.
- Night dashcam clip: Pond5 item 141272328 (commercial licence, not
  redistributed). Traffic clip: Pexels (free licence).

## Licence

MIT — see `LICENSE`.
