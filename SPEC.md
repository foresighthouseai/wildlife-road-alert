# Build spec: Wildlife Road Alert MVP

Oct 31, 2026 · @ForesightHouseAI

## Task

Build a small Python app that takes driver's-view road video (a dashcam on a moving vehicle), spots wild animals far ahead with SAM 3.1, follows them as the vehicle approaches, and gives the driver an early **WILDLIFE ON ROAD** warning, with the seconds left to react, well before the vehicle reaches the animal. Prefer simple, working code over features. Don't train any model.

Hand back:

1. The project folder below, ready to run on my iMac.
2. A `README.md` with setup and run steps a non-programmer can follow, in 10 lines or fewer.
3. A results table from running it on every clip in `test_videos/` (see **Done when**).

```
wildlife-alert/     the folder you are started in (inside META AI on my Desktop)
  config.py         all settings
  app.py            pipeline + command line
  ui.py             local web page
  tests/            automated tests (no API calls)
  test_videos/      test clips + CLIP_LIST.md (which clips I add)
  thermal_samples/  thermal try-out images + SOURCES.md
  output/           one subfolder per video
  requirements.txt
  README.md
```

## Stack

Everything runs on a normal Mac or Windows laptop with Python 3.10 or later. There's no GPU, no Hugging Face and no local model weights: every model call goes to the Meta Model API.

- **API:** base URL `https://api.meta.ai/v1`, key from the environment variable `MODEL_API_KEY`, called with the `openai` Python package.
- **Libraries:** `openai`, `meta-sam-parser` (0.0.6 or later), `opencv-python`, `numpy`, `pandas`, `gradio`. `ffmpeg` must be installed on the system.

**SAM 3.1: detect, segment and track**

- Model ID `sam-3.1`. At startup, check `client.models.list()`. If it lists a different SAM 3.1 ID (for example `sam-3-1`), use that one.
- Upload the prepared video once with the Files API (`POST /v1/files`, `purpose` = `user_data`), then pass its `id` as `file_id`.
- Make one Responses API call per word in `ANIMAL_PROMPTS`. The input is one user message containing `input_text` (the word, such as `deer`) and `input_video` (the `file_id`). Set `metadata` to `{"include_confidence": "true"}` and stream the response.
- The output is one text line per frame: a frame marker `<Nf>`, then for each object an integer id, a box `<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=..|>` in source pixels, and a mask `<|mask;...;data=H,W,<enc>payload|>`. The mask payload is base85, not base64.
- Parse with `meta-sam-parser` and decode masks with `decode_mask_to_raster`. Don't write your own parser. In the docs example the mask raster is the size of the box, so confirm placement against the "Reading segmentation output" page before drawing.
- Object ids are stable only within one response. Name tracks `<word>-<id>` (for example `deer-0`) so ids from different words never collide.
- Limits: up to 16 tracked objects per frame. Billing is per frame, $0.20 per 1,000 frames, with a 50-frame minimum per request.

**Muse Spark: check the alert and write the sign text**

- Model `muse-spark-1.2` via Chat Completions, with images sent as base64 JPEG data URLs in `image_url` parts.
- Make the model a setting (`MUSE_MODEL`) so I can switch to `muse-wonder-1.0`.
- Call it once per alert, never per frame.

**Prices for cost totals**

- SAM 3.1: $0.20 per 1,000 video frames (50-frame minimum per request); $2.50 per 1,000 images.
- Muse Spark (`muse-spark-1.2`): $1.25 per million input tokens, $4.25 per million output tokens.
- Muse Wonder (`muse-wonder-1.0`): $4 per million input tokens, $20 per million output tokens.
- Muse Voice Transcribe: report the seconds of audio sent. If the model page lists a price, apply it; otherwise show the seconds and say the price is unknown.

## Pipeline

1. **Prepare.** With ffmpeg, trim to `MAX_SECONDS` and resample to `PROCESS_FPS`, then make two copies with the same frames, encoded H.264 MP4 with `-movflags +faststart`:
   - the **analysis copy**, at most `MAX_WIDTH` px wide, which is sent to SAM 3.1. Tested Oct 31 on the night clip: at 1920 px the `deer` prompt found nothing, while at 960 px it found the deer 0.83 s earlier and about 10 m farther away, with no change on the traffic clip and the same cost.
   - the **display copy**, at most `DISPLAY_MAX_WIDTH` px wide, which everything is drawn on. Scale SAM 3.1's boxes and masks from the analysis copy up to the display copy. Frame indices are the same in both.
2. **Track.** Call SAM 3.1 once per word in `ANIMAL_PROMPTS`, plus once with `ROAD_PROMPT` to segment the road surface. Drop animal detections below `MIN_SCORE`. Save each raw response to `output/<video>/sam_raw_<word>.txt`, keyed by a hash of the prepared video, and reuse it on re-runs. One animal can be found by two words (an elk found as both `elk` and `deer`): when two tracks from different words overlap with IoU ≥ `DUPLICATE_IOU` in at least half the frames they share, merge them into one track that keeps every frame either word found it in (in shared frames, the more confident box), so it starts at the earliest detection. Label it with the word that has the higher mean confidence, and note the other word in its log line.
3. **Decide.** Apply the rules in **Driving scenario** to every track in every frame. This is plain geometry and maths, no AI.
4. **Check.** For each alert, send Muse two images: a close-up crop (the box plus 25% padding) and the full frame with the road outline drawn, together with the prompt in **Muse check prompt**. Store the reply's fields on the alert. If Muse answers `is_animal: false`, keep the alert but mark it `muse_rejected`. If Muse confirms an animal and names a species with confidence 0.6 or more, use Muse's species for the label, the severity and the sighting, and keep SAM's word in `detected_as`. Rename the track everywhere it is shown (for example `elk-0` becomes `deer-0`): labels, the inset, the status block, the event log, the dashboard and the sighting. SAM's word may appear only in `detected_as` and in one log line such as `also found by the word elk`. Then give the alert its assurance level (see **Driving scenario**).
5. **Render.** Write an annotated H.264 MP4 at display resolution that plays in a browser:
   - the road edges as a thin yellow outline and the roadside band as a faint dashed line, never a filled band
   - each animal's mask tinted by its level (grey when away from the road, amber, or red), plus its box and a label such as `deer #0 - 180 m - 6.4 s`
   - in the banner, the group count when two or more animals are visible (for example `3 DEER`) and `MORE MAY FOLLOW` while the group hold is on
   - the ring and magnified inset for far animals described in **Driving scenario**
   - a full-width banner at the top, amber `WILDLIFE AHEAD` or red with `sign_text`, matching the level
   - the timestamp in the bottom-left corner
   - text sized for a 1080p video watched on a laptop: the banner and caption at least 36 px tall, animal labels at least 28 px on a dark semi-transparent background, placed above the box and ring so no line crosses them; the inset label at least 24 px
6. **Save** the outputs listed in **Interface and outputs**.
7. **Speed.** Add up the duration of every API call for the clip (from `output/api_calls.log`) and divide by the seconds of video analysed. If the responses came from the cache, use the durations logged when those calls were first made. Save it in `run.json` as `api_seconds` and `realtime_factor` (for example `realtime_factor: 3.2` means 3.2 s of API time per second of video). This keeps the "processed from recorded footage" claim honest and shows how far real time is.

## Driving scenario

The footage is always filmed from a vehicle driving along a road. The point is to spot an animal far ahead, often small, at the roadside or crossing, and warn the driver with enough time to slow down. A box around a close-up animal is not the goal.

**Road.** Use SAM 3.1's road mask for each frame (the largest object from the `ROAD_PROMPT` call). If a frame has no road mask, reuse the last one. If the clip has none at all, fall back to `ROAD_ZONE`. The *roadside band* is the road mask widened on both sides by `ROADSIDE_MARGIN` × frame width.

**For every animal track, in every frame, compute:**

- **Where it is:** *on road* (its feet, the bottom-centre of the box, are inside the road mask), *near road* (inside the roadside band) or *away*.
- **Moving toward the road:** the feet's sideways speed, smoothed over 0.5 s, points toward the road's centre line and is above 2% of frame width per second.
- **Time to reach (s):** how long until the vehicle reaches the animal, from how fast its box grows: box height ÷ growth in box height per second, fitted over the last 1 s. Show it only while the box is growing steadily; otherwise show `-`.
- **Distance (≈ m):** `SPECIES_HEIGHT_M[species] × focal ÷ box height in px`, where focal = (frame height ÷ 2) ÷ tan(`CAMERA_VFOV_DEG` ÷ 2). Always label it as approximate. The species is the track's final one: when Muse confirms a species (see **Pipeline**, step 4), use the `SPECIES_HEIGHT_M` key it contains ("white-tailed deer" → `deer`) and recompute the track's distances for every frame, including frames before the alert; otherwise use SAM's word.
- **First seen:** the time, distance and box height in px when the track first appeared.

**Warning levels**, per frame, taking the highest over all tracks:

| Level | When |
| --- | --- |
| Green | No animal is on or near the road |
| Amber | An animal is near the road, or moving toward it, or on the road with time to reach above `RED_TTC_SECONDS` |
| Red | An animal has been on the road for `ALERT_MIN_SECONDS`, or is near the road and moving toward it with time to reach at or below `RED_TTC_SECONDS` |

Animals away from the road and not moving toward it are tracked quietly in grey and never raise the level. Fire one alert per track, the first time it reaches red. Move up a level immediately; step down only after the lower level has held for 1 s.

**More may follow.** Deer, elk, bison and wild pigs often travel in groups, so a driver who sees one should expect another. After any track has been amber or red, hold the level at amber or above for `GROUP_HOLD_SECONDS` after the last animal leaves the road and the roadside band, with the status text `MORE MAY FOLLOW`. The 1 s step-down rule still applies on top: red steps down to this amber after 1 s, and the amber steps down to green only when the hold ends. When two or more animal tracks are visible in the same frame, show the count with the species (for example `3 DEER`, or `2 ANIMALS` for mixed species) in the banner, the status block and the alert.

**Severity.** Each species has a severity from `SPECIES_SEVERITY` (1 moderate, 2 high, 3 very high), based on how much harm a collision does: moose, bison and cattle are tall and heavy enough to come through the windshield. Show it on the alert as text (`Severity: very high`), store it on the alert and the sighting, and use it to sort alerts and sightings on the dashboard. It does not change the warning levels.

**Assurance.** Each alert gets an assurance level that says how sure the system is, separate from how dangerous the situation is:

| Assurance | When |
| --- | --- |
| High | All three: the track's mean SAM confidence is at least `ASSURANCE_HIGH_SCORE`, the animal was tracked for at least `ASSURANCE_MIN_TRACK_S`, and Muse confirmed it is an animal |
| Medium | Two of the three |
| Low | One or none |

If Muse was not called (switched off or over `MAX_MUSE_CALLS`), count that condition as not met. Show assurance next to severity on the demo panel, the dashboard alert card and in `alerts.csv`.

**Lead time.** For each alert, record the warning lead time: from the track's first amber or red frame to the moment the vehicle reaches the animal (alert time + time to reach at the alert). This is the headline number: how many seconds the driver gained.

**Far animals.** When an animal's box is shorter than 6% of the frame height, draw a thin ring around it and a 4× magnified inset in the top-right corner of the camera view, joined to it by a line and labelled with species and distance, so viewers can see what was spotted.

**Test footage.** Only clips filmed from a moving vehicle count. `test_videos/archive/` stays out of batch runs, the showcase and the dashboard. Clip names follow `dashcam_<species>_<event>_<light>_<nn>.mp4`, where event is `crossing`, `on_road`, `beside_road`, `collision` or `no_animal`, and light is `day`, `dusk` or `night`. The existing `traffic_no_animal_01.mp4` counts as a `no_animal` clip. Expected results: `crossing`, `on_road` and `collision` reach red, `beside_road` reaches amber but never red, and `no_animal` stays green. I add every clip myself (see `test_videos/CLIP_LIST.md`); don't download clips.

## Dataset evaluation

**Status: done on Oct 28.** SAM 3.1 found 487 of 508 deer (95.9%): 31 of 32 small, 21 of 22 at night, precision 89.2%, cost $0.705. Don't re-run it unless I ask, and keep `output/eval/` and its cache as they are. The steps below stay as the record of how it was measured.

The folder `Deers on roads.v4i.coco/` in the project is a public dataset of 282 deer photos with deer boxes drawn by people ("Deers on roads" by Scalersai, Roboflow Universe, CC BY 4.0). It has `train/`, `valid/` and `test/` folders, each with images and a `_annotations.coco.json` file. Treat all three as one test set. Use it to measure how well SAM 3.1 finds deer with no training.

`python app.py --eval-images "Deers on roads.v4i.coco"`:

1. **Detect.** For every image, call SAM 3.1 with the image as `input_image` (a base64 data URL) and the prompt `deer`, with confidence on. Cache each raw response in `output/eval/raw/`, keyed by a hash of the image, so re-runs cost nothing.
2. **Match.** A human box counts as *found* when a SAM 3.1 box with confidence at or above `MIN_SCORE` overlaps it with IoU ≥ 0.5. Match greedily, using each SAM box once. SAM boxes left unmatched are *false detections*.
3. **Tag.** Tag each human box *night* when its image's mean brightness is below 70 (on 0–255), otherwise *day*. Tag it *small* (box under 10% of image height, a proxy for far away), *medium* (10–30%) or *large* (over 30%). Tag it *group* when its image has two or more human boxes, otherwise *single*.
4. **Report.** Write:
   - `output/eval/eval_results.csv`: one row per human box (image, tags, found or not, best IoU, SAM confidence)
   - `output/eval/eval_summary.json`: detection rate overall and for each tag, precision, false detections per image, with the count behind every number
   - `output/eval/eval_chart.png`: 1920×1080, same dark style, bars of detection rate for day vs night, small vs medium vs large, and single vs group, each labelled with its count (for example `78% (n=41)`)
   - `output/eval/eval_examples.png`: a grid of 12 images (6 found, including small and night ones, and 6 missed), human boxes in white, SAM 3.1 boxes in colour

   On both images, add the credit `Data: Deers on roads (Scalersai, Roboflow Universe), CC BY 4.0` and the note `Images are 416×416 and stretched; scores may be lower than on full-resolution video`.
5. **Cost.** SAM 3.1 images cost $2.50 per 1,000, so 282 images cost about $0.71. Include it in `run.json` and the cost totals.

Add a **Model test** tab to the web page showing the chart, the examples grid and the summary numbers. Report the summary numbers back to me with the file paths, and check 10 random matches by eye before reporting.

**Proof of model use.** Append every API call to `output/api_calls.log`: time, model ID, endpoint, input type and size (frames or image), response id, duration in ms and estimated cost. Never log the key.

## Network, voice and briefing

At 100 km/h a vehicle covers about 28 m per second, and low beams reach about 50 m, so the camera that first sees an animal can often give its own driver only a few seconds. The vehicles behind it can get minutes. This part turns every detection into a shared sighting.

**Sightings.** Every alert, and every voice report, becomes a sighting in `output/sightings.json`: id, time, source (`camera`, `roadside_camera` or `voice`), species, group count, severity, assurance, road, km, side, map position, confidence, `approved` (false by default) and, for camera sightings, the clip. Map each test clip to a camera in `sample_network.json` (for example `CAM 07 - PTH 12 km 18`) to give it a road and position.

**Roadside cameras (sample).** Fixed roadside detection systems, like thermal cameras on poles beside a high-risk stretch, can feed the same network. Add 2 roadside cameras to `sample_network.json` with `source` = `roadside_camera` and `sample` = true (for example `RS 01 - PTH 1 km 112 - roadside thermal camera (sample)`), each with one sample sighting. Draw them on the map with a square marker instead of a dot, list them in the legend, and label their sightings `sample`.

**Warnings to other vehicles.** Add 10 sample fleet vehicles to `sample_network.json` (id, road, km, direction, speed 80 to 100 km/h). For each sighting, find the vehicles on the same road heading toward it within `SIGHTING_RADIUS_KM`, and compute distance and minutes to arrival (distance ÷ speed). Each one is warned, for example `Truck 12 - 4.1 km away - arrives in 2.6 min - warned`. A sighting stays active for `SIGHTING_TTL_MIN`.

**Physics card.** Compute from the settings below and show on the dashboard: metres per second at 100 km/h, stopping distance on a dry road and on snow (speed × `REACTION_S` + speed² ÷ (2 × deceleration)), and low-beam reach. Under it, compare the warning lead time from the vehicle's own camera (from the alert) with the lead time from the network (the warned vehicles' minutes to arrival).

**Voice reports (Muse Voice Transcribe + Muse Spark).**

- `python app.py --voice-report voice_samples/<file>`, plus a **Report by voice** panel on the web page (record with the microphone or upload a file).
- Convert the audio with ffmpeg to PCM16, mono, 16 kHz. Transcribe it with `VOICE_MODEL` through Meta's realtime transcription endpoint (`wss://api.meta.ai/v1/asr/realtime`). Meta's API has no file-upload transcription, so stream the recording in short chunks as the speech-to-text docs describe. Follow those docs exactly; don't invent fields.
- Send the transcript to Muse Spark and have it return only JSON: `{"species", "road", "km", "side", "behaviour", "confidence"}`. Missing fields stay null; never guess a location. If road or km is missing, flag the report `needs location` and leave it off the map.
- On screen, show the transcript appearing word by word, then the pin dropping on the map.
- I'll put two recordings in `voice_samples/`.

**Network briefing (Muse Wonder).** `python app.py --briefing` sends the sightings, hotspot counts and time of day to `BRIEFING_MODEL` and asks for a briefing of at most 120 words for a road authority or fleet manager: the three highest-risk stretches tonight, why, and one recommended action for each. The prompt must say to use only the facts given and to invent no numbers. Save it to `output/briefing.md` and show it on the dashboard as `Tonight's briefing - Muse Wonder - draft for operator review`.

**Operator approval.** A person approves anything that leaves the system for the public. Each sighting starts with `approved` = false. Approve it with an **Approve for Waze** button next to each sighting on a **Sightings** tab of the web page, or with `python app.py --approve <sighting_id>`; both set `approved` = true, with the time, in `output/sightings.json`. Only approved, active sightings go into the Waze feed. Warnings to the sample fleet vehicles are internal and don't need approval.

**Demo time.** With `DEMO_NOW` = `"latest"`, the dashboard, the Replay and `--waze-feed` treat the latest camera sighting's time + 2 minutes as "now", so recorded screens show the sighting as active. `""` means real time; an ISO timestamp pins a moment.

**Dashboard replay.** Add a **Replay** button to the dashboard that animates the network moment in about 20 seconds, for screen recording: the alerting camera's clip plays, the sighting pin pulses on the map, the warned-vehicle list fills one by one with distance and minutes, the phone notification slides in (built from the sighting, for example `Wildlife ahead - 2 deer on PTH 12 near km 18, 4 km ahead. More may follow. Slow down.`), an `Operator approved - sent to Waze feed` line appears, then the briefing appears.

**Showcase.** After the crossing clips, add a 20-second network scene recorded from the dashboard replay. If recording it automatically isn't practical, tell me and I'll screen-record it.

Add these settings to `config.py`:

```python
SIGHTING_RADIUS_KM = 10      # warn vehicles heading toward a sighting within this distance
SIGHTING_TTL_MIN = 20        # how long a sighting stays active
REACTION_S = 1.5             # typical driver reaction time
DECEL_DRY = 6.9              # braking, m/s², dry road
DECEL_SNOW = 2.5             # braking, m/s², snow
LOW_BEAM_M = 50              # typical low-beam reach
VOICE_MODEL = "muse-voice-transcribe-1.0"
BRIEFING_MODEL = "muse-wonder-1.0"
```

**This part is done when** both voice samples become correct sightings (or are correctly flagged `needs location`), the briefing mentions only roads and numbers that exist in the data, the warned-vehicle numbers check out by hand for one sighting, the Waze feed holds only approved sightings, and every call appears in `output/api_calls.log`.

## Validation clips

The night crossing clip is checked against what a person can see in it, and a small image test checks thermal imagery.

**Night crossing clip: first sighting checked by eye.** `dashcam_deer_crossing_night_01.mp4` is a real dashcam at night: the deer only becomes visible when it enters the headlights, already on the road, about 40 m ahead. Measured Oct 31 at 960 px: a person can faintly see it on the roadside from about 13.8 s; the system first detects it at 15.0 s, ≈ 43 m ahead and already on the road, as `deer` (0.85); red 0.33 s later; ≈ 38 m at red, which is 1.4 s at 100 km/h against ≈ 98 m dry stopping. Each run, report:

- the first frame where a person can see the deer (check the frames by eye) and the first frame the system detects it, and the number of frames between them
- the distance and box height at first detection, and the time from first detection to red
- the highway comparison: the distance at red ÷ the speed at 100 km/h (27.8 m/s) = seconds left at highway speed, next to the dry stopping distance from the physics card

In the showcase, freeze this clip on its first red frame for 2 s with the caption `Spotted at ≈ [distance] m - red [delay] s later - at 100 km/h that leaves [seconds] s; stopping needs ≈ [stopping] m`, filled from the run. Never claim the first visible frame: a person can see the deer about a second before the system does.

**Collision clips (only if one is added later): time to reach checked against the real impact.** A clip whose event is `collision` (for example `dashcam_deer_collision_dusk_01.mp4`) shows the vehicle reaching the animal, so the true time to reach is known for every frame: impact time − frame time. I'll write each clip's impact time, in seconds from the start of the original clip, in `test_videos/impact_times.json`, for example `{"dashcam_deer_collision_dusk_01.mp4": 14.2}`. For every frame where the system shows a time to reach, compare it with the truth and report:

- the mean absolute error, in seconds and as a percentage
- the warning lead time: impact time − the first amber or red time for that animal
- `output/<video>/ttr_check.png`: 1920×1080, same dark style, estimated vs true time to reach across the clip, with the lead time marked

A `collision` clip is expected to reach red before the impact. In the showcase, freeze it 0.5 s before impact on the red alert, with the caption `Flagged [lead] s before impact - estimate within [error]% of the real impact time`. Never show the impact itself.

**Clean copies for the video edit.** For `dashcam_deer_crossing_night_01.mp4`, write four files to `output/<video>/`. The *freeze frame* is the first red frame (for a collision clip it would be the frame 0.5 s before impact):

- `opening_plain.mp4`: the prepared video with no overlays at all, from the start to the freeze frame, with that frame held for 2 s
- `opening_alert.mp4`: the same span taken from `demo.mp4`, ending on the same frame held for 3 s with the showcase caption above
- `ending_plain.mp4`: the prepared video with no overlays, from the freeze frame to the end of the clip
- `ending_alert.mp4`: the same span taken from `demo.mp4`

Write all four at 1920×1080 and at the original clip's own frame rate, so they play smoothly: the plain files straight from the original video, and the alert files by drawing on each original frame the overlay of the nearest analysed frame. H.264. Keep the original clip's sound on the two plain files if it has any; no sound on the alert files.

**Old preview.** Delete `output/dashcam_deer_crossing_night_preview/`, and never use anything from `previews/`: it was a watermarked test copy.

**Thermal try-out (images, not part of the showcase).** A quick look at whether SAM 3.1 finds animals in thermal pictures, with no training.

1. Make a `thermal_samples/` folder. Download 6 to 10 thermal images of deer, moose or elk from free, openly licensed sources only: the Wikimedia Commons category *Thermal images of animals*, and frames taken from the BAMBI Re-ID dataset on Zenodo (record 15532500, `reid.zip`, 75 MB, CC BY 4.0). Use a file only if its page states a free licence (CC BY, CC BY-SA, CC0 or public domain).
2. Write `thermal_samples/SOURCES.md` listing every file: source page link, author, licence.
3. `python app.py --try-images thermal_samples/ --prompt deer` calls SAM 3.1 once per image (with confidence on, using the image price), and writes `output/try/<image>.png` with the masks and boxes drawn, plus `output/try/summary.csv` (image, objects found, highest confidence).
4. There are no human boxes, so this is a visual check, not a score. Report to me what it found, with the overlay paths, and don't put it in the showcase or the dashboard.

## Waze feed

Waze lets approved partners publish road incidents through its partner data feed, in its CIFS format, and adds them to the Waze map automatically. `python app.py --waze-feed` writes `output/waze_feed.json`, with every active, operator-approved sighting as one CIFS incident, following Waze's [incident information docs](https://developers.google.com/waze/data-feed/incident-information) exactly.

- Use only the fields and types the docs list. If there is no animal-specific type, use the general hazard type they describe and put the species in the description, for example `Deer reported on the road (Wildlife Road Alert)`.
- Start and end times come from the sighting time and `SIGHTING_TTL_MIN`.
- Check the file against the docs' required fields and report any gaps.
- Sample sightings (from roadside cameras or sample data) never go into the feed.
- On the dashboard, show a small panel: `Waze partner feed (CIFS): [n] approved incidents ready - [m] waiting for operator approval - not live until Waze approves the feed`.

## Settings

Put every setting in `config.py`, using exactly these names and defaults:

```python
ANIMAL_PROMPTS = ["deer", "moose", "elk", "bear", "bison"]   # one SAM 3.1 call each
ROAD_PROMPT = "road"                        # one extra SAM 3.1 call per clip
MAX_SECONDS = 20                            # only analyse the start of each clip
PROCESS_FPS = 12                            # frames per second sent to SAM 3.1
MAX_WIDTH = 960                             # width of the copy sent to SAM 3.1 (tested better on dark footage)
DISPLAY_MAX_WIDTH = 1920                    # width of everything drawn and shown
ROADSIDE_MARGIN = 0.15                      # roadside band, as a fraction of frame width
ALERT_MIN_SECONDS = 0.3                     # time on the road before a red alert
RED_TTC_SECONDS = 5.0                       # red when the vehicle reaches the animal sooner than this
MIN_SCORE = 0.45                            # ignore detections less confident than this
CAMERA_VFOV_DEG = 55                        # assumed dashcam vertical field of view
SPECIES_HEIGHT_M = {"deer": 1.4, "moose": 2.2, "elk": 2.0, "bear": 1.0, "bison": 1.8, "cow": 1.5}   # rough standing height, for distance
SPECIES_SEVERITY = {"deer": 1, "bear": 2, "elk": 2, "moose": 3, "bison": 3, "cow": 3}   # 1 moderate, 2 high, 3 very high; unknown words = 1
DUPLICATE_IOU = 0.5                         # same animal found by two words
GROUP_HOLD_SECONDS = 5.0                    # stay at amber or above after an animal leaves: more may follow
ASSURANCE_HIGH_SCORE = 0.70                 # mean SAM confidence for high assurance
ASSURANCE_MIN_TRACK_S = 1.0                 # seconds tracked for high assurance
DEMO_NOW = "latest"                         # "now" for the dashboard, Replay and Waze feed: latest camera sighting + 2 min; "" = real time
ROAD_ZONE = [(0.35, 0.55), (0.65, 0.55), (1.0, 1.0), (0.0, 1.0)]        # fallback only, if SAM finds no road
USE_MUSE_CHECK = True
MUSE_MODEL = "muse-spark-1.2"
MAX_MUSE_CALLS = 20                         # cap per video
```

Also offer these road-zone presets:

- **Dashcam** (the default above)
- **Roadside camera:** `[(0.0, 0.4), (1.0, 0.4), (1.0, 0.75), (0.0, 0.75)]`
- **Road narrowing ahead:** `[(0.4, 0.5), (0.6, 0.5), (1.0, 1.0), (0.0, 1.0)]`

## Muse check prompt

Send this text with the two images, unchanged:

```
You are checking an alert from a wildlife road-safety camera.
Image 1 is a close-up of the detected object. Image 2 is the full camera view; the road edges are outlined in yellow.
Reply with only a JSON object:
{"is_animal": true, "species": "white-tailed deer", "confidence": 0.9,
 "heading_toward_road": "yes" | "no" | "unclear",
 "sign_text": "DEER ON ROAD - SLOW DOWN"}
sign_text: 6 words or fewer, capital letters, suitable for a roadside warning sign.
```

Parse the reply by taking the first `{...}` object, so extra words or code fences don't break it. If parsing fails, use `WILDLIFE ON ROAD - SLOW DOWN` as the sign text and carry on.

## Interface and outputs

**Commands**

- `python app.py --video test_videos/clip.mp4` runs one video.
- `python app.py --batch test_videos/` runs every clip in the folder and writes `output/summary.csv`.
- `python ui.py` opens a local web page.
- `python app.py --approve <sighting_id>` approves a sighting for the Waze feed.
- `python app.py --try-images thermal_samples/ --prompt deer` runs the thermal try-out.

**Web page (Gradio)**

- Upload a video and pick a road-zone preset. The first frame is shown with the zone drawn *before* the run, so I can check it.
- Animal words and the main thresholds are editable fields.
- A **Run** button, progress text, then the annotated video, the alerts table and download buttons for all outputs.

**Files written to `output/<video-name>/`**

| File | Contents |
| --- | --- |
| `demo.mp4` | The annotated video |
| `alerts.csv` | alert_no, time_s, track_id, detected_as, confidence, group_count, severity, assurance, muse_species, muse_confidence, heading_toward_road, sign_text, muse_rejected |
| `detections.csv` | time_s, frame, track_id, label, confidence, on_road, x1, y1, x2, y2 |
| `run.json` | Settings used, model IDs, number of API calls, frames billed, estimated cost, `api_seconds`, `realtime_factor` |
| `opening_plain.mp4`, `opening_alert.mp4`, `ending_plain.mp4`, `ending_alert.mp4` | The night crossing clip only: the clean copies for the video edit |

`output/summary.csv` (batch mode) has one row per video: video, animals_found, alerts, first_alert_s, species, frames_billed.

## Demo screen

Also build a view made for screen recording. It shows the system as a monitoring station watching a road camera, not as an upload form.

**Three states, worked out for every frame from the tracks:**

| State | When | Status panel | Road sign |
| --- | --- | --- | --- |
| Road clear (green) | Level is green | `ROAD CLEAR` | Blank |
| Wildlife ahead (amber) | Level is amber | `WILDLIFE AHEAD`, plus species, distance and time to reach | `CAUTION - WILDLIFE AHEAD` |
| Slow down (red) | Level is red | `SLOW DOWN`, plus species, distance and time to reach, counting down | The alert's `sign_text` |

Move up a level (green to amber to red) immediately. Move down only after the lower state has held for 1 s, so the screen doesn't flicker. While the group hold is on, the amber status reads `MORE MAY FOLLOW`. With every alert, the status block also shows `Severity: <level>` and `Assurance: <level>`.

**`demo_panel.mp4`, written for every video:** 1920×1080, dark background, one sans-serif font, H.264, playing at the prepared video's frame rate.

- **Left:** the annotated camera view from `demo.mp4`, scaled to fit, with a small label `CAMERA 01 · <file name>` and the timestamp.
- **Right column, top to bottom:**
  1. The status block in the state's colour, with large text.
  2. A road-sign mockup: a black rounded rectangle with amber capital letters, showing the sign text from the table above.
  3. An event log, newest line first, at most 8 lines. For example: `00:04.2 deer-0 tracked beside road`, `00:06.1 deer-0 entered road`, `00:06.6 ALERT raised, 0.5 s after entry`, `00:07.4 Muse: white-tailed deer, 0.93, heading toward road`. Show the Muse line at the alert time plus the measured Muse response time.
  4. Counters: animals tracked, alerts raised, frames analysed, and the words being looked for.
- **No overlaps:** nothing on the panel may overlap: the counters, the lead-time bar and its labels each get their own row, and event-log lines wrap instead of being cut off.
- **Bottom strip:** small grey text: `Recorded footage · simulated road sign · SAM 3.1 + Muse Spark`.

**Driving details on the demo panel.** Under the status block, show the time to reach as a large number (`6.4 s`) and the distance (`≈ 180 m`). Under the counters, add a lead-time bar marking *first seen*, *warning* and *vehicle reaches animal*, with the seconds between them. Scale it to the event (from 1 s before first seen to 1 s after the vehicle reaches the animal), not the whole clip, so the three labels never overlap. In the event log, use lines like `00:02.1 deer-0 first seen, ≈ 210 m, 14 px tall, left roadside`, `00:03.0 deer-0 moving toward road` and `00:03.6 SLOW DOWN, ≈ 150 m, 5.1 s to reach`. Use plain hyphens instead of middle dots in any text drawn on video.

**Try any animal.** On the web page, a **Look for** text box takes any animal word (for example `horse`). Clicking **Run** makes a new SAM 3.1 call only for words that aren't cached yet.

**Results screen.** After `--batch`, write `output/results.png` (1920×1080, same style): one row per clip with its verdict (a green tick if correct, a red cross if not), then the catch rate, number of false alarms, average alert delay after entering the road, total frames analysed, total API cost and the processing time per second of video (`realtime_factor`). Show the same on a **Results** tab of the web page.

**Showcase reel.** `python app.py --showcase` builds `output/showcase.mp4` from the existing `demo_panel.mp4` files, with no new API calls. Order: `no_animal` clips first, then `beside_road` clips, then crossing, on-road and collision clips, and finally `results.png` held for 6 s. Before each clip, show a 2-second title card, `Scene N · <plain description from the file name>`, for example `Scene 1 · Normal traffic, no animals`. Add a **Demo** tab to the web page that plays the reel.

In the showcase, at each clip's first sighting, hold the frame for 1.5 s with the ring, the inset and the caption `Spotted at ≈ 210 m - 7.2 s before reaching it`, then play on. Order the clips `no_animal`, then `beside_road`, then `crossing`, `on_road` and `collision`, with night and dusk clips last in each group.

## Product dashboard

Build one more page: the operations dashboard a road authority would use, combining the real results from `test_videos/` with sample network data. `python app.py --dashboard` writes `output/dashboard.html`, one self-contained page that opens in any browser with no server and no network calls except Google Fonts. Also show it on a **Dashboard** tab of the web page. If `dashboard_mockup.png` is in the project folder, match it closely.

**Look:** dark, built for recording at 1920×1080. Fonts: IBM Plex Sans, with IBM Plex Mono for times and numbers. Colours: background `#0B1118`, panels `#111A24` with a `#1F2A36` border, green `#2FBF71`, amber `#F5A524`, red `#FF4D5E` (fills `#D7263D`).

**Layout, top to bottom:**

1. **Top bar:** the name `Wildlife Road Alert`, `Network · Southern Manitoba`, three count chips (red active alerts, amber animals nearby, green roads clear), the date and time, and a dashed tag `CONCEPT · SAMPLE DATA`.
2. **Map panel (left, wide):** a schematic map of southern Manitoba drawn in SVG, showing Lake Winnipeg, Winnipeg, Steinbach and highways PTH 1, 12, 44, 52 and 59, with the Eastman region shaded amber and labelled. Each camera is a dot coloured by its state; the two sample roadside cameras are squares. Road and place labels must not overlap; offset them where roads cross. The alerting camera has a pulsing red dot and a label. Add a legend underneath.
3. **Right column:**
   - an **active alert** card (red): road, camera, how many seconds after entering the road it was raised, Muse's species and confidence, group count, severity, assurance, the heading, the road-sign text in amber capitals on black, and two buttons, `Open camera` and `Mark false alarm`
   - an **event log**, newest first, in the same wording as the demo screen
4. **Live cameras row:** 4 tiles, each looping one clip's `demo.mp4`, with a border in the camera's state colour and a badge (`ALERT`, `NEARBY` or `CLEAR`).
5. **Bottom row:**
   - a **driver warning**: a phone-style notification reading `Wildlife ahead · Deer on PTH 12, 2 km ahead. Slow down.`
   - **hotspots this week**: road segments with bar lengths by count of animals on the road
   - **network today**: cameras online, alerts raised, average alert delay, and false alarms marked
6. **Footer:** `Concept screen with sample data · detection by SAM 3.1, checks by Muse Spark (Meta)`.

**Real versus sample data.** The camera tiles, the active alert, the event log and the average alert delay come from real runs: use the clip with the clearest alert as the active one. The map, the other cameras, the hotspots and the network counts come from `sample_network.json`. Create that file with 26 cameras (id, road, map position, state) so I can edit it. Clicking a camera tile opens its `demo_panel.mp4`.

When it's built, give me the path to `output/dashboard.html`.

## Rules

- Never hard-code, print or log the API key. If `MODEL_API_KEY` is missing, stop with one line explaining how to set it.
- Every error I could see is a single plain-English line saying what to do. That covers an unsupported video format, 401/403 (key or billing), 404 (model not found) and 429 (rate limit).
- Retry 429 and 5xx errors up to 3 times with backoff. Never retry a 400.
- Never call Muse per frame, and never exceed `MAX_MUSE_CALLS`.
- Re-running the same video after changing only the zone or thresholds must not call SAM 3.1 again: use the cache.
- Keep it small: a few Python files, no database, no framework beyond Gradio.
- Don't download anything this spec doesn't name. Clips come from me.
- Don't invent API fields. If a call fails because the API differs from this spec, check the reference docs, adapt, and tell me what you changed.

**Cost.** Count billed frames from every SAM 3.1 request, the road call included (frames sent, with the 50-frame minimum per request), and add Muse token costs, using the prices in **Stack**. A run that called SAM 3.1 must never report 0 frames billed or $0.00. The last results screen showed both, so fix that.

## Done when

**Automated tests pass**, run with `pytest`, making no API calls:

- **Warning levels:** with fake tracks and a fake road mask, no animal gives green; an animal in the roadside band gives amber; an animal on the road for 0.3 s gives red and fires exactly one alert; an animal away from the road and not moving toward it never raises the level; after the animal leaves, the level steps down only once the lower level has held for 1 s.
- **Time to reach:** a fake box whose height follows a vehicle approaching at constant speed (height = k ÷ distance) gives a time to reach within 10% of the true value; a box that isn't growing gives `-`.
- **Distance:** a 1.4 m deer 100 px tall in a 1080 px-high frame with a 55° field of view gives about 14.5 m.
- **SAM output:** the docs' example line (frame 0, two objects) parses into two tracks with the correct boxes.
- **Muse reply:** JSON wrapped in extra text or code fences still parses, and a broken reply falls back to the default sign text.
- **More may follow:** after an animal leaves the roadside band, the level stays at amber or above, reading `MORE MAY FOLLOW`, for `GROUP_HOLD_SECONDS`, then steps down; two tracks in one frame give a group count of 2.
- **Duplicates:** two tracks from different words that overlap with IoU ≥ 0.5 in most shared frames become one track that keeps every frame of both, starts at the earlier of the two first frames, and is labelled with the more confident word.
- **Severity and assurance:** a moose alert gets severity 3; an alert with mean confidence 0.8, tracked 1.5 s and confirmed by Muse gets High; the same alert without Muse gets Medium; an unknown word gets severity 1.
- **Approval:** the Waze feed leaves out unapproved and sample sightings, and includes a sighting once `--approve` has been run for it.
- **Species from Muse:** a confirmed Muse species at 0.6 or more replaces SAM's word for the label, severity and distance (a track found as elk and confirmed as white-tailed deer uses the deer height in every frame), and SAM's word stays in `detected_as`; under 0.6, or when Muse rejects it, SAM's word stays.
- **Display scaling:** a box from the analysis copy lands in the matching place on the display copy.
- **Speed:** `realtime_factor` equals the summed call durations divided by the seconds analysed, for a fake log.

**The live run works** on the clips in `test_videos/` (outside `archive/`). File names say what should happen; the naming rule and expected results are in **Driving scenario**. The clips are `traffic_no_animal_01.mp4` and `dashcam_deer_crossing_night_01.mp4`.

- On every `crossing`, `on_road` and `collision` clip, a red alert fires before the vehicle reaches the animal, and its lead time is reported. If the animal was visible beside the road first, amber appears before red.
- `beside_road` clips reach amber but never red, and `no_animal` clips stay green.
- The yellow road outline follows the real road in every clip.
- Far animals get the ring and inset, the masks follow the animal, and the banner text comes from Muse.

**Report back** with a table giving, for each clip: video, expected result (from the file name), alert fired, first alert time, species, and a verdict of *correct alert*, *correct silence*, *missed* or *false alarm*. Below it, state the catch rate (correct alerts ÷ clips that should alert), the number of false alarms, and the total API cost from frames billed and Muse calls.

Add three columns to that table: first seen (time, ≈ distance and box height in px), highest level reached, and warning lead time. Under it, give the average lead time and the `realtime_factor` for each clip.

For the night crossing clip, also give the first-sighting check and the highway comparison, and the paths to `opening_plain.mp4`, `opening_alert.mp4`, `ending_plain.mp4` and `ending_alert.mp4`. Report the thermal try-out separately: what SAM 3.1 found in each image, with the overlay paths and `thermal_samples/SOURCES.md`.

Finally, run `python app.py --showcase` and `python app.py --dashboard`, and give me the paths to `output/showcase.mp4`, `output/results.png` and `output/dashboard.html`.

## Public repository

The competition requires the code in a public GitHub repository that anyone can open. Prepare the project for it; push only after I've logged in to GitHub. Keep it private until submission day.

- **Licence:** add `LICENSE` with the MIT licence, copyright the project owner.
- **`.gitignore`:** leave out everything that isn't ours to share or is generated: `output/`, `previews/`, every video file (`*.mp4`, `*.mov`) and the clips in `test_videos/`, `Deers on roads.v4i.coco/`, the images in `thermal_samples/` (keep `SOURCES.md`), the audio in `voice_samples/`, `.env` files, caches and virtual environments. Keep `test_videos/CLIP_LIST.md`.
- **Results folder:** copy into `docs/results/` the evaluation chart, the examples grid, `eval_summary.json`, `results.png` and the thermal overlays, each with its credit line in `docs/results/CREDITS.md`. Don't include anything drawn from the Pond5 clip.
- **README:** start with one paragraph on what the system does (video tracking with SAM 3.1, road and roadside rules, warning levels, the sighting network with Waze feed, voice reports with Muse Voice Transcribe, Muse Spark checks and the Muse Wonder briefing), then setup, how to get the data (the Pond5 item number, the Roboflow dataset link, `thermal_samples/SOURCES.md`), how to run each command, the measured results with their sources, and the credits.
- **Secret check:** before the first commit, search every file to be committed for anything that looks like an API key or token, and confirm `MODEL_API_KEY` appears only as an environment-variable name. Show me the result.
- **Push as private:** once I've run `gh auth login`, create a **private** repository named `wildlife-road-alert` under my GitHub organization `foresighthouseai` with `gh repo create foresighthouseai/wildlife-road-alert --private --source . --push`, then give me its link. Don't add repository topics, and keep the phrase "Meta Global AI Hackathon 2026" out of the repository's short description (the README may mention the hackathon).
- **Make public on submission day only:** when I say so, run `gh repo edit --visibility public --accept-visibility-change-consequences`, then confirm the link opens in a private browser window without logging in.

## Reference docs

Check these before guessing any API detail:

- [Segmenting with prompts](https://dev.meta.ai/docs/sam/segmenting): SAM 3.1 request format, video upload and limits
- [Reading segmentation output](https://dev.meta.ai/docs/sam/reading-segmentation): box and mask grammar, and object ids across frames
- [Client libraries](https://dev.meta.ai/docs/sam/client-libraries): `meta-sam-parser` and mask decoding
- [SAM 3.1 model page](https://dev.meta.ai/models/sam-3-1): model ID and pricing
- [Image understanding](https://ai.developer.meta.com/docs/features/image-understanding): sending images to Muse with Chat Completions
- [Meta Model API: get started](https://ai.developer.meta.com/docs/): base URL and authentication
