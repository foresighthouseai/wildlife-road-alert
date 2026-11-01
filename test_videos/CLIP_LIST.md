# Test clips

I add every clip to this folder myself. Don't download clips.
Only footage filmed from a moving vehicle counts. Old side-view clips stay in `archive/` and are left out of every run.
The folder `previews/` in the project holds a watermarked test copy: never use it.

## Clips in use

| File | What it is | Expected result |
| --- | --- | --- |
| `traffic_no_animal_01.mp4` | Dashcam, normal highway traffic, no animals (Pexels, free licence) | Stays green |
| `dashcam_deer_crossing_night_01.mp4` | Dashcam at night: a deer appears in the headlights, already on the road about 40 m ahead, and crosses to the right shoulder (Pond5 item 141272328, commercial licence, bought Oct 31, 2026) | Red while the deer is on the road |

## Naming rule for any later clip

`dashcam_<species>_<event>_<light>_<nn>.mp4`

- event: `crossing`, `on_road`, `beside_road`, `collision` or `no_animal`
- light: `day`, `dusk` or `night`

Expected results: `crossing`, `on_road` and `collision` reach red; `beside_road` reaches amber but never red; `no_animal` stays green.

A `collision` clip also needs its impact time, in seconds from the start of the clip, in `impact_times.json` in this folder.
