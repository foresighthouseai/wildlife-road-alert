# Thermal try-out sources

Downloaded Oct 28, 2026. Only files whose pages state a free licence
(CC BY, CC BY-SA, CC0 or public domain) are used.

| File | Source page | Author | Licence |
| --- | --- | --- | --- |
| thermal_deer_wiki_01.jpg | https://commons.wikimedia.org/wiki/File:Wiki_deer.jpg | Arno / Coen | CC BY-SA 3.0 |
| thermal_deer_wiki_02.jpg | https://commons.wikimedia.org/wiki/File:Wiki_deer2.jpg | Arno / Coen | CC BY-SA 3.0 |
| thermal_moose_wiki_01.jpg | https://commons.wikimedia.org/wiki/File:Thermal_image_of_a_cow_moose_during_winter.jpg | Zaereth | CC BY-SA 4.0 |
| thermal_deer_flir_01.jpg | https://commons.wikimedia.org/wiki/File:A_deer_in_the_woods._It_was_pitch_black,_but_this_camera_does_not_use_ANY_visible_light._FLIR_infrared._(6968297826).jpg | Mark Taylor from Rockville, USA | CC BY 2.0 |
| thermal_deer_flir_02.jpg | https://commons.wikimedia.org/wiki/File:Deer_in_the_woods._FLIR_infrared.jpg | Mark Taylor | CC BY 2.0 |
| thermal_deer_bambi_01.jpg | https://zenodo.org/records/15532500 (BAMBI Re-ID, `reid.zip`: reid/reid/train/object1/camera0/001049.jpg) | Coucke, Schedl, Praschl, Maschek et al. | CC BY 4.0 |
| thermal_deer_bambi_02.jpg | https://zenodo.org/records/15532500 (BAMBI Re-ID, `reid.zip`: reid/reid/train/object10/camera0/007788.jpg) | Coucke, Schedl, Praschl, Maschek et al. | CC BY 4.0 |
| thermal_deer_bambi_03.jpg | https://zenodo.org/records/15532500 (BAMBI Re-ID, `reid.zip`: reid/reid/train/object100/camera0/006131.jpg) | Coucke, Schedl, Praschl, Maschek et al. | CC BY 4.0 |
| thermal_deer_bambi_04.jpg | https://zenodo.org/records/15532500 (BAMBI Re-ID, `reid.zip`: reid/reid/query/object43/camera1/002499.jpg) | Coucke, Schedl, Praschl, Maschek et al. | CC BY 4.0 |

Run with: `python app.py --try-images thermal_samples/ --prompt deer`
Overlays and counts land in `output/try/`. This is a visual check only
(no human boxes), and it is not part of the showcase or the dashboard.
