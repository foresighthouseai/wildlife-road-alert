"""All settings for Wildlife Road Alert. Edit values here, then re-run."""

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
ROAD_ZONE = [(0.35, 0.55), (0.65, 0.55), (1.0, 1.0), (0.0, 1.0)]        # fallback only, if SAM finds no road
USE_MUSE_CHECK = True
MUSE_MODEL = "muse-spark-1.2"
MAX_MUSE_CALLS = 20                         # cap per video

SIGHTING_RADIUS_KM = 10      # warn vehicles heading toward a sighting within this distance
SIGHTING_TTL_MIN = 20        # how long a sighting stays active
DEMO_NOW = "latest"          # dashboard/Waze "now": "latest" = latest camera sighting + 2 min; "" = real time; or an ISO time
REACTION_S = 1.5             # typical driver reaction time
DECEL_DRY = 6.9              # braking, m/s2, dry road
DECEL_SNOW = 2.5             # braking, m/s2, snow
LOW_BEAM_M = 50              # typical low-beam reach
VOICE_MODEL = "muse-voice-transcribe-1.0"
BRIEFING_MODEL = "muse-wonder-1.0"

# Muse prices per million tokens: (input, output) in USD.
MUSE_PRICES = {"muse-spark-1.2": (1.25, 4.25), "muse-wonder-1.0": (4.0, 20.0)}

# Road-zone presets: name -> polygon as (x, y) fractions of frame width/height.
ROAD_ZONE_PRESETS = {
    "Dashcam": [(0.35, 0.55), (0.65, 0.55), (1.0, 1.0), (0.0, 1.0)],
    "Roadside camera": [(0.0, 0.4), (1.0, 0.4), (1.0, 0.75), (0.0, 0.75)],
    "Road narrowing ahead": [(0.4, 0.5), (0.6, 0.5), (1.0, 1.0), (0.0, 1.0)],
}

API_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
DEFAULT_SIGN_TEXT = "WILDLIFE ON ROAD - SLOW DOWN"
MUSE_PROMPT = (
    "You are checking an alert from a wildlife road-safety camera.\n"
    "Image 1 is a close-up of the detected object. Image 2 is the full camera view; the road edges are outlined in yellow.\n"
    "Reply with only a JSON object:\n"
    '{"is_animal": true, "species": "white-tailed deer", "confidence": 0.9,\n'
    ' "heading_toward_road": "yes" | "no" | "unclear",\n'
    ' "sign_text": "DEER ON ROAD - SLOW DOWN"}\n'
    "sign_text: 6 words or fewer, capital letters, suitable for a roadside warning sign."
)
DATASET_CREDIT = "Data: Deers on roads (Scalersai, Roboflow Universe), CC BY 4.0"
DATASET_NOTE = "Images are 416x416 and stretched; scores may be lower than on full-resolution video"
