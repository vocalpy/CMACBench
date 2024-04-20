import pathlib

# ---- constants for argparser -----------------------------------------------
PREP_STAGES = [
    "all",
    "mkdirs",
    "copy",
    "make",
    "clean"
]


# ---- directories -----------------------------------------------------------
DATA_DIR = pathlib.Path('./data')

# ---- raw data root, and species-specific raw data dirs
RAW_DATA_ROOT = DATA_DIR / "raw"

# --------- Bengalese finch song
BF_RAW_DATA = RAW_DATA_ROOT / "Bengalese-Finch-Song"

# ------------- Bengalese Finch Song Repository
BFSONGREPO_RAW_DATA = BF_RAW_DATA / "bfsongrepo"
BFSONGREPO_BIRD_IDS = (
    "bl26lb16",
    "gr41rd51",
    "gy6or6",
    "or60yw70",
)

# --------- Canary song
CANARY_RAW_DATA = RAW_DATA_ROOT / "Canary-Song"

# ------------- TweetyNet dataset
TWEETYNET_RAW_DATA = CANARY_RAW_DATA / "tweetynet-canary"
TWEETYNET_BIRD_IDS = (
    'llb3',
    'llb11',
    'llb16',
)

# -------- Mouse pup calls
MOUSE_RAW_DATA = RAW_DATA_ROOT / "Mouse-Pup-Calls"

# ------------- Jourjine et al. 2023 dataset
JOURJINE_ET_AL_2023_DATA = MOUSE_RAW_DATA / "Jourjine-et-al-2023"
JOURJINE_ET_AL_2023_SEGS_DIR = JOURJINE_ET_AL_2023_DATA / "processed_data" / "supplemental_figure_5"

# --------- Zebra finch song
ZB_RAW_DATA = RAW_DATA_ROOT / "Zebra-Finch-Song"

# ------------- Steinfath et al. 2021 dataset
STEINFATH_ET_AL_2021_RAW_DATA = ZB_RAW_DATA / "Steinfath-et-al-2021-DAS-Zebra-finch-train-and-test-data"
STEINFATH_ET_AL_2021_ZB_BIRD_ID = "blu285"

# ---- root for dataset we are making,
# and species-specific sub-dirs
DATASET_ROOT = DATA_DIR / "BioSoundSegBench"

# -------- Bengalese finch song
BF_DATA_DST = DATASET_ROOT / "Bengalese-Finch-Song"

# -------- Canary song
CANARY_DATA_DST = DATASET_ROOT / "Canary-Song"

# -------- Mouse pup calls
MOUSE_PUP_CALL_DATA_DST = DATASET_ROOT / "Mouse-Pup-Calls"

# -------- Zebra finch song
ZB_DATA_DST = DATASET_ROOT / "Zebra-Finch-Song"

# -------- Human speech
SPEECH_DATA_DST = DATASET_ROOT / "Human-Speech"

