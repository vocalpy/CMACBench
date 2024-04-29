import pathlib

# ---- constants for argparser -----------------------------------------------
PREP_STAGES = (
    "all",
    "mkdirs",
    "copy",
    "labels",
    "qc",
    "make",
    "split",
    "clean",
)


BIOSOUND_GROUPS = (
    'Bengalese-Finch-Song',
    'Canary-Song',
    'Mouse-Pup-Call',
    'Zebra-Finch-Song',
    'Human-Speech'
)


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

# ------------- Birdsong-Recognition dataset
BIRDSONGREC_RAW_DATA = BF_RAW_DATA / "BirdsongRecognition"
BIRDSONGREC_BIRD_IDS = (
    "Bird0",
    "Bird4",
    "Bird7",
    "Bird9",
)
BIRDSONGREC_ANNOT_ROOT = BIRDSONGREC_RAW_DATA / "annotation_tweetynet"

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
MOUSE_PUP_CALL_RAW_DATA = RAW_DATA_ROOT / "Mouse-Pup-Call"

# ------------- Jourjine et al. 2023 dataset
JOURJINE_ET_AL_2023_DATA = MOUSE_PUP_CALL_RAW_DATA / "Jourjine-et-al-2023"
JOURJINE_ET_AL_2023_SEGS_DIR = JOURJINE_ET_AL_2023_DATA / "processed_data" / "supplemental_figure_5"
JOURJINE_ET_AL_2023_MOUSE_SPECIES = (
    'BK',
    'MU',
)

# --------- Zebra finch song
ZB_RAW_DATA = RAW_DATA_ROOT / "Zebra-Finch-Song"

# ------------- Steinfath et al. 2021 dataset
STEINFATH_ET_AL_2021_RAW_DATA = ZB_RAW_DATA / "Steinfath-et-al-2021-DAS-Zebra-finch-train-and-test-data"
STEINFATH_ET_AL_2021_ZB_BIRD_ID = "blu285"

# -------- Human speech
SPEECH_RAW_DATA = RAW_DATA_ROOT / "Human-Speech"

# ------------ TIMIT NLTK sample
TIMIT_NLTK_RAW = SPEECH_RAW_DATA / "TIMIT-corpus-sample-from-NLTK" / "timit"

# ------------ TIMIT full corpus
TIMIT_FULL_CORPUS_RAW = SPEECH_RAW_DATA / "TIMIT-corpus-full" / "data"


# ---- root for dataset we are making,
# and species-specific sub-dirs
DATASET_ROOT = DATA_DIR / "BioSoundSegBench"

# ---- all the data from datasets with unpermissive licenses goes here
DATA_WE_CANT_SHARE = DATASET_ROOT / "DATA-WE-CANT-SHARE"

# -------- Bengalese finch song
BF_DATA_DST = DATASET_ROOT / "Bengalese-Finch-Song"

# -------- Canary song
CANARY_DATA_DST = DATASET_ROOT / "Canary-Song"

# -------- Mouse pup calls
MOUSE_PUP_CALL_DATA_DST = DATASET_ROOT / "Mouse-Pup-Call"

# -------- Zebra finch song
ZB_DATA_DST = DATASET_ROOT / "Zebra-Finch-Song"


# -------- Human speech

# -------- human speech data we can't share
HUMAN_SPEECH_WE_CANT_SHARE = DATA_WE_CANT_SHARE / "Human-Speech"

SPEECH_DATA_DST = DATASET_ROOT / "Human-Speech"

# ----- metadata -----------------------------------------------------------------------------------

LABELSETS_JSON = DATASET_ROOT / "labelsets.json"
LABELMAPS_JSON = DATASET_ROOT / "labelmaps.json"
SPLITS_DIR = DATASET_ROOT / "splits"
INPUTS_TARGETS_PATHS_CSVS_DIR = SPLITS_DIR / "inputs-targets-paths-csvs"
SAMPLE_ID_VECTORS_DIR = SPLITS_DIR / "sample-id-vectors"
INDS_IN_SAMPLE_VECTORS_DIR = SPLITS_DIR / "inds-in-sample-vectors"
SPLITS_JSONS_DIR = SPLITS_DIR / "splits-jsons"
