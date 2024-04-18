#!/usr/bin/env python
# coding: utf-8

# # 1. Make dataset folders; copy audio, copy/make annotation
# 
# This script makes the BioSoundSegBench dataset folders, copies audio files into it, and either copies annotation files or converts them as needed.

# In[1]:


import pathlib
import shutil

import crowsetta
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


# In[2]:


cd ../../..


# In[3]:


DATA_DIR = pathlib.Path('./data')
assert DATA_DIR.exists(), "couldn't find DATA_DIR"


# In[4]:


RAW_DATA_ROOT = DATA_DIR / "raw"
DATASET_ROOT = DATA_DIR / "BioSoundSegBench"


# In[5]:


DATASET_ROOT.mkdir(exist_ok=True)


# ## Dry run
# 
# Set this flag to `False` to actually copy/make files

# In[6]:


DRY_RUN = False


# ## Bengalese finch song

# In[7]:


BF_RAW_DATA = RAW_DATA_ROOT / "Bengalese-Finch-Song"


# In[8]:


BF_DATA_DST = DATASET_ROOT / "Bengalese-Finch-Song"
BF_DATA_DST.mkdir(exist_ok=True)


# ### Bengalese finch song repository
# 
# For each bird, copy all wav and csv files

# In[9]:


BFSONGREPO_RAW_DATA = BF_RAW_DATA / "bfsongrepo"


# In[10]:


BFSONGREPO_BIRD_IDS = (
    "bl26lb16",
    "gr41rd51",
    "gy6or6",
    "or60yw70",
)


# In[11]:


for bird_id in BFSONGREPO_BIRD_IDS:
    print(
        f"Copying audio and annotation files for bird ID {bird_id} from BFSongRepo"
    )
    bird_id_dir = BFSONGREPO_RAW_DATA / bird_id
    assert bird_id_dir.exists(), f"Couldn't find bird_id_dir: {bird_id_dir}"

    bird_id_dst = BF_DATA_DST / f"bfsongrepo-{bird_id}"
    bird_id_dst.mkdir(exist_ok=True)

    # bfsongrepo is split up by days; we throw away that split here
    day_dirs = [
        subdir for subdir in bird_id_dir.iterdir() if subdir.is_dir()
    ]
    for day_dir in day_dirs:
        print(
            f"Copying audio and annotation files from day: {day_dir.name}"
        )
        wav_paths = sorted(day_dir.glob("*wav"))
        print(
            f"Found {len(wav_paths)} audio files."
        )
        csv_paths = sorted(day_dir.glob("*csv"))
        print(
            f"Found {len(csv_paths)} annotation files."
        )
        wav_paths_filtered = []
        csv_paths_filtered = []
        for wav_path in wav_paths:
            csv_path_that_should_exist = wav_path.parent / (wav_path.name + '.csv')
            if csv_path_that_should_exist in csv_paths:
                wav_paths_filtered.append(wav_path)
                csv_paths_filtered.append(csv_path_that_should_exist)
        print(
            "After filtering audio files by whether or not they have an annotation file, "
            f"there are {len(wav_paths_filtered)} audio files and {len(csv_paths_filtered)} annotation files."
        )
        if not DRY_RUN:
            pbar = tqdm(
                zip(wav_paths_filtered, csv_paths_filtered)
            )
            for wav_path, csv_path in pbar:
                shutil.copy(wav_path, bird_id_dst)
                # NOTE we rename to include the unit in the annotation file name
                csv_path_dst = bird_id_dst / (csv_path.stem + '.syllable.csv')
                shutil.copy(csv_path, csv_path_dst)


# ### Birdsong Recognition
# 
# For each bird:
# 1. copy all wav files
# 2. generate the `"simple-seq"` annotation files from the source annotation files.
# 3. check that there is an annotation file for every audio file

# In[12]:


BIRDSONGREC_BIRD_IDS = (
    "Bird0",
    "Bird4",
    "Bird7",
    "Bird9",
)


# In[13]:


BIRDSONGREC_RAW_DATA = BF_RAW_DATA / "BirdsongRecognition"


# In[14]:


for bird_id in BIRDSONGREC_BIRD_IDS:
    print(
        f"Copying audio and annotation files for bird ID {bird_id} from Birdsong-Recognition dataset"
    )
    bird_id_dir = BIRDSONGREC_RAW_DATA / bird_id / "Wave"
    assert bird_id_dir.exists(), f"Couldn't find bird_id_dir: {bird_id_dir}"

    bird_id_dst = BF_DATA_DST / f"birdsongrec-{bird_id}"
    bird_id_dst.mkdir(exist_ok=True)

    wav_paths = sorted(bird_id_dir.glob("*wav"))
    print(
        f"Found {len(wav_paths)} audio files."
    )
    if not DRY_RUN:
        for wav_path in tqdm(wav_paths):
            shutil.copy(wav_path, bird_id_dst)


# Convert annotation files from `"yarden"` to `"simple-seq"`

# In[15]:


BIRDSONGREC_ANNOT_ROOT = BIRDSONGREC_RAW_DATA / "annotation_tweetynet"
assert BIRDSONGREC_ANNOT_ROOT.exists(), f"Couldn't find BIRDSONGREC_ANNOT_ROOT: {BIRDSONGREC_ANNOT_ROOT}"


# In[16]:


annot_mats = sorted(BIRDSONGREC_ANNOT_ROOT.glob('*annotation.mat'))


# In[17]:


scribe = crowsetta.Transcriber(format='yarden')


for annot_mat in annot_mats:
    bird_id = annot_mat.name.split('-')[1][:5]
    bird_id_dst = BF_DATA_DST / f"birdsongrec-{bird_id}"
    assert bird_id_dst.exists(), f"Couldn't find `bird_id_dst`: {bird_id_dst}"
    yarden = scribe.from_file(annot_mat)
    annots = []
    pbar = tqdm(
        zip(yarden.annotations, yarden.audio_paths)
    )
    for filenum, (annot_arr, audio_path) in enumerate(pbar):
        annot_tup = annot_arr.tolist()
        onsets = annot_tup[1]
        offsets = annot_tup[2]
        labels = np.array([str(lbl) for lbl in annot_tup[3]])
        simple_seq = crowsetta.formats.seq.SimpleSeq(
            onsets_s=onsets, 
            offsets_s=offsets, 
            labels=labels,
            annot_path=annot_mat
        )
        # NOTE: we assume here that the annotations are in sorted numerical order;
        # I inspected manually and confirmed that this was the case.
        # The `audio_paths` in the annotation are given by the GUI, 
        # and are not the original audio file names
        if not DRY_RUN:
            simple_seq.to_file(annot_path=bird_id_dst / f'{filenum}.wav.syllable.csv')


# Verify we have a csv file for every wav file

# In[18]:


for bird_id in BIRDSONGREC_BIRD_IDS:
    print(
        f"Verifying each audio file has an annotation file for bird ID {bird_id} from Birdsong-Recognition dataset."
    )

    bird_id_dst = BF_DATA_DST / f"birdsongrec-{bird_id}"

    wav_paths = sorted(bird_id_dst.glob("*wav"))
    csv_paths = sorted(bird_id_dst.glob("*csv"))
    assert all(
        [
            wav_path.parent / (wav_path.name + ".syllable.csv") in csv_paths
            for wav_path in wav_paths
        ]
    )
    assert len(wav_paths) == len(csv_paths)
    print("Verified.")


# ## Canary Song

# In[19]:


CANARY_DATA_DST = DATASET_ROOT / "Canary-Song"
CANARY_DATA_DST.mkdir(exist_ok=True)


# In[20]:


CANARY_RAW_DATA = RAW_DATA_ROOT / "Canary-Song"


# ### TweetyNet dataset

# In[21]:


TWEETYNET_RAW_DATA = CANARY_RAW_DATA / "tweetynet-canary"


# In[22]:


TWEETYNET_BIRD_IDS = (
    'llb3',
    'llb11',
    'llb16',
)


# Copy all audio files

# In[23]:


for bird_id in TWEETYNET_BIRD_IDS:
    print(
        f"Copying audio files for bird ID {bird_id} from TweetyNet dataset"
    )
    bird_id_dir = TWEETYNET_RAW_DATA / f"{bird_id}_data" / f"{bird_id}_songs"
    assert bird_id_dir.exists(), f"Couldn't find bird_id_dir: {bird_id_dir}"

    bird_id_dst = CANARY_DATA_DST / f"tweetynet-{bird_id}"
    bird_id_dst.mkdir(exist_ok=True)

    wav_paths = sorted(bird_id_dir.glob("*wav"))
    print(
        f"Found {len(wav_paths)} audio files."
    )
    if not DRY_RUN:
        for wav_path in wav_paths:
            shutil.copy(wav_path, bird_id_dst)


# Make annotation files.
# We convert from `"generic-seq"` (single file for all annotations) to `"simple-seq"` (one annotation file per one audio file)

# In[24]:


scribe = crowsetta.Transcriber(format='generic-seq')


for bird_id in TWEETYNET_BIRD_IDS:
    print(
        f"Making annotation files for bird ID {bird_id} from TweetyNet dataset"
    )
    bird_id_dir = TWEETYNET_RAW_DATA / f"{bird_id}_data"
    annot_csv_path = bird_id_dir / f"{bird_id}_annot.csv"
    df = pd.read_csv(annot_csv_path)
    # we need to fix column names to avoid a validation error
    df = df.rename(
        columns={
            'onset_Hz': 'onset_sample',
            'offset_Hz': 'offset_sample',
            'annot_file': 'annot_path',
            'audio_file': 'notated_path',
        }
    )
    # we don't care if we overwrite the fixed csv 
    # every time we do this step of creating the dataset
    annot_csv_dst = bird_id_dir / f"{bird_id}-generic-seq-annot.csv"
    df.to_csv(annot_csv_dst, index=False)

    annots = scribe.from_file(annot_csv_dst).to_annot()

    bird_id_dst = CANARY_DATA_DST / f"tweetynet-{bird_id}"
    assert bird_id_dst.exists()

    print(
        f"Converting {len(annots)} annotations to simple-seq format."
    )
    for annot in tqdm(annots):
        simple_seq = crowsetta.formats.seq.SimpleSeq(
            onsets_s=annot.seq.onsets_s, 
            offsets_s=annot.seq.offsets_s, 
            labels=annot.seq.labels,
            annot_path=annot.annot_path,
        )
        if not DRY_RUN:
            simple_seq.to_file(annot_path=bird_id_dst / f'{annot.notated_path.name}.syllable.csv')


# For this dataset, not every audio file has an annotation file.
# We do the clean up step of removing audio files with annootations here.

# In[25]:


for bird_id in TWEETYNET_BIRD_IDS:
    print(
        f"Removing audio files without an annotation file for bird ID {bird_id} from Birdsong-Recognition dataset."
    )

    bird_id_dst = CANARY_DATA_DST / f"tweetynet-{bird_id}"

    wav_paths = sorted(bird_id_dst.glob("*wav"))
    csv_paths = sorted(bird_id_dst.glob("*.syllable.csv"))

    wav_without_annot_csv = [
        wav_path
        for wav_path in wav_paths
        if wav_path.parent / (wav_path.name + ".syllable.csv") not in csv_paths
    ]

    print(
        f"Found {len(wav_paths)} audio files and {len(csv_paths)} annotation files."
        f"Remvoing {len(wav_without_annot_csv)} files without annotations."
    )
    if not DRY_RUN:
        for wav_path in wav_without_annot_csv:
            wav_path.unlink()    


# ## Mouse Pup Calls

# ### Jourjine et al 2023 dataset

# In[26]:


MOUSE_RAW_DATA = RAW_DATA_ROOT / "Mouse-Pup-Calls"


# In[27]:


MOUSE_DATA_DST = DATASET_ROOT / "Mouse-Pup-Calls"
MOUSE_DATA_DST.mkdir(exist_ok=True)


# In[28]:


JOURJINE_ET_AL_2023_DATA = MOUSE_RAW_DATA / "Jourjine-et-al-2023"


# In[29]:


ANNOT_DIR = JOURJINE_ET_AL_2023_DATA / "processed_data" / "supplemental_figure_5"


# In[30]:


segs_csv_path = ANNOT_DIR / "all_development_vocs_with_start_stop_times.csv"
segs_df = pd.read_csv(segs_csv_path)


# For this dataset we start with the annotations since there are less of those than audio files, and since we need to build the annotations on the fly.
# 
# We treat this as a binary classificaiton task: "vocalization" or "no vocalization". So we just label all the detected segments as "v" (for "vocalization")

# In[31]:


for source_file in tqdm(segs_df.source_file.unique()):
    this_file_segs_df = segs_df[
        segs_df.source_file == source_file
    ]
    if len(this_file_segs_df) > 0:
        species_id = source_file.split('_')[0]
        species_id_dst = MOUSE_DATA_DST / f"jourjine-et-al-2023-{species_id}"
        if not DRY_RUN:
            species_id_dst.mkdir(exist_ok=True)
        wav_path = JOURJINE_ET_AL_2023_DATA / f"development{species_id}" / f"development_{species_id}" / source_file
        if wav_path.exists():
            simple_seq = crowsetta.formats.seq.SimpleSeq(
                onsets_s=this_file_segs_df.start_seconds.values,
                offsets_s=this_file_segs_df.stop_seconds.values,
                labels=np.array(['v'] * this_file_segs_df.stop_seconds.values.shape[-1]),
                annot_path='dummy',
            )
            if not DRY_RUN:
                shutil.copy(wav_path, species_id_dst)
                csv_path = species_id_dst / f"{source_file}.call.csv"
                simple_seq.to_file(csv_path)


# ## Zebra finch song

# In[32]:


ZB_DATA_DST = DATASET_ROOT / "Zebra-Finch-Song"
ZB_DATA_DST.mkdir(exist_ok=True)


# In[33]:


ZB_RAW_DATA = RAW_DATA_ROOT / "Zebra-Finch-Song"


# ### Steinfath et al. 2021 dataset, from AVA dataset

# In[34]:


STEINFATH_ET_AL_2021_RAW_DATA = ZB_RAW_DATA / "Steinfath-et-al-2021-DAS-Zebra-finch-train-and-test-data"


# In[35]:


ZB_BIRD_ID = "blu285"


# In[36]:


print(
    f"Copying audio and annotation files for bird ID {ZB_BIRD_ID} from Steinfath et al. 2021 dataset"
)
# only one zebra finch in the dataset, 
# because it's so rare to have zebra finch data </s>
bird_id_dir = STEINFATH_ET_AL_2021_RAW_DATA
assert bird_id_dir.exists(), f"Couldn't find bird_id_dir: {bird_id_dir}"

bird_id_dst = ZB_DATA_DST / f"Steinfath-et-al-{ZB_BIRD_ID}"
bird_id_dst.mkdir(exist_ok=True)


# bfsongrepo is split up by days; we throw away that split here
print(
    f"Copying audio and annotation files from day: {bird_id_dir.name}"
)
wav_paths = sorted(bird_id_dir.glob("*wav"))
print(
    f"Found {len(wav_paths)} audio files."
)
csv_paths = sorted(bird_id_dir.glob("*csv"))
print(
    f"Found {len(csv_paths)} annotation files."
)
wav_paths_filtered = []
csv_paths_filtered = []
for wav_path in wav_paths:
    csv_path_that_should_exist = wav_path.parent / (wav_path.stem + '_annotations.csv')
    if csv_path_that_should_exist in csv_paths:
        df = pd.read_csv(csv_path_that_should_exist)
        if len(df) > 0:
            # at least one annotation has no annotated segments in it; we skip
            wav_paths_filtered.append(wav_path)
            csv_paths_filtered.append(csv_path_that_should_exist)
print(
    "After filtering audio files by whether or not they have a valid annotation file, "
    f"there are {len(wav_paths_filtered)} audio files and {len(csv_paths_filtered)} annotation files."
)
if not DRY_RUN:
    pbar = tqdm(
        zip(wav_paths_filtered, csv_paths_filtered)
    )
    for wav_path, csv_path in pbar:
        shutil.copy(wav_path, bird_id_dst)
        # NOTE we rename to include the unit in the annotation file name
        simple_seq = crowsetta.formats.seq.SimpleSeq.from_file(
            csv_path,
            columns_map={
                'start_seconds': 'onset_s', 'stop_seconds': 'offset_s', 'name': 'label'
            },
        )
        csv_path_dst = bird_id_dst / (wav_path.name + '.syllable.csv')
        simple_seq.to_file(csv_path_dst)


# ## Human Speech

# In[37]:


SPEECH_DATA_DST = DATASET_ROOT / "Human-Speech"
SPEECH_DATA_DST.mkdir(exist_ok=True)


# In[38]:


SPEECH_RAW_DATA = RAW_DATA_ROOT / "Human-Speech"


# ### TIMIT Corpus + NLTK Sample
# 
# For corpus sample in NLTK:
# 1. Copy wav files, keep track of dialect regions + speaker IDs
# 2. Convert annotations to csv: phoneme + word
# 
# For full corpus:
# 1. Only copy data NOT in sample in NLTK--we will use this as the training data, and use the NLTK sample as the test data.
# 2. Copy wav files, throw away their train/test split but make sure we keep IDs + dialect regions.
# 3. Convert annotations to csv: phoneme + word

# In[39]:


TIMIT_NLTK_RAW = SPEECH_RAW_DATA / "TIMIT-corpus-sample-from-NLTK" / "timit"


# In[40]:


DATA_DIRS = [
    subdir for subdir in TIMIT_NLTK_RAW.iterdir()
    if subdir.is_dir() and subdir.name.startswith('dr')
]


# In[41]:


from collections import defaultdict
# keys will be dialect region, 
# values will be list of speaker IDs
NLTK_DR_SPKR_MAP = defaultdict(list)

for data_dir in DATA_DIRS:
    dir_name_upper = data_dir.name.upper()
    dialect_region, speaker_id = dir_name_upper.split('-')
    NLTK_DR_SPKR_MAP[dialect_region].append(speaker_id)

    wav_paths = sorted(data_dir.glob('*wav'))
    wrd_paths = sorted(data_dir.glob('*wrd'))
    phn_paths = sorted(data_dir.glob('*phn'))
    assert len(wav_paths) == len(wrd_paths) == len(phn_paths)

    dst = SPEECH_DATA_DST / f"TIMIT-NLTK-{dialect_region}-{speaker_id}"
    dst.mkdir(exist_ok=True)
    for wav_path, wrd_path, phn_path in zip(
        wav_paths, wrd_paths, phn_paths
    ):
        shutil.copy(wav_path, dst)

        phn_seq = crowsetta.formats.seq.Timit.from_file(phn_path).to_seq()
        phn_simpleseq = crowsetta.formats.seq.SimpleSeq(
            onsets_s=phn_seq.onsets_s,
            offsets_s=phn_seq.offsets_s,
            labels=phn_seq.labels,
            annot_path='dummy',
        )
        phn_csv_dst = dst / f"{wav_path.name}.phoneme.csv"
        phn_simpleseq.to_file(phn_csv_dst)

        wrd_seq = crowsetta.formats.seq.Timit.from_file(wrd_path).to_seq()
        wrd_simpleseq = crowsetta.formats.seq.SimpleSeq(
            onsets_s=wrd_seq.onsets_s,
            offsets_s=wrd_seq.offsets_s,
            labels=wrd_seq.labels,
            annot_path='dummy',
        )
        wrd_csv_dst = dst / f"{wav_path.name}.word.csv"
        wrd_simpleseq.to_file(wrd_csv_dst)


# In[42]:


DATA_WE_CANT_SHARE = DATA_DIR / "DATA-WE-CANT-SHARE"
DATA_WE_CANT_SHARE.mkdir(exist_ok=True)


# In[43]:


HUMAN_SPEECH_WE_CANT_SHARE = DATA_WE_CANT_SHARE / "Human-Speech"
HUMAN_SPEECH_WE_CANT_SHARE.mkdir(exist_ok=True)


# In[44]:


TIMIT_FULL_CORPUS_RAW = SPEECH_RAW_DATA / "TIMIT-corpus-full" / "data"


# In[45]:


TIMIT_DATA_DIRS = sorted(TIMIT_FULL_CORPUS_RAW.glob("T*/DR*/*/"))


# In[46]:


n_skipped = 0


for data_dir in TIMIT_DATA_DIRS:
    speaker_id = data_dir.name
    dialect_region = data_dir.parents[0].name
    if speaker_id in NLTK_DR_SPKR_MAP[dialect_region]:
        print(f"Skipping speaker {speaker_id} because they are in NLTK TIMIT corpus sample")
        continue
    
    wav_paths = sorted(data_dir.glob('*wav'))
    wrd_paths = sorted(data_dir.glob('*WRD'))
    phn_paths = sorted(data_dir.glob('*PHN'))
    assert len(wav_paths) == len(wrd_paths) == len(phn_paths)

    dst = HUMAN_SPEECH_WE_CANT_SHARE / f"TIMIT-full-corpus-{dialect_region}-{speaker_id}"
    dst.mkdir(exist_ok=True)
    for wav_path, wrd_path, phn_path in zip(
        wav_paths, wrd_paths, phn_paths
    ):
        wav_path_dst = dst / wav_path.stem.replace('.WAV', '.wav')
        shutil.copy(wav_path, wav_path_dst)

        phn_seq = crowsetta.formats.seq.Timit.from_file(phn_path).to_seq()
        phn_simpleseq = crowsetta.formats.seq.SimpleSeq(
            onsets_s=phn_seq.onsets_s,
            offsets_s=phn_seq.offsets_s,
            labels=phn_seq.labels,
            annot_path='dummy',
        )
        phn_csv_dst = dst / f"{wav_path_dst.name}.phoneme.csv"
        phn_simpleseq.to_file(phn_csv_dst)

        wrd_seq = crowsetta.formats.seq.Timit.from_file(wrd_path).to_seq()
        wrd_simpleseq = crowsetta.formats.seq.SimpleSeq(
            onsets_s=wrd_seq.onsets_s,
            offsets_s=wrd_seq.offsets_s,
            labels=wrd_seq.labels,
            annot_path='dummy',
        )
        wrd_csv_dst = dst / f"{wav_path_dst.name}.word.csv"
        wrd_simpleseq.to_file(wrd_csv_dst)

