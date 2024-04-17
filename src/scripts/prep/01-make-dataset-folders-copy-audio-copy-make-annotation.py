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

# ### TweetyNet dataset

# In[19]:


CANARY_DATA_DST = DATASET_ROOT / "Canary-Song"
CANARY_DATA_DST.mkdir(exist_ok=True)


# In[20]:


CANARY_RAW_DATA = RAW_DATA_ROOT / "Canary-Song"


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

# In[25]:


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

# In[26]:


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

# In[18]:


MOUSE_RAW_DATA = RAW_DATA_ROOT / "Mouse-Pup-Calls"


# In[19]:


MOUSE_DATA_DST = DATASET_ROOT / "Mouse-Pup-Calls"
MOUSE_DATA_DST.mkdir(exist_ok=True)


# In[20]:


JOURJINE_ET_AL_2023_DATA = MOUSE_RAW_DATA / "Jourjine-et-al-2023"


# In[21]:


ANNOT_DIR = JOURJINE_ET_AL_2023_DATA / "processed_data" / "supplemental_figure_5"


# In[22]:


segs_csv_path = ANNOT_DIR / "all_development_vocs_with_start_stop_times.csv"
segs_df = pd.read_csv(segs_csv_path)


# For this dataset we start with the annotations since there are less of those than audio files, and since we need to build the annotations on the fly.
# 
# We treat this as a binary classificaiton task: "vocalization" or "no vocalization". So we just label all the detected segments as "v" (for "vocalization")

# In[23]:


len(segs_df)


# In[27]:


DRY_RUN = False


# In[28]:


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
                csv_path = species_id_dst / f"{source_file}.calls.csv"
                simple_seq.to_file(csv_path)


# In[6]:


import vocalpy as voc

root = "data/BioSoundSegBench/Mouse-Pup-Calls/jourjine-et-al-2023-BW"
wav_paths = voc.paths.from_dir(root, "wav")
csv_paths = voc.paths.from_dir(root, "csv")


# In[44]:


def plot_pup(ind=0, dur=5.0, tlim=[2., 4.]):
    sound = voc.Audio.read(wav_paths[ind])
    sound = voc.Audio(
        data=sound.data[:int(dur * sound.samplerate)],
        samplerate=sound.samplerate,
    )
    spect = voc.spectrogram(sound)
    annot = voc.Annotation.read(csv_paths[ind], format='simple-seq')
    voc.plot.annotated_spectrogram(spect, annot, tlim=tlim)


# In[14]:


n_plot = 0

for ind, csv_path in enumerate(csv_paths):
    if n_plot > 3:
        break
    annot = voc.Annotation.read(csv_path, format='simple-seq')
    if 'USV' in annot.data.seq.labels:
        usv_ind = np.nonzero(annot.data.seq.labels == 'USV')[0][0]
        onset = annot.data.seq.onsets_s[usv_ind]
        dur = onset + 2
        n_plot += 1
        sound = voc.Audio.read(wav_paths[ind])
        start_ind = int((onset - 2.0) * sound.samplerate)
        stop_ind = int((onset + 2.0) * sound.samplerate)
        sound = voc.Audio(
            data=sound.data[start_ind:stop_ind],
            samplerate=sound.samplerate,
        )
        spect = voc.spectrogram(sound)
        annot = voc.Annotation(
            data=crowsetta.Annotation(
                seq=crowsetta.Sequence.from_keyword(
                    onsets_s=annot.data.seq.onsets_s - (onset - 2.0),
                    offsets_s=annot.data.seq.offsets_s - (onset -2.0),
                    labels=annot.data.seq.labels,
                ),
                annot_path='dummy',
            ),
            path='dummy',
        )
        voc.plot.annotated_spectrogram(spect, annot, tlim=[0, 4.0])


# ## Zebra finch song

# ### Steinfath et al dataset, from AVA dataset

# In[ ]:





# ## Human Speech

# ### TIMIT

# ### TIMIT Corpus + NLTK Subset
# 
# 1. Copy wav files, throw away their train/test split but make sure we keep IDs + dialect regions
# 2. Convert annotations to csv: phoneme + word

# ### TIMIT Corpus NLTK Subset
