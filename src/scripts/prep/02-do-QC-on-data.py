#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import shutil

import crowsetta
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import vocalpy as voc


# In[2]:


cd ../../..


# In[3]:


DATA_DIR = pathlib.Path('./data')
assert DATA_DIR.exists(), "couldn't find DATA_DIR"


# In[4]:


DATASET_ROOT = DATA_DIR / "BioSoundSegBench"
assert DATASET_ROOT.exists(), "couldn't find DATASET_ROOT"


# In[5]:


DATA_DIRS = sorted(DATASET_ROOT.glob(
    "*/*/"
))


# In[6]:


DRY_RUN = False


# ## Validate annotations
# 
# For each annotation file we validate that
# 1. The first onset is not less than zero
# 2. No onsets are less than zero
# 3. No offsets are less than zero
# 4. The onsets and offsets can be concatenated and the result will be considered a valid boundaries vector by `vocalpy`

# In[7]:


from collections import defaultdict

SCRIBE = crowsetta.Transcriber(format='simple-seq')


def qc_annot(data_dir, csv_ext=".syllable.csv"):
    first_onset_lt_zero = []
    any_onset_lt_zero = []
    any_offset_lt_zero = []
    invalid_starts_stops = []

    wav_paths = voc.paths.from_dir(data_dir, '.wav')
    csv_paths = voc.paths.from_dir(data_dir, csv_ext)
    if not len(wav_paths) == len(csv_paths):
        raise ValueError(
            f"len(wav_paths)={len(wav_paths)} != len(csv_paths)={len(csv_paths)}"
        )
    id_ = data_dir.name.split('-')[-1]
    for wav_path, csv_path in zip(wav_paths, csv_paths):
        simpleseq = SCRIBE.from_file(csv_path)
        if simpleseq.onsets_s[0] < 0.:
            # print(
            #     f"File has first onset less than 0: {csv_path.name}"
            # )
            first_onset_lt_zero.append(
                (wav_path, csv_path)
            )
            # `continue` so we don't add same (wav, csv) tuple twice
            # and cause an error downstream
            continue
        elif np.any(simpleseq.onsets_s[1:]) < 0.:
            # print(
            #     f"File has onset (other than first) less than 0: {csv_path.name}"
            # )
            any_onset_lt_zero.append((wav_path, csv_path))
            continue
        elif np.any(simpleseq.offsets_s) < 0.:
            # print(
            #     f"File has offset less than 0: {csv_path.name}"
            # )
            any_offset_lt_zero.append((wav_path, csv_path))
            continue
        else:
            try:
                voc.metrics.segmentation.ir.concat_starts_and_stops(
                    simpleseq.onsets_s, simpleseq.offsets_s
                )
            except:
                print(
                    f"caused error when concatenating starts and stops: {csv_path.name}"
                )
                invalid_starts_stops.append((wav_path, csv_path))

    return first_onset_lt_zero, any_onset_lt_zero, any_offset_lt_zero, invalid_starts_stops


# In[8]:


SPECIES_CSV_EXT_MAP = {
    'Bengalese-Finch': ".syllable.csv",
    'Canary': ".syllable.csv",
    'Zebra-Finch': ".syllable.csv",
    'Mouse': ".call.csv",
    # we assume the human speech dataset has valid onset + offset times
}


for species, csv_ext in SPECIES_CSV_EXT_MAP.items():
    print(
        f"QCing annotation for species '{species}' and csv extension '{csv_ext}'")
    data_dirs = [
        data_dir for data_dir in DATA_DIRS
        if species in data_dir.parents[0].name
    ]
    for data_dir in data_dirs:
        print(f"data dir name: {data_dir.name}")
        (first_onset_lt_zero,
         any_onset_lt_zero,
         any_offset_lt_zero,
         invalid_starts_stops
        ) = qc_annot(data_dir, csv_ext)
    
        print(
            f"\tNum. w/first onset less than zero: {len(first_onset_lt_zero)}\n"
            f"\tNum. w/any onset less than zero: {len(any_onset_lt_zero)}\n"
            f"\tNum. w/any offset less than zero: {len(any_offset_lt_zero)}\n"
            f"\tNum. w/invalid starts + stops: {len(invalid_starts_stops)}\n"
        )
        if not DRY_RUN:
            for wav_csv_tup_list, dir_name in zip(
                (first_onset_lt_zero,
                 any_onset_lt_zero,
                 any_offset_lt_zero,
                 invalid_starts_stops),
                ('first_onset_lt_zero',
                 'any_onset_lt_zero',
                 'any_offset_lt_zero',
                 'invalid_starts_stops'),
            ):
                if len(wav_csv_tup_list) > 0:
                    remove_dst = data_dir / dir_name
                    remove_dst.mkdir(exist_ok=True)
                    for wav_path, csv_path in wav_csv_tup_list:
                        shutil.move(wav_path, remove_dst)
                        shutil.move(csv_path, remove_dst)                


# ## Remove any sounds that have labels not in labelset
# 
# For the bird datasets in particular we need to remove cases where a sound is labeled with a label not in the set of labels we use during training, since we will always get that one wrong.

# In[9]:


import vak


SPECIES_ID_LABELSETS_MAP = {
    'Bengalese-Finch-Song': {
        'bl26lb16': "iabcdef",
        'gr41rd51': "iabcdefgjkm",
        'gy6or6': "iabcdefghjk",
        'or60yw70': "iabcdefg",
        'Bird0': "0123456789",
        'Bird4': "01234567",
        'Bird7': "0123456",
        'Bird9': "012345", 
    },
    'Canary-Song': {
        'llb3': "range: 1-20",
        'llb11': "range: 1-30",
        'llb16': "range: 1-30",
    },
}

for species in SPECIES_ID_LABELSETS_MAP.keys():
    id_labelset_map = SPECIES_ID_LABELSETS_MAP[species]
    id_labelset_map = {
        id: vak.common.converters.labelset_to_set(labelset)
        for id, labelset in id_labelset_map.items()
    }
    SPECIES_ID_LABELSETS_MAP[species] = id_labelset_map


# In[ ]:


for species, id_labelset_map in SPECIES_ID_LABELSETS_MAP.items():
    species_root = DATASET_ROOT / species
    species_subdirs = [
        subdir for subdir in species_root.iterdir()
        if subdir.is_dir()
    ]
    for id, labelset in id_labelset_map.items():
        id_dir = [
            id_dir for id_dir in species_subdirs
            if id_dir.name.endswith(id)
        ]
        assert len(id_dir) == 1
        id_dir = id_dir[0]
        wav_paths = voc.paths.from_dir(id_dir, '.wav')
        csv_paths = voc.paths.from_dir(id_dir, '.syllable.csv')
        if not len(wav_paths) == len(csv_paths):
            raise ValueError(
                f"len(wav_paths)={len(wav_paths)} != len(csv_paths)={len(csv_paths)}"
            )
        labels_not_in_labelset = []
        for wav_path, csv_path in zip(wav_paths, csv_paths):
            simpleseq = crowsetta.formats.seq.SimpleSeq.from_file(csv_path)
            if not all(
                [lbl in labelset for lbl in simpleseq.labels]
            ):
                labels_not_in_labelset.append(
                    (wav_path, csv_path)
                )
        print(
            f"Found {len(labels_not_in_labelset)} annotations with labels "
            f"not in labelset for ID: {id}"
        )
        if not DRY_RUN:
            not_in_labelset_dst = id_dir / 'labels-not-in-labelset'
            not_in_labelset_dst.mkdir(exist_ok=True)
            for wav_path, csv_path in labels_not_in_labelset:
                shutil.move(wav_path, not_in_labelset_dst)
                shutil.move(csv_path, not_in_labelset_dst)

