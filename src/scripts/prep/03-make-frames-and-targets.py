#!/usr/bin/env python
# coding: utf-8

# # Make inputs and targets
# 
# Uses raw data--audio and annotations--to make inputs and targets for neural network models.

# In[1]:


import pathlib
import shutil

import crowsetta
import dask.delayed
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import vak
import vocalpy as voc
from vak.config.spect_params import SpectParamsConfig


# In[2]:


cd ../../..


# In[3]:


DATA_DIR = pathlib.Path('./data')
assert DATA_DIR.exists(), "couldn't find DATA_DIR"


# In[4]:


DATASET_ROOT = DATA_DIR / "BioSoundSegBench"
assert DATASET_ROOT.exists(), "couldn't find DATASET_ROOT"


# In[5]:


import json


with open("./data/raw/labelmaps.json", "r") as fp:
    SPECIES_ID_LABELSETS_MAP = json.load(fp)


SPECIES_ID_LABELMAP_MAP = {}
for species in SPECIES_ID_LABELSETS_MAP.keys():
    id_labelset_map = SPECIES_ID_LABELSETS_MAP[species]
    id_labelmap_map = {
        id: vak.common.labels.to_map(
            # we need to convert from list back to set when loading from json
            set(labelset)
        )
        for id, labelset in id_labelset_map.items()
    }
    SPECIES_ID_LABELMAP_MAP[species] = id_labelmap_map


# ## Bengalese finch song
# 
# 1. Make standard spectrograms
# 2. Make spectrograms same way but with 1 ms time bins
# 3. For both sets of spectrograms, make: multi-class frame label, binary frame label, boundary detection

# In[6]:


BF_DATA = DATASET_ROOT / "Bengalese-Finch-Song"
assert BF_DATA.exists(), f"Directory doesn't exist: {BF_DATA}"


# In[7]:


from dataclasses import dataclass


@dataclass
class SpectParams:
    """Parameters for spectrogramming.

    Attributes
    ----------
    fft_size : int
        size of window for Fast Fourier transform, number of time bins. 
    step_size : int
        step size for Fast Fourier transform.
    freq_cutoffs : list
        of two elements, lower and higher frequencies. Used to bandpass filter audio
        (using a Butter filter) before generating spectrogram.
        Default is None, in which case no bandpass filtering is applied.
    transform_type : str
        one of {'log_spect', 'log_spect_plus_one'}.
        'log_spect' transforms the spectrogram to log(spectrogram), and
        'log_spect_plus_one' does the same thing but adds one to each element.
        Default is None. If None, no transform is applied.
    timebin_dur : int
        Expected duration of timebins in spectrogram, in milliseconds.
        Used to validate output of spectrogram function,
        and in creating filename for spectrogram (as a form of metadata).
    thresh: float, optional
        threshold minimum power for log spectrogram.
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    freqbins_key : str
        key for accessing vector of frequency bins in files. Default is 'f'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    audio_path_key : str
        key for accessing path to source audio file for spectogram in files.
        Default is 'audio_path'.
    """
    fft_size: int
    step_size: int
    transform_type: str
    timebin_dur: int
    freq_cutoffs: list[int] | None = None
    thresh: float | None = None
    spect_key: str = 's'
    freqbins_key: str = 'f'
    timebins_key: str = 't'
    audio_path_key: str = 'audio_path'


# In[8]:


BF_SPECT_PARAMS = [
    SpectParams(
        fft_size=512,
        step_size=64,
        freq_cutoffs=[500, 10000],
        transform_type='log_spect',
        timebin_dur=2,
    ),
    # time bin size = 0.001 s
    SpectParams(
        fft_size=512,
        step_size=32,
        freq_cutoffs=[500, 10000],
        transform_type='log_spect',
        timebin_dur=1,
    ),
]


# In[9]:


def frame_labels_to_boundary_onehot(frame_labels):
    """Converts vector of frame labels to one-hot vector
    indicating "boundary" (1) or "not a boundary" (0)."""
    # a boundary occurs in frame labels
    # wherever the first-order difference is not 0.
    boundary_onehot = (np.diff(frame_labels, axis=0) != 0).astype(int)
    # The actual change occurs at :math:`i`=np.diff(frame_labels) + 1,
    # but we shift everything to the right by 1 when we add a first index 
    # indicating a boundary.
    # This first index we add is the onset of the "first" segment 
    # -- typically will be a segment belonging to the background class
    boundary_onehot = np.insert(boundary_onehot, 0, 1)
    return boundary_onehot


# In[10]:


def frame_labels_multi_to_binary(
    frame_labels, labelmap, bg_class_name: str = 'unlabeled'
):
    """Converts vector of frame labels with multiple classes
    to a vector for binary classification."""
    bg_class_int = labelmap[bg_class_name]
    return (frame_labels != bg_class_int).astype(int)


# In[11]:


def get_frames_filename(audio_path, timebin_dur):
    """Get name for frames file, where frames are the input to the network.

    Helper function we use to standardize the name"""
    return audio_path.name + f".timebin-{timebin_dur}-ms.frames.npz"


# In[12]:


def get_multi_frame_labels_filename(audio_path, timebin_dur, unit):
    """Get name for multiclass frame labels file, 
    the target outputs for the network.

    Helper function we use to standardize the name"""
    return audio_path.name + f".timebin-{timebin_dur}-ms.{unit}.multi-frame-labels.npy"


# In[13]:


def get_binary_frame_labels_filename(audio_path, timebin_dur, unit):
    """Get name for binary classification frame labels file, 
    the target outputs for the network.

    Helper function we use to standardize the name"""
    return audio_path.name + f".timebin-{timebin_dur}-ms.{unit}.binary-frame-labels.npy"


# In[35]:


def get_boundary_onehot_filename(audio_path, timebin_dur, unit):
    """Get name for boundary detection onehot encoding file, 
    the target outputs for the network.

    Helper function we use to standardize the name"""
    return audio_path.name + f".timebin-{timebin_dur}-ms.{unit}.boundary-onehot.npy"


# In[64]:


import math

from vak.prep.spectrogram_dataset.spect import spectrogram


SCRIBE = crowsetta.Transcriber(format='simple-seq')


def audio_and_annot_to_inputs_and_targets(
    audio_path: pathlib.Path,
    annot_path: pathlib.Path,
    spect_params: SpectParams,
    dst: pathlib.Path,
    labelmap: dict,
    unit: str,
):
    """Generate frames (spectrogram) and frame label / boundary detection vectors.

    This is a helper function used to parallelize, by calling it
    with `dask.delayed`.
    It is used with Bengalese finch song, canary song, and zebra finch song.
    """
    sound = voc.Audio.read(audio_path)

    s, f, t = spectrogram(
        sound.data,
        sound.samplerate,
        spect_params.fft_size,
        spect_params.step_size,
        spect_params.thresh,
        spect_params.transform_type,
        spect_params.freq_cutoffs,
    )
    timebin_dur = np.diff(t).mean()
    if not math.isclose(
        timebin_dur, 
        spect_params.timebin_dur * 1e-3,
        abs_tol=0.001,
    ):
        raise ValueError(
            f"Expected spectrogram with timebis of duration {spect_params.timebin_dur * 1e-3} "
            f"but got duration {timebin_dur} for audio path: {audio_path}"
        )
    spect_dict = {
        spect_params.spect_key: s,
        spect_params.freqbins_key: f,
        spect_params.timebins_key: t,
    }

    frames_filename = get_frames_filename(
        audio_path, spect_params.timebin_dur
    )
    frames_path = dst / frames_filename
    np.savez(frames_path, **spect_dict)

    annot = SCRIBE.from_file(annot_path)
    lbls_int = [labelmap[lbl] for lbl in annot.labels]
    frame_labels_multi = vak.transforms.frame_labels.from_segments(
        lbls_int,
        annot.onsets_s,
        annot.offsets_s,
        t,
        unlabeled_label=labelmap["unlabeled"],
    )
    frame_labels_multi_filename = get_multi_frame_labels_filename(
        audio_path, spect_params.timebin_dur, unit
    )
    frame_labels_multi_path = dst / frame_labels_multi_filename
    np.save(frame_labels_multi_path, frame_labels_multi)

    frame_labels_binary = frame_labels_multi_to_binary(frame_labels_multi, labelmap)
    frame_labels_binary_filename = get_binary_frame_labels_filename(
        audio_path, spect_params.timebin_dur, unit
    )
    frame_labels_binary_path = dst / frame_labels_binary_filename
    np.save(frame_labels_binary_path, frame_labels_binary)
    
    boundary_onehot = frame_labels_to_boundary_onehot(frame_labels_multi)
    boundary_onehot_filename = get_boundary_onehot_filename(
        audio_path, spect_params.timebin_dur, unit
    )
    boundary_onehot_path = dst / boundary_onehot_filename
    np.save(boundary_onehot_path, boundary_onehot)


# In[17]:


def make_inputs_targets_bf(id_dir, labelmap, unit='syllable'):
    wav_paths = voc.paths.from_dir(id_dir, '.wav')
    csv_paths = voc.paths.from_dir(id_dir, f'.{unit}.csv')

    for spect_params in BF_SPECT_PARAMS:
        print(
            f"Making inputs and targets with `spect_params`: {spect_params}"
        )
        todo = []
        for wav_path, csv_path in zip(wav_paths, csv_paths):
            todo.append(
                dask.delayed(audio_and_annot_to_inputs_and_targets)(
                    audio_path=wav_path,
                    annot_path=csv_path,
                    spect_params=spect_params,
                    dst=id_dir,
                    labelmap=labelmap,
                    unit=unit,
                )
            )
        with ProgressBar():
            dask.compute(*todo)


# In[18]:


ID_DIRS = [dir_ for dir_ in sorted(BF_DATA.glob("*/"))
           if dir_.is_dir()]


# In[18]:


for id_dir in ID_DIRS:
    print(
        f"Making neural network inputs and targets for: {id_dir.name}"
    )
    id = id_dir.name.split('-')[-1]
    labelmap = SPECIES_ID_LABELMAP_MAP['Bengalese-Finch-Song'][id]
    make_inputs_targets_bf(id_dir, labelmap)
    break


# In[55]:


# import matplotlib.pyplot as plt

# spect_dict, frame_multi, frame_binary, boundary_onehot = out

# fig, ax_arr = plt.subplots(4, 1, sharex=True)
# t = spect_dict['t']
# ax_arr[0].pcolormesh(
#     t, spect_dict['f'], spect_dict['s']
# )
# ax_arr[1].plot(t, frame_multi)
# ax_arr[2].plot(t, frame_binary)
# ax_arr[3].vlines(t[np.nonzero(boundary_onehot)[0]], 0., 1.)

# for ax in ax_arr:
#     ax.set_xlim([17, 22])


# In[ ]:


for id_dir in ID_DIRS:
    make_inputs_targets_bf(id_dir, spect_params_list)


# ## Canary song
# 1. Copy spectrograms with 2.7 ms time bins from data set
# 2. Make spectrograms with 1 ms time bins
# 3. For both sets of spectrograms, make: multi-class frame label, binary frame label, boundary detection

# In[19]:


CANARY_DATA = DATASET_ROOT / "Canary-Song"
assert CANARY_DATA.exists(), f"Can't find: {CANARY_DATA}"


# For one set of spectrogram files, we copy them from the raw data.
# 
# We do this *after* we copy raw audio and convert annotations, then QC the annotations and remove any (audio, annotation) pairs where the annotations are invalid (e.g., because of a first onset less than zero).
# 
# That's because we want to make sure the audio file exists before we bother to copy the spectrogram. We also need to do some clean up of the vectors, and save the labels.

# In[33]:


import math

from vak.prep.spectrogram_dataset.spect import spectrogram


SCRIBE = crowsetta.Transcriber(format='simple-seq')


def spect_npz_and_annot_to_inputs_and_targets_canary(
    spect_npz_path: pathlib.Path,
    annot_path: pathlib.Path,
    dst: pathlib.Path,
    labelmap: dict,
    unit: str,
    timebin_dur: float = 2.7,  # milliseconds
):
    """Generate frames (spectrogram) and frame label / boundary detection vectors.
    
    This helper function is used just with canary song,
    for the case where we are converting / renaming existing spectrogram files
    """
    spect_dict = np.load(spect_npz_path)

    timebin_dur_from_t = np.diff(spect_dict["t"]).mean() * 1000
    if not math.isclose(timebin_dur_from_t, timebin_dur, abs_tol=0.002):
        raise ValueError(
            f"Expected spectrogram with timebis of duration {timebin_dur} "
            f"but got duration {timebin_dur_from_t * 1000} for audio path: {audio_path}. "
            f"Difference ({abs((timebin_dur_from_t * 1000)) - 2.7})) greater than tolerance of 0.002"
        )
    s = spect_dict["s"]
    f = np.squeeze(spect_dict["f"])
    t = np.squeeze(spect_dict["t"])
    spect_dict_out = {
        "s": s,
        "f": f,
        "t": t,
    }

    # we fake audio path
    # we already confirmed in calling function that it exists
    audio_path = spect_npz_path.parent / spect_npz_path.name.replace('.npz', '')
    frames_filename = get_frames_filename(
        audio_path, timebin_dur
    )
    frames_path = dst / frames_filename
    np.savez(frames_path, **spect_dict_out)

    annot = SCRIBE.from_file(annot_path)
    lbls_int = [labelmap[lbl] for lbl in annot.labels]
    frame_labels_multi = vak.transforms.frame_labels.from_segments(
        lbls_int,
        annot.onsets_s,
        annot.offsets_s,
        t,
        unlabeled_label=labelmap["unlabeled"],
    )
    frame_labels_multi_filename = get_multi_frame_labels_filename(
        audio_path, timebin_dur, unit
    )
    frame_labels_multi_path = dst / frame_labels_multi_filename
    np.save(frame_labels_multi_path, frame_labels_multi)

    frame_labels_binary = frame_labels_multi_to_binary(frame_labels_multi, labelmap)
    frame_labels_binary_filename = get_binary_frame_labels_filename(
        audio_path, timebin_dur, unit
    )
    frame_labels_binary_path = dst / frame_labels_binary_filename
    np.save(frame_labels_binary_path, frame_labels_binary)
    
    boundary_onehot = frame_labels_to_boundary_onehot(frame_labels_multi)
    boundary_onehot_filename = get_boundary_onehot_filename(
        audio_path, timebin_dur, unit
    )
    boundary_onehot_path = dst / boundary_onehot_filename
    np.save(boundary_onehot_path, boundary_onehot)


# In[59]:


CANARY_SPECT_PARAMS = [
    # time bin size = 0.001 s
    SpectParams(
        fft_size=512,
        step_size=45,
        transform_type='log_spect',
        timebin_dur=1,
    ),
]


# In[60]:


def make_inputs_targets_canary(raw_data_dir, id_dir, labelmap, unit='syllable'):
    spect_npz_paths = voc.paths.from_dir(raw_data_dir, '.npz')
    wav_paths = voc.paths.from_dir(id_dir, '.wav')
    csv_paths = voc.paths.from_dir(id_dir, f'.{unit}.csv')

    print(
        "Making inputs and targets from spectrograms in original dataset."
    )

    hashmap = {
        wav_path.name: (wav_path, csv_path)
        for wav_path, csv_path in zip(wav_paths, csv_paths)
    }
    todo = []
    for spect_npz_path in spect_npz_paths:
        wav_filename_from_spect_npz_path = spect_npz_path.name.replace('.npz', '')
        if wav_filename_from_spect_npz_path in hashmap:
            _, csv_path = hashmap[wav_filename_from_spect_npz_path]
            todo.append(
                dask.delayed(
                    spect_npz_and_annot_to_inputs_and_targets_canary
                )(
                    spect_npz_path,
                    csv_path,
                    dst=id_dir,
                    labelmap=labelmap,
                    unit=unit,
                )
            )
    with ProgressBar():
        dask.compute(*todo)

    for spect_params in CANARY_SPECT_PARAMS:
        print(
            f"Making inputs and targets with `spect_params`: {spect_params}"
        )
        todo = []
        # TODO: parallelize here and make frame labels / boundary vectors
        # at the same time so we know we have the correct timebin duration
        # for the frames
        for wav_path, csv_path in zip(wav_paths, csv_paths):
            todo.append(
                dask.delayed(audio_and_annot_to_inputs_and_targets)(
                    audio_path=wav_path,
                    annot_path=csv_path,
                    spect_params=spect_params,
                    dst=id_dir,
                    labelmap=labelmap,
                    unit=unit
                )
            )
        with ProgressBar():
            dask.compute(*todo)


# In[61]:


RAW_DATA_ROOT = DATA_DIR / "raw"
CANARY_RAW_DATA = RAW_DATA_ROOT / "Canary-Song"
TWEETYNET_RAW_DATA = CANARY_RAW_DATA / "tweetynet-canary"


# In[62]:


CANARY_IDS = (
    'llb3',
    'llb11',
    'llb16',
)


# In[65]:


for canary_id in CANARY_IDS:
    print(
        f"Making frames and targets for ID: {canary_id}"
    )
    raw_data_dir = TWEETYNET_RAW_DATA / f"{canary_id}_data_matrices" / f"{canary_id}_data_matrices"
    assert raw_data_dir.exists(), f"Couldn't find raw_data_dir: {raw_data_dir}"
    id_dir = CANARY_DATA / f"tweetynet-{canary_id}"
    labelmap = SPECIES_ID_LABELMAP_MAP['Canary-Song'][canary_id]
    make_inputs_targets_canary(raw_data_dir, id_dir, labelmap)


# ## Mouse calls
# 1. Make spectrograms same way as Nick did
# 2. Make spectrograms with 1 ms time bins

# In[ ]:





# In[ ]:





# In[ ]:





# ## Zebra finch song
# 1. Make spectrograms with time bins from DAS?
# 2. Make spectrograms with 1 ms time bins

# In[ ]:





# In[ ]:





# In[ ]:





# ## Human speech
# 1. Make standard MFCC features spectrograms, 10 ms time bins
# 2. Make standard MFCC features spectrograms, 1 ms time bins

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




