import pathlib
import shutil

import biosoundsegbench.prep
import crowsetta
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import vak
import vocalpy as voc


def get_human_speech_train_data_dirs():
    """Get list of directories with training data for human speech.

    Each directory contains data from one speaker from one dialect region.
    The directories are sub-directories in :data:`biosoundsegbench.prep.constants.HUMAN_SPEECH_WE_CANT_SHARE`.
    The source data is from the full TIMIT corpus, minus the speakers that appear
    in the NLTK sample, that is used for testing.
    """
    return sorted(
        [id_dir
         for id_dir in biosoundsegbench.prep.constants.HUMAN_SPEECH_WE_CANT_SHARE.iterdir()
         if id_dir.is_dir()]
    )


# In[9]:


def get_human_speech_test_data_dirs():
    """Get list of directories with test data for human speech.

    Each directory contains data from one speaker from one dialect region.
    The directories are sub-directories in :data:`biosoundsegbench.prep.constants.SPEECH_DATA_DST`.
    The source data is from the NLTK Sample of the TIMIT corpus.
    """
    return sorted(
        [id_dir
         for id_dir in biosoundsegbench.prep.constants.SPEECH_DATA_DST.iterdir()
         if id_dir.is_dir()]
    )


def get_inputs_targets_human_speech(
    split: str = 'train', task: str = 'boundary classification', unit='phoneme', timebin_dur_str: str = '1'
):
    """Get inputs and targets for neural network models for human speech data"""
    if split == 'train':
        data_dirs = get_human_speech_train_data_dirs()
    elif split == 'test':
        data_dirs = get_human_speech_test_data_dirs()

    input_paths = []
    target_paths = defaultdict(list)
    for data_dir in data_dirs:
        input_paths.extend(
            get_frames_npz_paths
        )


def get_csv_paths(dir_path: pathlib.Path, unit='syllable'):
    return sorted(
        dir_path.glob(f"*{unit}.csv")
    )



def get_wav_paths(dir_path: pathlib.Path):
    return sorted(
        dir_path.glob(f"*.wav")
    )


def get_durs_from_wav_paths(wav_paths: list[pathlib.Path]):
    durs = []
    for wav_path in wav_paths:
        sound = voc.Audio.read(wav_path)
        durs.append(
            sound.data.shape[-1] / sound.samplerate
        )
    return durs


SCRIBE = crowsetta.Transcriber(format='simple-seq')


def get_labels_from_csv_paths(csv_paths: list[pathlib.Path]):
    return [
        SCRIBE.from_file(csv_path).to_seq().labels
        for csv_path in csv_paths
    ]


def get_split_wav_paths(
    data_dir: pathlib.Path,
    train_dur: int,
    test_dur: int,
    val_dur: int,
    labelset: set,
    unit: str = 'syllable'
):
    """Helper function that gets dataset splits: train, val, and test.

    Returns dict mapping the name of each split to a list of wav paths.
    These wav paths can then be matched with inputs and targets as required
    for training different models.

    Uses :func:`vak.prep.split.split.train_tset_dur_split_inds`."""
    wav_paths = get_wav_paths(data_dir)
    csv_paths = get_csv_paths(data_dir)
    durs = get_durs_from_wav_paths(wav_paths)
    labels = get_labels_from_csv_paths(csv_paths)

    # TODO: here we get total training pool of 10k seconds
    # later we will make "folds" of training data, say 3 splits
    train_inds, test_inds, val_inds = vak.prep.split.split.train_test_dur_split_inds(
        durs,
        labels,
        labelset,
        train_dur,
        test_dur,
        val_dur
    )

    return {
        'train': [wav_paths[train_ind] for train_ind in train_inds],
        'val': [wav_paths[val_ind] for val_ind in val_inds],
        'test': [wav_paths[test_ind] for test_ind in test_inds],
    }


TARGET_COLUMNS = [
    "multi_frame_labels_path",
    "binary_frame_labels_path",
    "boundary_onehot_path",
]


def split_wav_paths_to_df(
    data_dir: pathlib.Path, split_wav_paths:dict, unit: str = 'syllable',
    timebin_dur_str: str = "1", annot_format="simple-seq",
    target_columns=TARGET_COLUMNS,
):
    if not all(
        [target_column in TARGET_COLUMNS for target_column in target_columns]
    ):
        invalid = set(target_columns) - set(TARGET_COLUMNS)
        raise ValueError(
            f"Invalid target column(s): {invalid}.\n"
            f"Valid columns are: {TARGET_COLUMNS}"
        )

    timebin_dur = int(timebin_dur_str) * 1e-3
    records = []
    for split, audio_paths in split_wav_paths.items():
        for audio_path in audio_paths:
            sound = voc.Audio.read(audio_path)
            duration = sound.data.shape[-1] / sound.samplerate

            parent = audio_path.parent
            name = audio_path.name

            annot_path =  parent / (name + f".{unit}.csv")
            if not annot_path.exists():
                raise FileNotFoundError(
                    f"Did not find `annot_path` for `audio_path`.\n"
                    f"`audio_path`: {audio_path}"
                    f"`annot_path`: {annot_path}"
                )

            frames_path = parent / (name + f".timebin-{timebin_dur_str}-ms.frames.npz")
            if not frames_path.exists():
                raise FileNotFoundError(
                    f"Did not find `frames_path` for `audio_path`.\n"
                    f"`audio_path`: {audio_path}"
                    f"`frames_path`: {frames_path}"
                )

            record = {
                'audio_path': str(
                    audio_path.relative_to(biosoundsegbench.prep.constants.DATASET_ROOT)
                ),
                'annot_path': str(
                    annot_path.relative_to(biosoundsegbench.prep.constants.DATASET_ROOT)
                ),
                'frames_path': str(
                    frames_path.relative_to(biosoundsegbench.prep.constants.DATASET_ROOT)
                ),
                'timebin_dur': timebin_dur,
                'duration': duration,
                'annot_format': annot_format,
                'split': split,
            }

            if "multi_frame_labels_path" in target_columns:
                multi_frame_labels_path = parent / (
                    name + f".timebin-{timebin_dur_str}-ms.{unit}.multi-frame-labels.npy"
                )
                if not multi_frame_labels_path.exists():
                    raise FileNotFoundError(
                        f"Did not find `multi_frame_labels_path` for `audio_path`.\n"
                        f"`audio_path`: {audio_path}"
                        f"`multi_frame_labels_path`: {annot_path}"
                    )
                record['multi_frame_labels_path'] = str(
                    multi_frame_labels_path.relative_to(biosoundsegbench.prep.constants.DATASET_ROOT)
                )

            if "binary_frame_labels_path" in target_columns:
                binary_frame_labels_path = parent / (
                    name + f".timebin-{timebin_dur_str}-ms.{unit}.binary-frame-labels.npy"
                )
                if not binary_frame_labels_path.exists():
                    raise FileNotFoundError(
                        f"Did not find `binary_frame_labels_path` for `audio_path`.\n"
                        f"`audio_path`: {audio_path}"
                        f"`binary_frame_labels_path`: {binary_frame_labels_path}"
                    )
                record['binary_frame_labels_path'] = str(
                    binary_frame_labels_path.relative_to(biosoundsegbench.prep.constants.DATASET_ROOT)
                )

            if "boundary_onehot_path" in target_columns:
                boundary_onehot_path = parent / (
                    name + f".timebin-{timebin_dur_str}-ms.{unit}.boundary-onehot.npy"
                )
                if not boundary_onehot_path.exists():
                    raise FileNotFoundError(
                        f"Did not find `boundary_onehot_path` for `audio_path`.\n"
                        f"`audio_path`: {audio_path}"
                        f"`boundary_onehot_path`: {boundary_onehot_path}"
                    )
                record['boundary_onehot_path'] = str(
                    boundary_onehot_path.relative_to(biosoundsegbench.prep.constants.DATASET_ROOT)
                )
            records.append(record)
    df = pd.DataFrame.from_records(records)
    return df


BIOSOUND_GROUP_DIR_MAP = {
    'Bengalese-Finch-Song': biosoundsegbench.prep.constants.BF_DATA_DST,
    'Canary-Song':

}

def get_df_per_id(
    biosound_group: str,
    unit: str,
    train_dur: int,
    val_dur: int,
    test_dur: int,
):
    group_dir = BIOSOUND_GROUP_DIR_MAP[biosound_group]
    bf_id_dirs = [
        subdir
        for subdir in group_dir.iterdir()
        if subdir.is_dir()
    ]

    labelsets = biosoundsegbench.prep.labels.get_labelsets()[biosound_group][unit]

    id_dfs = {}
    for bf_id_dir in bf_id_dirs:
        bf_id = bf_id_dir.name.split('-')[-1]
        print(
            f"Finding split for: {bf_id}"
        )
        labelset = set(
            bf_labelsets[bf_id]
        )
        split_wav_paths = get_split_wav_paths(
            bf_id_dir,
            train_dur=900,
            test_dur=400,
            val_dur=80,
            labelset=labelset,
            unit='syllable'
        )
        df = split_wav_paths_to_df(
            bf_id_dir,
            split_wav_paths
        )
        # csv_filename = f"Bengalese-Finch-Song.id-{bf_id}.timebin-{timebin_dur_str}-ms.{unit}.splits.csv"
        # df.to_csv(
        #     biosoundsegbench.prep.constants.DATASET_ROOT / csv_filename
        #     index=False,
        # )
        bf_id_dfs[bf_id] = df