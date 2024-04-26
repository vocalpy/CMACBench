"""Functions to make dataset splits--training, validation, and test--and save as csv files in dataset root."""
import logging
import pathlib
from dataclasses import dataclass

import crowsetta
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import vak
import vocalpy as voc

from . import constants, labels


logger = logging.getLogger(__name__)


def get_human_speech_train_data_dirs():
    """Get list of directories with training data for human speech.

    Each directory contains data from one speaker from one dialect region.
    The directories are sub-directories in :data:`biosoundsegbench.prep.constants.HUMAN_SPEECH_WE_CANT_SHARE`.
    The source data is from the full TIMIT corpus, minus the speakers that appear
    in the NLTK sample, that is used for testing.
    """
    return sorted(
        [id_dir
         for id_dir in constants.HUMAN_SPEECH_WE_CANT_SHARE.iterdir()
         if id_dir.is_dir()]
    )


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


def get_durs_from_wav_paths(wav_paths: list[pathlib.Path]):
    durs = []
    for wav_path in wav_paths:
        sound = voc.Audio.read(wav_path)
        durs.append(
            sound.data.shape[-1] / sound.samplerate
        )
    return durs


# In[7]:


SCRIBE = crowsetta.Transcriber(format='simple-seq')


def get_labels_from_csv_paths(csv_paths: list[pathlib.Path]):
    return [
        SCRIBE.from_file(csv_path).to_seq().labels
        for csv_path in csv_paths
    ]


# ### `get_split_wav_paths`
#
# Gets splits at the ID level. This should work for birdsong when we are training per-individual models.
#
# To really be able to compare with a generic segmenting model trained on multiple individuals, you would want the same amount of training data for both models. This raises the question of how you should allocate your data budget among multiple individuals.

# In[8]:


def get_split_wav_paths(
    data_dir: pathlib.Path,
    train_dur: float,
    test_dur: float,
    val_dur: float,
    labelset: set,
    unit: str = 'syllable'
):
    """Helper function that gets dataset splits: train, val, and test.

    Returns dict mapping the name of each split to a list of wav paths.
    These wav paths can then be matched with inputs and targets as required
    for training different models.

    Uses :func:`vak.prep.split.split.train_tset_dur_split_inds`."""
    wav_paths = voc.paths.from_dir(data_dir, 'wav')
    csv_paths = voc.paths.from_dir(data_dir, f".{unit}.csv")
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


# In[9]:


TARGET_COLUMNS = [
    "multi_frame_labels_path",
    "binary_frame_labels_path",
    "boundary_onehot_path",
]


def split_wav_paths_to_df(
    split_wav_paths:dict,
    unit: str = 'syllable',
    timebin_dur_str: str = "1",
    target_columns=TARGET_COLUMNS,
    annot_format="simple-seq",
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


# For canaries, we report training on 3600s, and I had guesstimated to use a training data pool of 10k seconds, then do 3 3600s grabs from that for training replicates.
# Close to what we did in the TweetyNet paper.
#
# For Birdsong-Recognition dataset, Bird0 has just 1380s of data.
# So that limits how much we can train on -- we can't do 1800s training sets for the Birdsong-Recognition birds.
# Probably better to do Leave-One-Bird-Out with the BFSongRepo birds because there we have much more data.

# In[10]:


def get_df_with_train_subset(
    df: pd.DataFrame,
    train_subset_dur: float,
    labelset: set,
    keep_val_and_test_split: bool,
    n_iters=10
):
    """Given a DataFrame representing splits in a dataset,
    return a new DataFrame where the 'train' split is a subset of the total training data
    in the original DataFrame."""
    df_train = df[df.split == 'train'].copy()
    labels = vak.common.labels.from_df(df_train, biosoundsegbench.prep.constants.DATASET_ROOT)
    durs = df_train['duration'].values

    # Because of how `train_test_dur_split_inds` works, we will get a split with duration >= train_subset_dur.
    # Instead of taking the time to figure out how to get as close as possible to the target duration
    # with a constrained optimization problem, we take the lazy way out for now:
    # we make multiple subsets and take the one with the duration closest to the target
    subset_dfs = []
    for _ in range(n_iters):
        train_inds, val_inds, test_inds = vak.prep.split.split.train_test_dur_split_inds(
            durs=durs,
            labels=labels,
            labelset=labelset,
            train_dur=train_subset_dur,
            val_dur=None,
            test_dur=None,
        )
        subset_dfs.append(
            df_train.iloc[sorted(train_inds), :].reset_index(drop=True)
        )
    subset_durs = np.array([df.duration.sum() for df in subset_dfs])
    ind_of_df_with_dur_closest_to_target = np.argmin([subset_dur - train_subset_dur for subset_dur in subset_durs])
    df_out = subset_dfs[ind_of_df_with_dur_closest_to_target]
    if keep_val_and_test_split:
        df_out = pd.concat(
            (df_out, df[df.split.isin(('val', 'test'))])
        )
    return df_out


# In[11]:


def get_splits_csv_filename_id(biosound_group: str, id: str, timebin_dur_str, unit: str) -> str:
    """Get filename for csv that contains all splits for one ID.

    These splits are used to get subsets of training data for each training replicate,
    while holding the validation and test set constant.
    """
    return f"{biosound_group}.id-{id}.timebin-{timebin_dur_str}-ms.{unit}.splits.csv"


# In[12]:


def get_replicate_csv_filename_id_data_only(
    biosound_group: str, id: str, timebin_dur_str, unit: str, train_dur: float, replicate_num: int,
) -> str:
    """Get filename for csv that contains the split for one training replicate for one ID.

    This function is used when all the splits contain *only* data from the ID.

    The 'train' split will contain a subset of the training data in the total splits.
    The validation and test set will be the same as for other training replicates.
    """
    return f"{biosound_group}.id-{id}.timebin-{timebin_dur_str}-ms.{unit}.id-data-only.train-dur-{train_dur}.replicate-{replicate_num}.splits.csv"


# In[13]:


def get_replicate_csv_filename_leave_one_id_out(
    biosound_group: str, id: str, timebin_dur_str, unit: str, train_dur: float, replicate_num: int,
) -> str:
    """Get filename for csv that contains the split for one training replicate for one ID.

    This function is used when all the splits contain *only* data from the ID.

    The 'train' split will contain a subset of the training data in the total splits.
    The validation and test set will be the same as for other training replicates.
    """
    return f"{biosound_group}.id-{id}.timebin-{timebin_dur_str}-ms.{unit}.leave-one-id-out.train-dur-{train_dur}.replicate-{replicate_num}.splits.csv"


# In[14]:


BIOSOUND_GROUP_DIR_MAP = {
    'Bengalese-Finch-Song': biosoundsegbench.prep.constants.BF_DATA_DST,
    'Canary-Song': biosoundsegbench.prep.constants.CANARY_DATA_DST,
    'Mouse-Pup-Call': biosoundsegbench.prep.constants.MOUSE_PUP_CALL_DATA_DST,
    'Zebra-Finch-Song': biosoundsegbench.prep.constants.ZB_DATA_DST,
}


def get_splits_df_and_replicate_dfs_per_id(
    biosound_group: str,
    unit: str,
    timebin_dur_str: str,
    total_train_dur: float,
    val_dur: float,
    test_dur: float,
    train_subset_dur: float,
    num_replicates: int,
    target_columns=TARGET_COLUMNS,
    dry_run=True,
):
    group_dir = BIOSOUND_GROUP_DIR_MAP[biosound_group]
    id_dirs = [
        subdir
        for subdir in group_dir.iterdir()
        if subdir.is_dir()
    ]
    print(
        f"Found ID dirs:\n{id_dirs}"
    )
    labelsets = biosoundsegbench.prep.labels.get_labelsets()[biosound_group][unit]

    id_splits_df_map = {}
    id_replicate_dfs_map = defaultdict(list)
    for id_dir in id_dirs:
        id = id_dir.name.split('-')[-1]
        print(
            f"Getting splits for ID: {id}"
        )
        labelset = set(
            labelsets[id]
        )
        split_wav_paths = get_split_wav_paths(
            id_dir,
            train_dur=total_train_dur,
            test_dur=test_dur,
            val_dur=val_dur,
            labelset=labelset,
            unit=unit
        )
        splits_df = split_wav_paths_to_df(
            split_wav_paths,
            unit,
            timebin_dur_str,
            target_columns,
        )
        id_splits_df_map[id] = splits_df

        splits_csv_filename = get_splits_csv_filename_id(biosound_group, id, timebin_dur_str, unit)
        splits_csv_path = biosoundsegbench.prep.constants.DATASET_ROOT / splits_csv_filename
        print(
            f"Saving splits as: {splits_csv_path}"
        )
        if not dry_run:
            splits_df.to_csv(splits_csv_path, index=False)

        print(
            f"Getting {num_replicates} training replicates, "
            f"each with training split that has a duration of {train_subset_dur} s."
        )
        for replicate_num in range(1, num_replicates + 1):
            replicate_df = get_df_with_train_subset(
                splits_df, train_subset_dur, labelset, keep_val_and_test_split=True,
            )
            id_replicate_dfs_map[id].append(replicate_df)
            replicate_csv_filename = get_replicate_csv_filename_id_data_only(
                biosound_group, id, timebin_dur_str, unit, train_subset_dur, replicate_num
            )
            replicate_csv_path = biosoundsegbench.prep.constants.DATASET_ROOT / replicate_csv_filename
            print(
                f"Saving replicate as: {replicate_csv_path}"
            )
            if not dry_run:
                replicate_df.to_csv(replicate_csv_path, index=False)

    return id_splits_df_map, id_replicate_dfs_map


# In[20]:


def make_leave_one_out_df_per_id(
    id_replicate_df_map: dict,
    biosound_group: str,
    unit: str,
    train_subset_dur: float | None = None,
    use_id_all_for_labelset=False,
):
    labelsets = biosoundsegbench.prep.labels.get_labelsets()[biosound_group][unit]

    id_leave_one_out_df_map = {}
    ids = list(id_replicate_df_map.keys())
    for test_id in ids:
        leave_one_out_dfs = []

        train_ids = [id_ for id_ in ids if id_ != test_id]
        for train_id in train_ids:
            train_id_df = id_replicate_df_map[train_id]
            if train_subset_dur is not None:
                if use_id_all_for_labelset:
                    labelset = set(labelsets['all'])
                else:
                    labelset = set(labelsets[train_id])
                subset_df = get_df_with_train_subset(
                    train_id_df, train_subset_dur, labelset, keep_val_and_test_split=False
                )
                leave_one_out_dfs.append(subset_df)
            else:
                leave_one_out_dfs.append(train_id_df)

        test_id_df = id_replicate_df_map[test_id]
        test_id_df = test_id_df[test_id_df.split.isin(('val', 'test'))].copy()
        leave_one_out_dfs.append(test_id_df)

        id_leave_one_out_df_map[test_id]  = pd.concat(leave_one_out_dfs)
    return id_leave_one_out_df_map


# In[23]:


def make_splits_per_id(
    biosound_group: str,
    unit: str,
    timebin_dur_str: str,
    total_train_dur: int,
    val_dur: int,
    test_dur: int,
    train_subset_dur_id_only: int,
    num_replicates: int,
    train_subset_dur_leave_one_id_out: int | None = None,
    target_columns: list[str] =TARGET_COLUMNS,
    dry_run: bool = True,
    use_id_all_for_labelset: bool = False
):
    """Make splits per ID.

    This function first makes splits where all data in each split--training, validation, and test--is from
    one ID in a group. The semantics of 'ID' depends on the group:
    for Bengalese finch song, canary song, and Zebra finch song,
    it is per-individual; for mouse pup calls, an "ID" is a species.

    For each ID, we get a pool of training data specified by `total_train_dur`;
    this pool along with the fixed validation and test set are trained in 'split' csvs.
    Then from the pool of training data we take subsets of duration `train_subset_dur_id_only`,
    to make splits for each training replicate, for a total of `num_replicates` splits.

    Then this function makes "leave-one-ID-out" splits.
    For these splits, the test and validation split contains data from one ID in a group,
    and the training set contains data from all other IDs in the group.
    In order to compare models trained on data from one ID only with models
    trained on data from multiple IDs, we create training sets of the same duration
    for both scenarios.
    To do this, for each ID in the training set we use a
    subset duration equal to train_subset_dur_id_only / (total number of IDs in the group - 1).
    This is done by default when `train_subset_dur_leave_one_id_out` has no argument and so is None.
    E.g., if for each training replicate using one-ID-only data we had
    a training split of 700 seconds, and there were 8 IDs total in the group,
    then for the leave-one-ID-out splits we would use 700 / (8 -1) = 100 seconds of data
    from each ID in the training set.

    Finally we make another set of leave-one-out splits where we combine *all* the training data
    from all other IDs.
    """
    print(
        f"Making per-ID splits for biosound group '{biosound_group}', unit '{unit}', and timebin {timebin_dur_str}. "
        f"Splits will have a training split with duration of {total_train_dur} s, a validation split with duration of {val_dur} s, "
        f"and a test set with duration of {test_dur}. For each ID will there will be {num_replicates} training replicates, "
        f"that subsets the training data split to a duration of {train_subset_dur_id_only} s."
    )
    id_splits_df_map, id_replicate_dfs_map = get_splits_df_and_replicate_dfs_per_id(
        biosound_group,
        unit,
        timebin_dur_str,
        total_train_dur,
        val_dur,
        test_dur,
        train_subset_dur_id_only,
        num_replicates,
        target_columns,
        dry_run,
    )

    # make leave-one-out csvs
    ids = list(id_splits_df_map.keys())
    if train_subset_dur_leave_one_id_out is None:
        # we want the same duration for the training set, but with the data
        # divided equally among each of the IDs that will be in the training set
        train_subset_dur_leave_one_id_out = train_subset_dur_id_only / (len(ids) - 1)

    # note here we grab subsets from the training replicate *subsets* for each ID;
    # we're not grabbing from the total training pool
    for replicate_num in range(1, num_replicates + 1):
        id_replicate_df_map = {
            id_: id_replicate_dfs_map[id_][replicate_num - 1]
            for id_ in ids
        }
        id_leave_one_out_df_map = make_leave_one_out_df_per_id(
            id_replicate_df_map,
            biosound_group,
            unit,
            train_subset_dur_leave_one_id_out,
            use_id_all_for_labelset,
        )
        for id, leave_one_out_df in id_leave_one_out_df_map.items():
            replicate_csv_filename = get_replicate_csv_filename_leave_one_id_out(
                biosound_group, id, timebin_dur_str, unit, train_subset_dur_leave_one_id_out, replicate_num
            )
            replicate_csv_path = biosoundsegbench.prep.constants.DATASET_ROOT / replicate_csv_filename
            print(
                f"Saving replicate as: {replicate_csv_path}"
            )
            if not dry_run:
                replicate_df.to_csv(replicate_csv_path, index=False)

    # TODO: now make the leave-one-out CSVs where we use *all* the training data for each ID
    n_train_ids = len(ids) - 1
    train_dur_from_n_train_ids = train_subset_dur_id_only * n_train_ids
    for replicate_num in range(1, num_replicates + 1):
        id_replicate_df_map = {
            id_: id_replicate_dfs_map[id_][replicate_num - 1]
            for id_ in ids
        }
        id_leave_one_out_df_map = make_leave_one_out_df_per_id(
            id_replicate_df_map,
            biosound_group,
            unit,
            train_subset_dur_leave_one_id_out=None,
            use_id_all_for_labelset=use_id_all_for_labelset,
        )
        for id, leave_one_out_df in id_leave_one_out_df_map.items():
            replicate_csv_filename = get_replicate_csv_filename_leave_one_id_out(
                biosound_group, id, timebin_dur_str, unit, train_dur_from_n_train_ids, replicate_num
            )
            replicate_csv_path = biosoundsegbench.prep.constants.DATASET_ROOT / replicate_csv_filename
            print(
                f"Saving replicate as: {replicate_csv_path}"
            )
            if not dry_run:
                replicate_df.to_csv(replicate_csv_path, index=False)


@dataclass
class BiosoundGroupSplit:
    subgroup: str
    unit: str
    train_pool: int
    train: int
    val: int
    test: int
    num_replicates: int


BIOSOUND_GROUP_SPLITS = {
    "Bengalese-Finch-Song": [
        ("id", "syllable", {'train': 900, 'val': 80, 'test': 400}),
    ]

}


def make_splits_all(biosound_groups: list[str], dry_run=True) -> None:
    if "Bengalese-Finch-Song" in biosound_groups:
        logger.info(
            f"Making inputs and targets for Bengalese finch song."
        )
        make_inputs_and_targets_bengalese_finch_song(dry_run)

    if "Canary-Song" in biosound_groups:
        logger.info(
            f"Making inputs and targets for canary song."
        )
        make_inputs_and_targets_canary_song(dry_run)

    if "Mouse-Pup-Call" in biosound_groups:
        logger.info(
            f"Making inputs and targets for mouse pup calls."
        )
        make_inputs_and_targets_mouse_pup_call(dry_run)

    if "Zebra-Finch-Song" in biosound_groups:
        logger.info(
            f"Making inputs and targets for Zebra finch song."
        )
        make_inputs_and_targets_zebra_finch_song(dry_run)

    if "Human-Speech" in biosound_groups:
        logger.info(
            f"Making inputs and targets for human speech."
        )
        make_inputs_and_targets_human_speech(dry_run)

