"""Functions to make dataset splits--training, validation, and test--and save as csv files in dataset root."""
import copy
import dataclasses
import json
import logging
import pathlib
import collections

import crowsetta
import numpy as np
import pandas as pd
import pandera.errors
import tqdm
import vak
import vocalpy as voc

from . import constants, labels


logger = logging.getLogger(__name__)


def get_durs_from_wav_paths(wav_paths: list[pathlib.Path]):
    """Get list of durations of wav files in seconds,
    given a list of paths to the wav files"""
    durs = []
    for wav_path in wav_paths:
        sound = voc.Audio.read(wav_path)
        durs.append(
            sound.data.shape[-1] / sound.samplerate
        )
    return durs


SCRIBE = crowsetta.Transcriber(format='simple-seq')


def get_labels_from_csv_paths(csv_paths: list[pathlib.Path]):
    """Get list of labels in csv annotation files,
    given a list of paths to the csv files"""
    all_labels = []
    for csv_path in csv_paths:
        try:
            labels = SCRIBE.from_file(csv_path).to_seq().labels
        except pandera.errors.SchemaError as e:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                labels = np.array([])
            else:
                raise ValueError(
                    f"Unable to parse csv_path: {csv_path}"
                ) from e
        all_labels.append(labels)
    return all_labels


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

    # here we get total pool of training data
    # later we will make subsets of training data for each training replicate
    train_inds, val_inds, test_inds = vak.prep.split.split.train_test_dur_split_inds(
        durs,
        labels,
        labelset,
        train_dur,
        test_dur,
        val_dur,
    )

    return {
        'train': [wav_paths[train_ind] for train_ind in train_inds],
        'val': [wav_paths[val_ind] for val_ind in val_inds],
        'test': [wav_paths[test_ind] for test_ind in test_inds],
    }


TARGET_COLUMNS = [
    "multi_frame_labels_path",
    "binary_frame_labels_path",
    "boundary_frame_labels_path",
]


def split_wav_paths_to_df(
    split_wav_paths:dict,
    unit: str = 'syllable',
    frame_dur_str: str = "1",
    target_columns=TARGET_COLUMNS,
    annot_format="simple-seq",
):
    """Takes dictionary mapping split names to wav paths,
    and builds a DataFrame with paths to the inputs and targets
    for neural network models.

    This function takes advantage of the fact that we use a naming scheme
    based on the wav files, e.g. the inputs for a network have a name
    like "file1.wav.frames.npz" and the multi-class frame label vectors
    have names like "file1.wav.multi-frame-labels.npy".
    """
    if not all(
        [target_column in TARGET_COLUMNS for target_column in target_columns]
    ):
        invalid = set(target_columns) - set(TARGET_COLUMNS)
        raise ValueError(
            f"Invalid target column(s): {invalid}.\n"
            f"Valid columns are: {TARGET_COLUMNS}"
        )

    frame_dur = float(frame_dur_str) * 1e-3
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

            frames_path = parent / (name + f".timebin-{frame_dur_str}-ms.frames.npz")
            if not frames_path.exists():
                raise FileNotFoundError(
                    f"Did not find `frames_path` for `audio_path`.\n"
                    f"`audio_path`: {audio_path}"
                    f"`frames_path`: {frames_path}"
                )

            record = {
                'audio_path': str(
                    audio_path.relative_to(constants.DATASET_ROOT)
                ),
                'annot_path': str(
                    annot_path.relative_to(constants.DATASET_ROOT)
                ),
                'frames_path': str(
                    frames_path.relative_to(constants.DATASET_ROOT)
                ),
                'frame_dur': frame_dur,
                'duration': duration,
                'annot_format': annot_format,
                'split': split,
            }

            if "multi_frame_labels_path" in target_columns:
                multi_frame_labels_path = parent / (
                    name + f".timebin-{frame_dur_str}-ms.{unit}.multi-frame-labels.npy"
                )
                if not multi_frame_labels_path.exists():
                    raise FileNotFoundError(
                        f"Did not find `multi_frame_labels_path` for `audio_path`.\n"
                        f"`audio_path`: {audio_path}\n"
                        f"`multi_frame_labels_path`: {multi_frame_labels_path}\n"
                    )
                record['multi_frame_labels_path'] = str(
                    multi_frame_labels_path.relative_to(constants.DATASET_ROOT)
                )

            if "binary_frame_labels_path" in target_columns:
                binary_frame_labels_path = parent / (
                    name + f".timebin-{frame_dur_str}-ms.{unit}.binary-frame-labels.npy"
                )
                if not binary_frame_labels_path.exists():
                    raise FileNotFoundError(
                        f"Did not find `binary_frame_labels_path` for `audio_path`.\n"
                        f"`audio_path`: {audio_path}"
                        f"`binary_frame_labels_path`: {binary_frame_labels_path}"
                    )
                record['binary_frame_labels_path'] = str(
                    binary_frame_labels_path.relative_to(constants.DATASET_ROOT)
                )

            if "boundary_frame_labels_path" in target_columns:
                boundary_frame_labels_path = parent / (
                    name + f".timebin-{frame_dur_str}-ms.{unit}.boundary-frame-labels.npy"
                )
                if not boundary_frame_labels_path.exists():
                    raise FileNotFoundError(
                        f"Did not find `boundary_frame_labels_path` for `audio_path`.\n"
                        f"`audio_path`: {audio_path}"
                        f"`boundary_frame_labels_path`: {boundary_frame_labels_path}"
                    )
                record['boundary_frame_labels_path'] = str(
                    boundary_frame_labels_path.relative_to(constants.DATASET_ROOT)
                )
            records.append(record)
    df = pd.DataFrame.from_records(records)
    return df


def get_df_with_train_subset(
    df: pd.DataFrame,
    train_subset_dur: float,
    labelset: set,
    keep_val_and_test_split: bool,
    n_iters=10,
):
    """Given a DataFrame representing splits in a dataset,
    return a new DataFrame where the 'train' split is a subset of the total training data
    in the original DataFrame."""
    df_train = df[df.split == 'train'].copy()
    csv_paths = [constants.DATASET_ROOT / csv_path
                 for csv_path in df_train.annot_path.values]
    labels = get_labels_from_csv_paths(csv_paths)
    durs = df_train['duration'].values

    # Because of how `train_test_dur_split_inds` works, we will get a split with duration >= train_subset_dur.
    # Instead of taking the time to figure out how to get as close as possible to the target duration
    # with a constrained optimization problem, we take the lazy way out for now:
    # we make multiple subsets and take the one with the duration closest to the target
    subset_dfs = []
    for _ in range(n_iters):
        train_inds, _, _ = vak.prep.split.split.train_test_dur_split_inds(
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
    logger.info(
        f"Actual duration of randomly-drawn subset of training split: {subset_durs[ind_of_df_with_dur_closest_to_target]}"
    )
    df_out = subset_dfs[ind_of_df_with_dur_closest_to_target]
    if keep_val_and_test_split:
        logger.info(
            f"Adding validation and test splits to subset of training split"
        )
        df_out = pd.concat(
            (df_out, df[df.split.isin(('val', 'test'))])
        )
    else:
        logger.info(
            f"Not adding validation and test splits to subset of training split, just returning training subset"
        )
    return df_out


def get_splits_csv_filename_id(biosound_group: str, id: str, frame_dur_str, unit: str) -> str:
    """Get filename for csv that contains all splits for one ID.

    These splits are used to get subsets of training data for each training replicate,
    while holding the validation and test set constant.
    """
    return f"{biosound_group}.{unit}.id-{id}.frame-dur-{frame_dur_str}-ms.splits.csv"


def get_replicate_csv_filename_id_data_only(
    biosound_group: str, id: str, frame_dur_str, unit: str, train_dur: float, replicate_num: int,
) -> str:
    """Get filename for csv that contains the split for one training replicate for one ID.

    This function is used when all the splits contain *only* data from the ID.

    The 'train' split will contain a subset of the training data in the total splits.
    The validation and test set will be the same as for other training replicates.
    """
    return f"{biosound_group}.id-{id}.timebin-{frame_dur_str}-ms.{unit}.id-data-only.train-dur-{train_dur:.1f}.replicate-{replicate_num}.splits.csv"


def get_replicate_csv_filename_leave_one_id_out(
    biosound_group: str, id: str, frame_dur_str, unit: str, train_dur: float, replicate_num: int,
) -> str:
    """Get filename for csv that contains the split for one training replicate for one ID.

    This function is used when all the splits contain *only* data from the ID.

    The 'train' split will contain a subset of the training data in the total splits.
    The validation and test set will be the same as for other training replicates.
    """
    return f"{biosound_group}.id-{id}.timebin-{frame_dur_str}-ms.{unit}.leave-one-id-out.train-dur-{train_dur}.replicate-{replicate_num}.splits.csv"


BIOSOUND_GROUP_DIR_MAP = {
    'Bengalese-Finch-Song': constants.BF_DATA_DST,
    'Canary-Song': constants.CANARY_DATA_DST,
    'Mouse-Pup-Call': constants.MOUSE_PUP_CALL_DATA_DST,
    'Zebra-Finch-Song': constants.ZB_DATA_DST,
}


def get_splits_df_and_replicate_dfs_per_id(
    biosound_group: str,
    unit: str,
    frame_dur_str: str,
    total_train_dur: float,
    val_dur: float,
    test_dur: float,
    train_subset_dur: float,
    num_replicates: int,
    target_columns=TARGET_COLUMNS,
    use_id_all_for_labelset: bool = False,
    dry_run=True,
):
    group_dir = BIOSOUND_GROUP_DIR_MAP[biosound_group]
    id_dirs = [
        subdir
        for subdir in group_dir.iterdir()
        if subdir.is_dir()
    ]
    logger.info(
        f"Found ID dirs:\n{id_dirs}"
    )
    labelsets = labels.get_labelsets()[biosound_group][unit]

    id_splits_df_map = {}
    id_replicate_dfs_map = collections.defaultdict(list)
    replicate_csv_paths = []
    for id_dir in id_dirs:
        id = id_dir.name.split('-')[-1]
        logger.info(
            f"Getting splits for ID: {id}"
        )
        if biosound_group == "Mouse-Pup-Call" and unit == "call":
            # we have to special-case mouse pup call:
            # since we label segments with species class, data from a single ID only has labels from that ID
            labelset = set([id])
        elif use_id_all_for_labelset:
            # this applies to human speech only
            labelset = set(labelsets['all'])
        else:
            labelset = set(
                labelsets[id]
            )
        logger.info(
            f"Getting wav paths for splits:\ntraining split with duration of {total_train_dur} s\n"
            f"validation split with duration of {val_dur} s\n"
            f"test set with duration of {test_dur}\n"
        )
        split_wav_paths = get_split_wav_paths(
            id_dir,
            train_dur=total_train_dur,
            test_dur=test_dur,
            val_dur=val_dur,
            labelset=labelset,
            unit=unit
        )
        for split_name, wav_paths_list in split_wav_paths.items():
            logger.info(
                f"Found {len(wav_paths_list)} wav paths for split '{split_name}'"
            )
        logger.info(
            f"Building DataFrame with neural network labels and targets from wav paths."
        )
        splits_df = split_wav_paths_to_df(
            split_wav_paths,
            unit,
            frame_dur_str,
            target_columns,
        )
        splits_df['id'] = id
        splits_df['group'] = biosound_group
        splits_from_df = sorted(splits_df.split.unique())
        logger.info(
            f"Splits in DataFrame with all splits: {splits_from_df}"
        )
        for split in splits_from_df:
            logger.info(
                f"Split '{split}' has duration: {splits_df[splits_df.split == split].duration.sum()}"
            )
        id_splits_df_map[id] = splits_df

        splits_csv_filename = get_splits_csv_filename_id(biosound_group, id, frame_dur_str, unit)
        splits_csv_path = constants.INPUTS_TARGETS_PATHS_CSVS_DIR / splits_csv_filename
        logger.info(
            f"Saving splits as: {splits_csv_path}"
        )
        if not dry_run:
            splits_df.to_csv(splits_csv_path, index=False)

        logger.info(
            f"Getting {num_replicates} training replicates, "
            f"each with training split that has a duration of {train_subset_dur} s."
        )
        for replicate_num in range(1, num_replicates + 1):
            logger.info(
                f"Getting subset of training data for training replicate {replicate_num}."
            )
            replicate_df = get_df_with_train_subset(
                splits_df, train_subset_dur, labelset, keep_val_and_test_split=True,
            )
            splits_from_df = sorted(replicate_df.split.unique())
            logger.info(
                f"Splits in DataFrame for training replicate: {splits_from_df}"
            )
            for split in splits_from_df:
                logger.info(
                    f"Split '{split}' has duration: {replicate_df[replicate_df.split == split].duration.sum()}"
                )

            id_replicate_dfs_map[id].append(replicate_df)
            replicate_csv_filename = get_replicate_csv_filename_id_data_only(
                biosound_group, id, frame_dur_str, unit, train_subset_dur, replicate_num
            )
            replicate_csv_path = constants.INPUTS_TARGETS_PATHS_CSVS_DIR / replicate_csv_filename
            logger.info(
                f"Saving replicate as: {replicate_csv_path}"
            )
            if not dry_run:
                replicate_df.to_csv(replicate_csv_path, index=False)
            replicate_csv_paths.append(replicate_csv_path)

    return id_splits_df_map, id_replicate_dfs_map, replicate_csv_paths


def make_leave_one_out_df_per_id(
    id_replicate_df_map: dict,
    biosound_group: str,
    unit: str,
    train_subset_dur: float,
    train_subset_dur_per_id: float | None = None,
    use_id_all_for_labelset=False,
    n_iter=100,
):
    """Make DataFrames representing splits for leave-one-ID-out experiments.

    Takes a dict mapping IDs from a group to a list of DataFrames,
    one DataFrame for each training replicate.
    For each ID, combines training splits from all other IDs,
    and keeps the same validation and test splits for that ID.
    """
    labelsets = labels.get_labelsets()[biosound_group][unit]

    id_leave_one_out_df_map = {}
    ids = list(id_replicate_df_map.keys())
    for test_id in ids:
        # list of DataFrames that we pd.concat after loop
        leave_one_out_df = []

        train_ids = [id_ for id_ in ids if id_ != test_id]
        for train_id in train_ids:
            train_id_df = id_replicate_df_map[train_id]
            if train_subset_dur_per_id is None:
                # remove val and test splits for this ID "manually" (instead of in function call as in 'if' block)
                train_id_df = train_id_df[train_id_df.split == 'train'].copy()
                leave_one_out_df.append(train_id_df)
            else:
                # we get a training subset of a specified duration **per ID**
                if biosound_group == "Mouse-Pup-Call" and unit == "call":
                    # we have to special-case mouse pup call:
                    # since we label segments with species class, data from a single ID only has labels from that ID
                    labelset = set([train_id])
                elif use_id_all_for_labelset:
                    labelset = set(labelsets['all'])
                else:
                    labelset = set(labelsets[train_id])
                subset_df = get_df_with_train_subset(
                    train_id_df, train_subset_dur_per_id, labelset, keep_val_and_test_split=False
                )
                leave_one_out_df.append(subset_df)

        test_id_df = id_replicate_df_map[test_id]
        test_id_df = test_id_df[test_id_df.split.isin(('val', 'test'))].copy()
        leave_one_out_df.append(test_id_df)

        leave_one_out_df = pd.concat(leave_one_out_df)
        if train_subset_dur_per_id is not None:
            # we need to take another subset so we don't
            # end up having a total training split > train_subset_dir.
            # This happens because of how we draw subsets -- would need a better method to avoid
            if biosound_group == "Mouse-Pup-Call" and unit == "call":
                # we have to special-case mouse pup call:
                # since we label segments with species class, data from a single ID only has labels from that ID
                labelset = set(train_ids)
            elif use_id_all_for_labelset:
                labelset = set(labelsets['all'])
            else:
                labelset = []
                for train_id in train_ids:
                    labelset.extend(labelsets[train_id])
                labelset = set(labelset)
            for iter_n in tqdm.tqdm(range(n_iter)):
                tmp_df = get_df_with_train_subset(
                    leave_one_out_df, train_subset_dur, labelset, keep_val_and_test_split=True
                )
                if set(tmp_df[tmp_df.split == 'train'].id.unique()) == set(train_ids):
                    leave_one_out_df = tmp_df
                    break
                if iter_n == n_iter - 1:
                    raise ValueError(
                        f"Could not find subset with all train IDs in train split"
                    )

        splits_from_df = sorted(leave_one_out_df.split.unique())
        logger.info(
            f"Splits in DataFrame for leave-one-ID-out training replicate: {splits_from_df}"
        )
        for split in splits_from_df:
            logger.info(
                f"Split '{split}' has duration: {leave_one_out_df[leave_one_out_df.split == split].duration.sum()}"
            )
            logger.info(
                f"Split '{split}' has IDs: {sorted(leave_one_out_df[leave_one_out_df.split == split].id.unique())}"
            )
        id_leave_one_out_df_map[test_id]  = leave_one_out_df
    return id_leave_one_out_df_map


def make_splits_per_id(
    biosound_group: str,
    unit: str,
    frame_dur_str: str,
    total_train_dur: float,
    val_dur: float,
    test_dur: float,
    train_subset_dur_id_only: float,
    num_replicates: int,
    train_subset_dur_leave_one_id_out: int | None = None,
    target_columns: list[str] =TARGET_COLUMNS,
    dry_run: bool = True,
    use_id_all_for_labelset: bool = False,
    make_leave_one_id_out_splits=True,
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
    logger.info(
        f"Making per-ID splits for biosound group '{biosound_group}', unit '{unit}', and timebin {frame_dur_str}. "
        f"Splits will have a training split with duration of {total_train_dur} s, a validation split with duration of {val_dur} s, "
        f"and a test set with duration of {test_dur}. For each ID will there will be {num_replicates} training replicates, "
        f"that subsets the training data split to a duration of {train_subset_dur_id_only} s."
    )

    id_splits_df_map, id_replicate_dfs_map, replicate_csv_paths = get_splits_df_and_replicate_dfs_per_id(
        biosound_group,
        unit,
        frame_dur_str,
        total_train_dur,
        val_dur,
        test_dur,
        train_subset_dur_id_only,
        num_replicates,
        target_columns,
        use_id_all_for_labelset,
        dry_run,
    )

    if make_leave_one_id_out_splits:
        # make leave-one-out csvs
        logger.info(
            "\nMaking leave-one-ID-out splits for each training replicate, with same-sized training set duration"
        )
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
                train_subset_dur_per_id=train_subset_dur_leave_one_id_out,
                train_subset_dur=train_subset_dur_id_only,
                use_id_all_for_labelset=use_id_all_for_labelset,
            )
            for id, leave_one_out_df in id_leave_one_out_df_map.items():
                replicate_csv_filename = get_replicate_csv_filename_leave_one_id_out(
                    biosound_group, id, frame_dur_str, unit, train_subset_dur_id_only, replicate_num
                )
                replicate_csv_path = constants.INPUTS_TARGETS_PATHS_CSVS_DIR / replicate_csv_filename
                logger.info(
                    f"Saving replicate as: {replicate_csv_path}"
                )
                if not dry_run:
                    leave_one_out_df.to_csv(replicate_csv_path, index=False)
                replicate_csv_paths.append(replicate_csv_path)

        # now make the leave-one-out CSVs where we use *all* the training data for each ID
        logger.info(
            "\nMaking leave-one-ID-out splits for each training replicate, "
            "with training set duration = per-ID duration * num. train IDs"
        )
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
                train_subset_dur=train_subset_dur_id_only * n_train_ids,
                train_subset_dur_per_id=None,
                use_id_all_for_labelset=use_id_all_for_labelset,
            )
            for id, leave_one_out_df in id_leave_one_out_df_map.items():
                replicate_csv_filename = get_replicate_csv_filename_leave_one_id_out(
                    biosound_group, id, frame_dur_str, unit, train_dur_from_n_train_ids, replicate_num
                )
                replicate_csv_path = constants.INPUTS_TARGETS_PATHS_CSVS_DIR / replicate_csv_filename
                logger.info(
                    f"Saving replicate as: {replicate_csv_path}"
                )
                if not dry_run:
                    leave_one_out_df.to_csv(replicate_csv_path, index=False)
                replicate_csv_paths.append(replicate_csv_path)
    return replicate_csv_paths


def get_timit_train_data_dirs():
    """Get list of directories with training data for TIMIT.

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


def get_timit_test_data_dirs():
    """Get list of directories with test data for TIMIT.

    Each directory contains data from one speaker from one dialect region.
    The directories are sub-directories in :data:`biosoundsegbench.prep.constants.SPEECH_DATA_DST`.
    The source data is from the NLTK Sample of the TIMIT corpus.
    """
    return sorted(
        [id_dir
         for id_dir in constants.SPEECH_DATA_DST.iterdir()
         if id_dir.is_dir()]
    )


def make_splits_timit(
    total_train_dur: float,
    val_dur: float,
    train_subset_dur: float,
    num_replicates: int = 3,
    dry_run: bool = True,
):
    # test set stays the same for all training replicates, we get that first
    logger.info(
        f"Getting test data for TIMIT."
    )
    timit_test_dirs = get_timit_test_data_dirs()
    test_wav_paths = []
    for timit_test_dir in timit_test_dirs:
        test_wav_paths.extend(voc.paths.from_dir(timit_test_dir, 'wav'))
    split_wav_paths = {}
    split_wav_paths['test'] = test_wav_paths

    timit_train_dirs = get_timit_train_data_dirs()
    wav_paths = []
    csv_paths = []
    for timit_train_dir in timit_train_dirs:
        wav_paths.extend(voc.paths.from_dir(timit_train_dir, 'wav'))
        csv_paths.extend(voc.paths.from_dir(timit_train_dir, '.phoneme.csv'))

    durs = get_durs_from_wav_paths(wav_paths)
    # next line, `labels_` to not clobber name of module
    labels_ = get_labels_from_csv_paths(csv_paths)
    labelset = set([lbl for labels_arr in labels_ for lbl in labels_arr])

    total_dur = sum(durs)
    logger.info(
        f"Total duration of TIMIT training data: {total_dur}"
    )

    logger.info(
        f"Getting indices for total training data pool with duration of {total_train_dur} s "
        f"and validation split with duration of {val_dur} s"
    )
    train_inds, val_inds, _ = vak.prep.split.split.train_test_dur_split_inds(
        durs,
        labels_,
        labelset,
        train_dur=total_train_dur,
        test_dur=None,
        val_dur=val_dur,
    )
    split_wav_paths['train'] = [wav_paths[train_ind] for train_ind in train_inds]
    split_wav_paths['val'] = [wav_paths[val_ind] for val_ind in val_inds]

    logger.info(
        f"Getting DataFrame with phoneme-level inputs and targets that have timebin duration of 10.0 ms"
    )
    splits_df = split_wav_paths_to_df(
        split_wav_paths,
        unit='phoneme',
        frame_dur_str="10.0",
        target_columns=['multi_frame_labels_path', 'boundary_frame_labels_path'],
    )
    splits_df['group'] = 'Human-Speech'
    splits_from_df = sorted(splits_df.split.unique())
    logger.info(
        f"Splits in DataFrame: {splits_from_df}"
    )
    for split in splits_from_df:
        logger.info(
            f"Split '{split}' has duration: {splits_df[splits_df.split == split].duration.sum()}"
        )

    split_csv_filename = f"Human-Speech.timebin-10.0-ms.phoneme.splits.csv"
    split_csv_path = constants.INPUTS_TARGETS_PATHS_CSVS_DIR / split_csv_filename
    if not dry_run:
        splits_df.to_csv(split_csv_path, index=False)

    replicate_csv_paths = []
    for replicate_num in range(1, num_replicates + 1):
        logger.info(
            f"Getting subset of training data for training replicate {replicate_num}."
        )
        replicate_df = get_df_with_train_subset(
            splits_df, train_subset_dur, labelset, keep_val_and_test_split=True,
        )
        splits_from_df = sorted(replicate_df.split.unique())
        logger.info(
            f"Splits in DataFrame for training replicate: {splits_from_df}"
        )
        for split in splits_from_df:
            logger.info(
                f"Split '{split}' has duration: {replicate_df[replicate_df.split == split].duration.sum()}"
            )

        replicate_csv_filename = f"Human-Speech.timebin-10.0-ms.phoneme.train-dur-{train_subset_dur}.replicate-{replicate_num}.splits.csv"
        replicate_csv_path = constants.INPUTS_TARGETS_PATHS_CSVS_DIR / replicate_csv_filename
        logger.info(
            f"Saving replicate as: {replicate_csv_path}"
        )
        if not dry_run:
            replicate_df.to_csv(replicate_csv_path, index=False)
        replicate_csv_paths.append(replicate_csv_path)

        # use replicate DataFrame to create 1.0-ms phoneme DataFrame + a word-level dataframe
        logger.info(
            "Using data from replicate to get DataFrame with 1.0 ms timebins, "
            "and DataFrame with word-level inputs and targets"
        )
        replicate_wav_paths_for_other_csvs = {}
        for split in replicate_df.split.unique():
            replicate_split_df = replicate_df[replicate_df.split == split].copy()
            wav_paths = [
                constants.DATASET_ROOT / wav_path
                for wav_path in replicate_split_df.audio_path.values
            ]
            replicate_wav_paths_for_other_csvs[split] = wav_paths

        replicate_1ms_df = split_wav_paths_to_df(
            replicate_wav_paths_for_other_csvs,
            unit='phoneme',
            frame_dur_str="1.0",
            target_columns=['multi_frame_labels_path', 'boundary_frame_labels_path'],
        )
        splits_from_df = sorted(replicate_1ms_df.split.unique())
        logger.info(
            f"Splits in DataFrame for 1.0 ms timebin training replicate: {splits_from_df}"
        )
        for split in splits_from_df:
            logger.info(
                f"Split '{split}' has duration: {replicate_1ms_df[replicate_1ms_df.split == split].duration.sum()}"
            )

        replicate_1ms_csv_filename = f"Human-Speech.timebin-1.0-ms.phoneme.train-dur-{train_subset_dur}.replicate-{replicate_num}.splits.csv"
        replicate_1ms_csv_path = constants.INPUTS_TARGETS_PATHS_CSVS_DIR / replicate_1ms_csv_filename
        logger.info(
            f"Saving replicate with 1.0 ms timebins as: {replicate_1ms_csv_path}"
        )
        if not dry_run:
            replicate_1ms_df.to_csv(replicate_1ms_csv_path, index=False)
        replicate_csv_paths.append(replicate_1ms_csv_path)

        replicate_word_df = split_wav_paths_to_df(
            replicate_wav_paths_for_other_csvs,
            unit='word',
            frame_dur_str="10.0",
            target_columns=['boundary_frame_labels_path'],
        )
        splits_from_df = sorted(replicate_word_df.split.unique())
        logger.info(
            f"Splits in DataFrame for word-level training replicate: {splits_from_df}"
        )
        for split in splits_from_df:
            logger.info(
                f"Split '{split}' has duration: {replicate_word_df[replicate_word_df.split == split].duration.sum()}"
            )

        replicate_word_csv_filename = f"Human-Speech.timebin-10.0-ms.word.train-dur-{train_subset_dur}.replicate-{replicate_num}.splits.csv"
        replicate_word_csv_path = constants.INPUTS_TARGETS_PATHS_CSVS_DIR / replicate_word_csv_filename
        logger.info(
            f"Saving replicate with word-level annotations as: {replicate_word_csv_path}"
        )
        if not dry_run:
            replicate_word_df.to_csv(replicate_word_csv_path, index=False)
        replicate_csv_paths.append(replicate_word_csv_path)
    return replicate_csv_paths


def argsort_by_label_freq(
        labels_lists: list[list[str]]
        ) -> list[int]:
    """Returns indices to sort a list of annotations
     in order of more frequently appearing labels,
     i.e., the first annotation will have the label
     that appears least frequently and the last annotation
     will have the label that appears most frequently.

    Used to sort a dataframe representing a dataset of annotated audio
    or spectrograms before cropping that dataset to a specified duration,
    so that it's less likely that cropping will remove all occurrences
    of any label class from the total dataset.

     Parameters
     ----------
     annots: list
         List of list of strings, labels from :class:`crowsetta.Annotation` instances.

     Returns
     -------
     sort_inds: list
         Integer values to sort ``annots``.
    """
    all_labels = [lbl for labels_list in labels_lists for lbl in labels_list]
    label_counts = collections.Counter(all_labels)

    sort_inds = []
    # make indices ahead of time so they stay constant as we remove things from the list
    ind_labels_tuples = list(enumerate(copy.deepcopy(labels_lists)))
    # starting with least common label, iterate through all labels list,
    # and if a labels list contains that label, add the corresponding ind to sort inds
    for label, _ in reversed(label_counts.most_common()):
        # next line, [:] to make a temporary copy
        # so the list we're iterating through doesn't shorten out from underneath us
        for ind_labels_tuple in ind_labels_tuples[:]:
            ind, labels_list = ind_labels_tuple
            if label in labels_list:
                sort_inds.append(ind)
                ind_labels_tuples.remove(ind_labels_tuple)

    # make sure we got all source_paths + annots
    if len(ind_labels_tuples) > 0:
        # next line, [:] to make a temporary copy
        # so the list we're iterating through doesn't shorten out from underneath us
        for ind_labels_tuple in ind_labels_tuples[:]:
            ind, _ = ind_labels_tuple
            sort_inds.append(ind)
            ind_labels_tuples.remove(ind_labels_tuple)

    if len(ind_labels_tuples) > 0:
        raise ValueError(
            "Not all ``labels_lists`` were used in sorting."
            f"Left over (with indices from list): {ind_labels_tuples}"
        )

    if not (sorted(sort_inds) == list(range(len(labels_lists)))):
        raise ValueError(
            "sorted(sort_inds) does not equal range(len(labels)):"
            f"sort_inds: {sort_inds}\nrange(len(annots)): {list(range(len(labels_lists)))}"
        )

    return sort_inds


def sample_vecs_and_splits_df_from_splits_csv_path(
        splits_csv_path: pathlib.Path
        ) -> pd.DataFrame:
    logger.info(
        f"Loading DataFrame from splits_csv_path: {splits_csv_path}"
    )
    splits_df = pd.read_csv(splits_csv_path)

    split_sample_id_vec_map = {}
    split_inds_in_sample_vec_map = {}
    splits_df_out = []
    for split in sorted(splits_df.split.unique()):
        logger.info(
            f"Processing split: {split}"
        )
        split_df = splits_df[splits_df.split == split].copy()
        csv_paths = [
            constants.DATASET_ROOT / csv_path
            for csv_path in split_df.annot_path.values
        ]
        labels_lists = get_labels_from_csv_paths(csv_paths)
        labels_lists = [
            labels_arr.tolist() for labels_arr in labels_lists
        ]

        sort_inds = argsort_by_label_freq(labels_lists)
        split_df["sort_inds"] = sort_inds
        split_df = split_df.sort_values(by="sort_inds").drop(
            columns="sort_inds"
        )
        splits_df_out.append(split_df)

        sample_id_vec = []
        inds_in_sample_vec = []
        pbar = tqdm.tqdm(split_df.frames_path.values)
        for source_id, frames_path in enumerate(pbar):
            pbar.set_description(
                f"Making sample/inds vec for: {pathlib.Path(frames_path).name}"
            )
            frames_path = constants.DATASET_ROOT / frames_path
            frames_dict = np.load(frames_path)
            frames = frames_dict['s']
            n_frames = frames.shape[-1]
            sample_id_vec.append(
                np.ones((n_frames,)).astype(np.int32) * source_id
            )
            inds_in_sample_vec.append(
                np.arange(n_frames)
            )
        sample_id_vec = np.concatenate(sample_id_vec)
        split_sample_id_vec_map[split] = sample_id_vec

        inds_in_sample_vec = np.concatenate(inds_in_sample_vec)
        split_inds_in_sample_vec_map[split] = inds_in_sample_vec

    splits_df_out = pd.concat(splits_df_out).reset_index(drop=True)
    return split_sample_id_vec_map, split_inds_in_sample_vec_map, splits_df_out


def get_sample_id_vector_filename_from_splits_csv_path(splits_csv_path, split):
    return splits_csv_path.stem + f'.{split}.' + vak.datapipes.frame_classification.constants.SAMPLE_IDS_ARRAY_FILENAME


def get_inds_in_sample_vector_filename_from_splits_csv_path(splits_csv_path, split):
    return splits_csv_path.stem + f'.{split}.' + vak.datapipes.frame_classification.constants.INDS_IN_SAMPLE_ARRAY_FILENAME


def save_vecs_and_make_json_from_csv_paths(
    splits_csv_paths: list[pathlib.Path],
    dry_run=True
):
    splits_path_json_paths = []
    for splits_csv_path in splits_csv_paths:
        logger.info(
            f"Making sample ID vector and inds in sample vector for splits in csv path:\n{splits_csv_path}"
        )
        splits_path_json_dict = {
            'splits_csv_path': str(
                splits_csv_path.relative_to(constants.DATASET_ROOT)
            )
        }

        (split_sample_id_vec_map,
         split_inds_in_sample_vec_map,
         splits_df_out
        ) = sample_vecs_and_splits_df_from_splits_csv_path(
            splits_csv_path
        )
        splits_df_out.to_csv(splits_csv_path)

        splits_path_json_dict['sample_id_vec_path'] = {}
        for split, sample_id_vec in split_sample_id_vec_map.items():
            sample_id_vec_filename = get_sample_id_vector_filename_from_splits_csv_path(splits_csv_path, split)
            logger.info(
                f"Saving sample ID vector: {sample_id_vec_filename}"
            )
            sample_id_vec_path = constants.SAMPLE_ID_VECTORS_DIR / sample_id_vec_filename
            splits_path_json_dict['sample_id_vec_path'][split] = str(sample_id_vec_path.relative_to(
                constants.DATASET_ROOT
            ))
            if not dry_run:
                np.save(
                    sample_id_vec_path,
                    sample_id_vec
                )

        splits_path_json_dict['inds_in_sample_vec_path'] = {}
        for split, inds_in_sample_vec in split_inds_in_sample_vec_map.items():
            inds_in_sample_vec_filename = get_inds_in_sample_vector_filename_from_splits_csv_path(splits_csv_path, split)
            logger.info(
                f"Saving inds in sample vector: {inds_in_sample_vec_filename}"
            )
            inds_in_sample_vec_path = constants.INDS_IN_SAMPLE_VECTORS_DIR / inds_in_sample_vec_filename
            splits_path_json_dict['inds_in_sample_vec_path'][split] = str(inds_in_sample_vec_path.relative_to(
                constants.DATASET_ROOT
            ))
            if not dry_run:
                np.save(
                    inds_in_sample_vec_path,
                    inds_in_sample_vec,
                )
        splits_path_json_filename = splits_csv_path.stem + ".json"
        logger.info(
            f"Saving splits path json: {splits_path_json_filename}"
        )
        splits_path_json_path = constants.SPLITS_JSONS_DIR / splits_path_json_filename
        splits_path_json_paths.append(
            splits_path_json_path
        )
        if not dry_run:
            with splits_path_json_path.open('w') as fp:
                json.dump(splits_path_json_dict, fp, indent=4)
    return splits_path_json_paths


@dataclasses.dataclass
class TrainingReplicateMetadata:
    biosound_group: str
    id: str | None
    frame_dur: float
    unit: str
    data_source: str | None
    train_dur: float
    replicate_num: int
    splits_json_path: str


def metadata_from_splits_json_path(splits_json_path: pathlib.Path) -> TrainingReplicateMetadata:
    try:
        # Human-Speech doesn't have ID or data source in filename
        # so it will raise a ValueError
        name = splits_json_path.name
        (biosound_group,
        id_,
        frame_dur_1st_half,
        frame_dur_2nd_half,
        unit,
        data_source,
        train_dur_1st_half,
        train_dur_2nd_half,
        replicate_num,
        _, _
        ) = name.split('.')
    except ValueError:
        name = splits_json_path.name
        (biosound_group,
        frame_dur_1st_half,
        frame_dur_2nd_half,
        unit,
        train_dur_1st_half,
        train_dur_2nd_half,
        replicate_num,
        _, _
        ) = name.split('.')
        id_ = None
        data_source = None
    if id_ is not None:
        id_ = id_.split('-')[-1]
    frame_dur = float(
        frame_dur_1st_half.split('-')[-1] + '.' + frame_dur_2nd_half.split('-')[0]
    )
    train_dur = float(
        train_dur_1st_half.split('-')[-1] + '.' + train_dur_2nd_half.split('-')[0]
    )
    replicate_num = int(
            replicate_num.split('-')[-1]
    )
    return TrainingReplicateMetadata(
        biosound_group,
        id_,
        frame_dur,
        unit,
        data_source,
        train_dur,
        replicate_num,
        str(splits_json_path.relative_to(constants.DATASET_ROOT))
    )


@dataclasses.dataclass
class MakeSplitsParams:
    biosound_group: str
    unit: str
    frame_dur_str: str | list[str]
    total_train_dur: float
    val_dur: float
    test_dur: float
    train_subset_dur_id_only: float
    num_replicates: int
    make_leave_one_id_out_splits: bool = True


BIOSOUND_GROUP_MAKE_SPLITS_PARAMS_MAP = {
    "Bengalese-Finch-Song": MakeSplitsParams(
        biosound_group='Bengalese-Finch-Song',
        unit='syllable',
        frame_dur_str='1.0',
        total_train_dur=900.,
        val_dur=80.,
        test_dur=400.,
        train_subset_dur_id_only=600.,
        num_replicates=3,
    ),
    "Canary-Song": MakeSplitsParams(
        biosound_group='Canary-Song',
        unit='syllable',
        frame_dur_str='2.7',
        total_train_dur=10000.,
        val_dur=250.,
        test_dur=5000.,
        train_subset_dur_id_only=3600.,
        num_replicates=3,
    ),
    "Mouse-Pup-Call": MakeSplitsParams(
        biosound_group='Mouse-Pup-Call',
        unit='call',
        frame_dur_str='1.5',
        total_train_dur=2100.,
        val_dur=50.,
        test_dur=750.,
        train_subset_dur_id_only=1500.,
        num_replicates=3,
    ),
    "Zebra-Finch-Song": MakeSplitsParams(
        biosound_group='Zebra-Finch-Song',
        unit='syllable',
        frame_dur_str='0.5',
        total_train_dur=130.,
        val_dur=10.,
        test_dur=40.,
        train_subset_dur_id_only=100.,
        num_replicates=3,
        make_leave_one_id_out_splits=False,
    ),
    "Human-Speech": MakeSplitsParams(
        biosound_group='Human-Speech',
        unit='phoneme',
        frame_dur_str='10.0',
        total_train_dur=18000.,
        val_dur=500.,
        test_dur=500.,
        train_subset_dur_id_only=16000.,
        num_replicates=3,
        make_leave_one_id_out_splits=False,
    ),
}


def make_splits_all(
    biosound_groups: list[str], dry_run=True
) -> None:
    metadata_for_json = []
    for biosound_group in biosound_groups:
        logger.info(
            f"Making splits for biosound group: {biosound_group}"
        )
        params = BIOSOUND_GROUP_MAKE_SPLITS_PARAMS_MAP[biosound_group]
        if biosound_group == 'Human-Speech':
            replicate_csv_paths = make_splits_timit(
                total_train_dur=params.total_train_dur,
                val_dur=params.val_dur,
                train_subset_dur=params.train_subset_dur_id_only,
                num_replicates=params.num_replicates,
                dry_run=dry_run,
            )
        else:
            params = dataclasses.asdict(params)
            params['dry_run'] = dry_run
            replicate_csv_paths = make_splits_per_id(**params)
        splits_path_json_paths = save_vecs_and_make_json_from_csv_paths(
            replicate_csv_paths, dry_run
        )
        for splits_path_json_path in splits_path_json_paths:
            metadata = metadata_from_splits_json_path(
                splits_path_json_path
            )
            metadata_for_json.append(
                dataclasses.asdict(metadata)
            )
    with (constants.TRAINING_REPLICATE_METADATA_JSON_PATH).open('w') as fp:
        json.dump(metadata_for_json, fp, indent=4)
