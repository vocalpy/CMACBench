import logging
import shutil
from collections import defaultdict

import crowsetta
import numpy as np
import pandera.errors
from tqdm.notebook import tqdm
import vocalpy as voc

from . import constants, labels


logger = logging.getLogger(__name__)


SCRIBE = crowsetta.Transcriber(format='simple-seq')


def qc_boundary_times_id_dir(data_dir, unit, unit_list):
    """Helper function for checking boundary times are valid"""
    if unit not in unit_list:
        raise ValueError(
            f"In `_qc_annot`, specified unit '{unit}', but `unit_list` is: {unit_list}"
        )

    first_onset_lt_zero = []
    any_onset_lt_zero = []
    any_offset_lt_zero = []
    invalid_starts_stops = []

    wav_paths = voc.paths.from_dir(data_dir, '.wav')
    unit_csv_paths = {}
    for unit in unit_list:
        csv_ext = f".{unit}.csv"
        csv_paths = voc.paths.from_dir(data_dir, csv_ext)
        unit_csv_paths[unit] = csv_paths
    n_wav_paths = len(wav_paths)
    if not all(
        [len(csv_paths) == n_wav_paths
         for csv_paths in unit_csv_paths.values()]
    ):
        raise ValueError(
            "Did not find csv paths for all units equal to the number of wav paths. "
            f"Num. wav paths: {n_wav_paths}. Num. csv paths per unit: {[(k, len(v)) for k, v in unit_csv_paths.items()]}"
        )
    id_ = data_dir.name.split('-')[-1]
    n_csv_paths_with_no_segments = 0
    for paths_ind, (wav_path, unit_csv_path) in enumerate(zip(wav_paths, unit_csv_paths[unit])):
        # handle files without segments; we should do this better in crowsetta
        try:
            simpleseq = SCRIBE.from_file(unit_csv_path)
        except pandera.errors.SchemaError:
            import pandas as pd
            df = pd.read_csv(unit_csv_path)
            if len(df) == 0:
                n_csv_paths_with_no_segments += 1
                continue

        if simpleseq.onsets_s[0] < 0.:
            logger.info(
                f"File has first onset less than 0: {unit_csv_path.name}"
            )
            to_append = [wav_path]
            to_append.extend([unit_csv_paths[unit][paths_ind] for unit in unit_list])
            first_onset_lt_zero.append(
                to_append
            )
            # `continue` so we don't add same (wav, csv) tuple twice
            # and cause an error downstream
            continue
        elif np.any(simpleseq.onsets_s[1:]) < 0.:
            logger.info(
                f"File has onset (other than first) less than 0: {unit_csv_path.name}"
            )
            to_append = [wav_path]
            to_append.extend([unit_csv_paths[unit][paths_ind] for unit in unit_list])
            any_onset_lt_zero.append(to_append)
            continue
        elif np.any(simpleseq.offsets_s) < 0.:
            logger.info(
                f"File has offset less than 0: {unit_csv_path.name}"
            )
            to_append = [wav_path]
            to_append.extend([unit_csv_paths[unit][paths_ind] for unit in unit_list])
            any_offset_lt_zero.append(to_append)
            continue
        else:
            try:
                voc.metrics.segmentation.ir.concat_starts_and_stops(
                    simpleseq.onsets_s, simpleseq.offsets_s
                )
            except:
                logger.info(
                    f"caused error when concatenating starts and stops: {unit_csv_path.name}"
                )
                to_append = [wav_path]
                to_append.extend([unit_csv_paths[unit][paths_ind] for unit in unit_list])
                invalid_starts_stops.append(to_append)

    logger.info(
        f"Found {n_csv_paths_with_no_segments} csv paths with no annotated segments, "
        f"{n_csv_paths_with_no_segments / len(csv_paths) * 100:.2f}% of csv paths"
    )
    return first_onset_lt_zero, any_onset_lt_zero, any_offset_lt_zero, invalid_starts_stops


BIOSOUND_GROUP_UNIT_DATA_DIR_MAP = {
    "Bengalese-Finch-Song": [
        ["syllable"], [constants.DATASET_ROOT],
    ],
    "Canary-Song": [
        ["syllable"], [constants.DATASET_ROOT],
    ],
    "Zebra-Finch-Song": [
        ["syllable"], [constants.DATASET_ROOT],
    ],
    "Mouse-Pup-Call": [
        ["call"], [constants.DATASET_ROOT],
    ],
    "Human-Speech": [
        ["phoneme", "word"], [constants.DATASET_ROOT, constants.DATA_WE_CANT_SHARE]
    ]
}


DATA_DIRS = sorted(constants.DATASET_ROOT.glob(
    "*/*/"
))


def qc_boundary_times(biosound_group, dry_run=True):
    """Do quality control checks on boundary times in annotations.

    We check each sample, where a sample is
    one wav audio file and all associated annotation files,
    each annotation file having a different level of annotation.
    For each sample, if an annotation is invalid for *any* unit of annotation,
    then we remove the audio and *all* annotations for *all* units for that sample.
    """
    unit_dir_lists = BIOSOUND_GROUP_UNIT_DATA_DIR_MAP[biosound_group]
    unit_list, dir_list = unit_dir_lists
    logger.info(
        f"QCing boundary times in annotations for BioSound group '{biosound_group}'"
        )

    for root_dir in dir_list:
        logger.info(
            f"QCing boundary times in annotations for root directory '{root_dir.resolve()}'"
            )
        biosound_group_dir = root_dir / biosound_group
        id_dirs = [
            id_dir for id_dir in biosound_group_dir.iterdir()
            if id_dir.is_dir()
        ]
        for id_dir in id_dirs:
            logger.info(f"Data dir name: {id_dir.name}")
            for unit in unit_list:
                logger.info(
                    f"QCing boundary times in annotations for unit '{unit}'"
                    )

                (first_onset_lt_zero,
                any_onset_lt_zero,
                any_offset_lt_zero,
                invalid_starts_stops
                ) = qc_boundary_times_id_dir(id_dir, unit, unit_list)

                logger.info(
                    f"\tNum. w/first onset less than zero: {len(first_onset_lt_zero)}\n"
                    f"\tNum. w/any onset less than zero: {len(any_onset_lt_zero)}\n"
                    f"\tNum. w/any offset less than zero: {len(any_offset_lt_zero)}\n"
                    f"\tNum. w/invalid starts + stops: {len(invalid_starts_stops)}\n"
                )

                for wav_csv_paths_lists, dir_name in zip(
                    (first_onset_lt_zero,
                    any_onset_lt_zero,
                    any_offset_lt_zero,
                    invalid_starts_stops),
                    ('first_onset_lt_zero',
                    'any_onset_lt_zero',
                    'any_offset_lt_zero',
                    'invalid_starts_stops'),
                ):
                    if len(wav_csv_paths_lists) > 0:
                        remove_dst = id_dir / dir_name
                        if not dry_run:
                            remove_dst.mkdir(exist_ok=True)
                        for paths_list in wav_csv_paths_lists:
                            for path in paths_list:
                                if not dry_run:
                                    shutil.move(path, remove_dst)


def qc_labels_in_labelset(biosound_group, unit='syllable', dry_run=True):
    group_unit_id_labelsets_map = labels.get_labelsets()
    id_labelset_map = group_unit_id_labelsets_map[biosound_group][unit]
    species_root = constants.DATASET_ROOT / biosound_group
    species_subdirs = [
        subdir for subdir in species_root.iterdir()
        if subdir.is_dir()
    ]
    for id, labelset in id_labelset_map.items():
        id_dir = [
            id_dir for id_dir in species_subdirs
            if id_dir.name.endswith(id)
        ]
        if not len(id_dir) == 1:
            raise ValueError(
                f"Did not find exactly one directory for id '{id_dir}' for biosound group '{biosound_group}', "
                f" instead found: {id_dir}"
            )
        id_dir = id_dir[0]
        wav_paths = voc.paths.from_dir(id_dir, '.wav')
        csv_paths = voc.paths.from_dir(id_dir, f'.{unit}.csv')
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
        logger.info(
            f"Found {len(labels_not_in_labelset)} annotations with labels "
            f"not in labelset for ID: {id}"
        )
        if len(labels_not_in_labelset) > 0:
            not_in_labelset_dst = id_dir / 'labels-not-in-labelset'
            if not dry_run:
                not_in_labelset_dst.mkdir(exist_ok=True)
            for wav_path, csv_path in labels_not_in_labelset:
                if not dry_run:
                    shutil.move(wav_path, not_in_labelset_dst)
                    shutil.move(csv_path, not_in_labelset_dst)


def qc_labels_in_labelset_human_speech(dry_run=True):
    """For TIMIT, we remove samples from the training set
    that have phoneme labels that are not in the test set.

    We remove the audio as well as both the phoneme and word level annotations.
    This is to be able to make apples-to-apples comparisons between
    models trained on one level versus another.
    """
    group_unit_id_labelsets_map = labels.get_labelsets()
    labelset = group_unit_id_labelsets_map['Human-Speech']['phoneme']['all']
    data_root = constants.HUMAN_SPEECH_WE_CANT_SHARE
    speaker_dirs = [
        subdir for subdir in data_root.iterdir()
        if subdir.is_dir()
    ]
    for speaker_dir in speaker_dirs:
        wav_paths = voc.paths.from_dir(speaker_dir, '.wav')
        phn_paths = voc.paths.from_dir(speaker_dir, '.phoneme.csv')
        wrd_paths = voc.paths.from_dir(speaker_dir, '.word.csv')
        if not len(wav_paths) == len(phn_paths) == len(wrd_paths):
            raise ValueError(
                f"len(wav_paths)={len(wav_paths)} != len(phn_paths)={len(phn_paths)} != len(wrd_paths)={len(wrd_paths)}"
            )
        labels_not_in_labelset = []
        for wav_path, phn_path, wrd_path in zip(wav_paths, phn_paths, wrd_paths):
            simpleseq = crowsetta.formats.seq.SimpleSeq.from_file(phn_path)
            if not all(
                [lbl in labelset for lbl in simpleseq.labels]
            ):
                labels_not_in_labelset.append(
                    (wav_path, phn_path, wrd_path)
                )
        logger.info(
            f"Found {len(labels_not_in_labelset)} annotations with labels "
            f"not in labelset for dir: {speaker_dir.name}"
        )
        if len(labels_not_in_labelset) > 0:
            not_in_labelset_dst = speaker_dir / 'labels-not-in-labelset'
            if not dry_run:
                not_in_labelset_dst.mkdir(exist_ok=True)
            for wav_path, phn_path, wrd_path in labels_not_in_labelset:
                if not dry_run:
                    shutil.move(wav_path, not_in_labelset_dst)
                    shutil.move(phn_path, not_in_labelset_dst)
                    shutil.move(wrd_path, not_in_labelset_dst)


def do_qc(biosound_groups, dry_run=True):
    """Do quality control checks after copying audio files,
    and copying/converting/generating annotation files."""
    if "Bengalese-Finch-Song" in biosound_groups:
        logger.info(
            f"Doing quality control checks for Bengalese finch song."
        )
        qc_boundary_times("Bengalese-Finch-Song", dry_run)
        qc_labels_in_labelset("Bengalese-Finch-Song", "syllable", dry_run)

    if "Canary-Song" in biosound_groups:
        logger.info(
            f"Doing quality control checks for canary song."
        )
        qc_boundary_times("Canary-Song", dry_run)
        qc_labels_in_labelset("Canary-Song", "syllable", dry_run)

    if "Mouse-Pup-Call" in biosound_groups:
        logger.info(
            f"Doing quality control checks for mouse pup calls."
        )
        qc_boundary_times("Mouse-Pup-Call", dry_run)

    if "Zebra-Finch-Song" in biosound_groups:
        logger.info(
            f"Doing quality control checks for Zebra finch song."
        )
        qc_boundary_times("Zebra-Finch-Song", dry_run)
        qc_labels_in_labelset("Zebra-Finch-Song", "syllable", dry_run)

    if "Human-Speech" in biosound_groups:
        logger.info(
            f"Doing quality control checks for human speech."
        )
        qc_boundary_times("Human-Speech", dry_run)
        # we force the training set to only have classes that are in the test set
        qc_labels_in_labelset_human_speech(dry_run)
