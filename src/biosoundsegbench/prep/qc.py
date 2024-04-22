import json
import logging
import shutil

import crowsetta
import numpy as np
from tqdm.notebook import tqdm
import vocalpy as voc

from . import constants, labels


logger = logging.getLogger(__name__)


SCRIBE = crowsetta.Transcriber(format='simple-seq')


def _qc_annot(data_dir, csv_ext=".syllable.csv"):
    """Helper function for checking annotations are valid"""
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
            logger.info(
                f"File has first onset less than 0: {csv_path.name}"
            )
            first_onset_lt_zero.append(
                (wav_path, csv_path)
            )
            # `continue` so we don't add same (wav, csv) tuple twice
            # and cause an error downstream
            continue
        elif np.any(simpleseq.onsets_s[1:]) < 0.:
            logger.info(
                f"File has onset (other than first) less than 0: {csv_path.name}"
            )
            any_onset_lt_zero.append((wav_path, csv_path))
            continue
        elif np.any(simpleseq.offsets_s) < 0.:
            logger.info(
                f"File has offset less than 0: {csv_path.name}"
            )
            any_offset_lt_zero.append((wav_path, csv_path))
            continue
        else:
            try:
                voc.metrics.segmentation.ir.concat_starts_and_stops(
                    simpleseq.onsets_s, simpleseq.offsets_s
                )
            except:
                logger.info(
                    f"caused error when concatenating starts and stops: {csv_path.name}"
                )
                invalid_starts_stops.append((wav_path, csv_path))

    return first_onset_lt_zero, any_onset_lt_zero, any_offset_lt_zero, invalid_starts_stops


SPECIES_CSV_EXT_MAP = {
    'Bengalese-Finch': ".syllable.csv",
    'Canary': ".syllable.csv",
    'Zebra-Finch': ".syllable.csv",
    'Mouse': ".call.csv",
    # we assume the human speech dataset has valid onset + offset times
}


DATA_DIRS = sorted(constants.DATASET_ROOT.glob(
    "*/*/"
))


def qc_boundary_times(dry_run=True):
    """Do quality control checks on boundary times in annotations"""
    for species, csv_ext in SPECIES_CSV_EXT_MAP.items():
        logger.info(
            f"QCing boundary times in annotations for species '{species}' and csv extension '{csv_ext}'")
        data_dirs = [
            data_dir for data_dir in DATA_DIRS
            if species in data_dir.parents[0].name
        ]
        for data_dir in data_dirs:
            logger.info(f"Data dir name: {data_dir.name}")
            (first_onset_lt_zero,
            any_onset_lt_zero,
            any_offset_lt_zero,
            invalid_starts_stops
            ) = _qc_annot(data_dir, csv_ext)

            logger.info(
                f"\tNum. w/first onset less than zero: {len(first_onset_lt_zero)}\n"
                f"\tNum. w/any onset less than zero: {len(any_onset_lt_zero)}\n"
                f"\tNum. w/any offset less than zero: {len(any_offset_lt_zero)}\n"
                f"\tNum. w/invalid starts + stops: {len(invalid_starts_stops)}\n"
            )

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
                    if not dry_run:
                        remove_dst.mkdir(exist_ok=True)
                    for wav_path, csv_path in wav_csv_tup_list:
                        if not dry_run:
                            shutil.move(wav_path, remove_dst)
                            shutil.move(csv_path, remove_dst)


def qc_labels_in_labelset(dry_run=True):
    species_id_labelset_map = labels.get_labelsets()
    for species, id_labelset_map in species_id_labelset_map.items():
        species_root = constants.DATASET_ROOT / species
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
                    f"Did not find exactly one directory for id '{id_dir}' for species '{species}', "
                    f" instead found: {id_dir}"
                )
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
            logger.info(
                f"Found {len(labels_not_in_labelset)} annotations with labels "
                f"not in labelset for ID: {id}"
            )
            not_in_labelset_dst = id_dir / 'labels-not-in-labelset'
            if not dry_run:
                not_in_labelset_dst.mkdir(exist_ok=True)
            for wav_path, csv_path in labels_not_in_labelset:
                if not dry_run:
                    shutil.move(wav_path, not_in_labelset_dst)
                    shutil.move(csv_path, not_in_labelset_dst)


def do_qc(dry_run=True):
    """Do quality control checks after copying audio files,
    and copying/converting/generating annotation files."""
    qc_boundary_times(dry_run)
    qc_labels_in_labelset(dry_run)
