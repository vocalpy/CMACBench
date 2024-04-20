"""
Helper functinos for stage 1 of making BioSoundSegBench dataset:
1. copy raw audio
2. copy annotations if they are in the SimpleSeq format already;
convert annotations if they are in another format;
generate placeholder annotations if we need to.
"""
import logging
import shutil

import crowsetta
import numpy as np
import pandas as pd
import tqdm

from . import constants


logger = logging.getLogger(__name__)


def copy_wav_csv_files_bfsongrepo(dry_run=True):
    """Copy wav audio files and copy/rename csv annotation files for Bengalese Finch Song Repository.

    We rename the annotation files to indicate they are annotated with units of 'syllable'.
    """
    for bird_id in constants.BFSONGREPO_BIRD_IDS:
        logger.info(
            f"Copying audio and annotation files for bird ID {bird_id} from BFSongRepo"
        )
        bird_id_dir = constants.BFSONGREPO_RAW_DATA / bird_id
        if not bird_id_dir.exists():
            raise NotADirectoryError(
                f"Couldn't find bird_id_dir: {bird_id_dir}"
            )

        bird_id_dst = constants.BF_DATA_DST / f"bfsongrepo-{bird_id}"
        bird_id_dst.mkdir(exist_ok=True)

        # bfsongrepo is split up by days; we throw away that split here
        day_dirs = [
            subdir for subdir in bird_id_dir.iterdir() if subdir.is_dir()
        ]
        for day_dir in day_dirs:
            logger.info(
                f"Copying audio and annotation files from day: {day_dir.name}"
            )
            wav_paths = sorted(day_dir.glob("*wav"))
            logger.info(
                f"Found {len(wav_paths)} audio files."
            )
            csv_paths = sorted(day_dir.glob("*csv"))
            logger.info(
                f"Found {len(csv_paths)} annotation files."
            )
            wav_paths_filtered = []
            csv_paths_filtered = []
            for wav_path in wav_paths:
                csv_path_that_should_exist = wav_path.parent / (wav_path.name + '.csv')
                if csv_path_that_should_exist in csv_paths:
                    wav_paths_filtered.append(wav_path)
                    csv_paths_filtered.append(csv_path_that_should_exist)
            logger.info(
                "After filtering audio files by whether or not they have an annotation file, "
                f"there are {len(wav_paths_filtered)} audio files and {len(csv_paths_filtered)} annotation files."
            )
            pbar = tqdm.tqdm(
                zip(wav_paths_filtered, csv_paths_filtered)
            )
            for wav_path, csv_path in pbar:
                pbar.set_description(f"Copying: {wav_path.name}")
                if not dry_run:
                    shutil.copy(wav_path, bird_id_dst)
                # NOTE we rename to include the unit in the annotation file name
                csv_path_dst = bird_id_dst / (csv_path.stem + '.syllable.csv')
                pbar.set_description(f"Copying/renaming: {csv_path_dst.name}")
                if not dry_run:
                    shutil.copy(csv_path, csv_path_dst)


YARDEN_SCRIBE = crowsetta.Transcriber(format='yarden')


def copy_wav_convert_annot_birdsongrec(dry_run=True):
    """Copy audio files and convert annotation files for birdsong-recognition dataset

    We convert the annotation files from the TweetyNet paper from the 'yarden' format
    to the 'simple-seq' format
    """
    # validate dirs exist at top so we fail early
    if not constants.BIRDSONGREC_ANNOT_ROOT.exists():
        raise NotADirectoryError(
            f"Couldn't find BIRDSONGREC_ANNOT_ROOT: {constants.BIRDSONGREC_ANNOT_ROOT}"
        )

    for bird_id in constants.BIRDSONGREC_BIRD_IDS:
        logger.info(
            f"Copying audio and annotation files for bird ID {bird_id} from Birdsong-Recognition dataset"
        )
        bird_id_dir = constants.BIRDSONGREC_RAW_DATA / bird_id / "Wave"
        if not bird_id_dir.exists():
            raise NotADirectoryError(
                f"Couldn't find bird_id_dir: {bird_id_dir}"
            )

        bird_id_dst = constants.BF_DATA_DST / f"birdsongrec-{bird_id}"
        bird_id_dst.mkdir(exist_ok=True)

        wav_paths = sorted(bird_id_dir.glob("*wav"))
        logger.info(
            f"Found {len(wav_paths)} audio files."
        )

        pbar = tqdm.tqdm(wav_paths)
        for wav_path in pbar:
            pbar.set_description(f"Copying file: {wav_path.name}")
            if not dry_run:
                shutil.copy(wav_path, bird_id_dst)

    annot_mats = sorted(constants.BIRDSONGREC_ANNOT_ROOT.glob('*annotation.mat'))
    for annot_mat in annot_mats:
        bird_id = annot_mat.name.split('-')[1][:5]
        bird_id_dst = constants.BF_DATA_DST / f"birdsongrec-{bird_id}"
        if not bird_id_dst.exists():
            raise NotADirectoryError(
                f"Couldn't find `bird_id_dst`: {bird_id_dst}"
            )
        yarden = YARDEN_SCRIBE.from_file(annot_mat)
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
            # NOTE: we assume here when naming files that the annotations are in sorted numerical order;
            # I inspected manually and confirmed that this was the case.
            # The `audio_paths` in the annotation are given by the GUI,
            # and are not the original audio file names
            simple_seq_path = annot_path=bird_id_dst / f'{filenum}.wav.syllable.csv'
            pbar.set_description(f"Saving converted annotation: {simple_seq_path.name}")
            if not dry_run:
                simple_seq.to_file()

    for bird_id in constants.BIRDSONGREC_BIRD_IDS:
        logger.info(
            f"Verifying each audio file has an annotation file for bird ID {bird_id} from Birdsong-Recognition dataset."
        )

        bird_id_dst = constants.BF_DATA_DST / f"birdsongrec-{bird_id}"

        wav_paths = sorted(bird_id_dst.glob("*wav"))
        csv_paths = sorted(bird_id_dst.glob("*csv"))
        assert all(
            [
                wav_path.parent / (wav_path.name + ".syllable.csv") in csv_paths
                for wav_path in wav_paths
            ]
        )
        assert len(wav_paths) == len(csv_paths)
        logger.info("Verified.")


GENERIC_SEQ_SCRIBE = crowsetta.Transcriber(format='generic-seq')


def copy_wav_convert_annot_tweetynet_canary(dry_run=True):
    """"""
    for bird_id in constants.TWEETYNET_BIRD_IDS:
        logger.info(
            f"Copying audio files for bird ID {bird_id} from TweetyNet dataset"
        )
        bird_id_dir = constants.TWEETYNET_RAW_DATA / f"{bird_id}_data" / f"{bird_id}_songs"
        if not bird_id_dir.exists():
            raise NotADirectoryError(
                f"Couldn't find bird_id_dir: {bird_id_dir}"
            )

        bird_id_dst = constants.CANARY_DATA_DST / f"tweetynet-{bird_id}"
        bird_id_dst.mkdir(exist_ok=True)

        wav_paths = sorted(bird_id_dir.glob("*wav"))
        logger.info(
            f"Found {len(wav_paths)} audio files."
        )

        pbar = tqdm.tqdm(wav_paths)
        for wav_path in pbar:
            pbar.set_description(f"Copying wav file: {wav_path.name}")
            if not dry_run:
                shutil.copy(wav_path, bird_id_dst)


    # Make annotation files.
    # We convert from `"generic-seq"` (single file for all annotations) to `"simple-seq"` (one annotation file per one audio file)
    for bird_id in constants.TWEETYNET_BIRD_IDS:
        logger.info(
            f"Making annotation files for bird ID {bird_id} from TweetyNet dataset"
        )
        bird_id_dir = constants.TWEETYNET_RAW_DATA / f"{bird_id}_data"
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

        annots = GENERIC_SEQ_SCRIBE.from_file(annot_csv_dst).to_annot()

        bird_id_dst = constants.CANARY_DATA_DST / f"tweetynet-{bird_id}"
        assert bird_id_dst.exists()

        logger.info(
            f"Converting {len(annots)} annotations to simple-seq format."
        )
        pbar = tqdm.tqdm(annots)
        for annot in pbar:
            simple_seq = crowsetta.formats.seq.SimpleSeq(
                onsets_s=annot.seq.onsets_s,
                offsets_s=annot.seq.offsets_s,
                labels=annot.seq.labels,
                annot_path=annot.annot_path,
            )
            annot_path = bird_id_dst / f'{annot.notated_path.name}.syllable.csv'
            pbar.set_description(f"Saving converted annotation: {annot_path.name}")
            if not dry_run:
                simple_seq.to_file(annot_path)

    # For this dataset, not every audio file has an annotation file.
    # We do the clean up step of removing audio files with annotations here.
    for bird_id in constants.TWEETYNET_BIRD_IDS:
        logger.info(
            f"Removing audio files without an annotation file for bird ID {bird_id} from Birdsong-Recognition dataset."
        )

        bird_id_dst = constants.CANARY_DATA_DST / f"tweetynet-{bird_id}"

        wav_paths = sorted(bird_id_dst.glob("*wav"))
        csv_paths = sorted(bird_id_dst.glob("*.syllable.csv"))

        wav_without_annot_csv = [
            wav_path
            for wav_path in wav_paths
            if wav_path.parent / (wav_path.name + ".syllable.csv") not in csv_paths
        ]

        logger.info(
            f"Found {len(wav_paths)} audio files and {len(csv_paths)} annotation files."
            f"Removing {len(wav_without_annot_csv)} files without annotations."
        )
        pbar = tqdm.tqdm(wav_without_annot_csv)
        for wav_path in pbar:
            pbar.set_description(f"Removing audio file without annotation: {wav_path.name}")
            if not dry_run:
                wav_path.unlink()


def copy_audio_copy_make_annot_all(dry_run=True):
    # ---- Bengalese finch song
    logger.info(
        f"Getting audio and annotations for Bengalese finch song."
    )
    copy_wav_csv_files_bfsongrepo(dry_run)
    copy_wav_convert_annot_birdsongrec(dry_run)

    # ---- Canary song
    logger.info(
        f"Getting audio and annotations for canary song."
    )
    copy_wav_convert_annot_tweetynet_canary(dry_run)

    # ---- Zebra finch song
    logger.info(
        f"Getting audio and annotations for canary song."
    )
    copy_wav_convert_annot_tweetynet_canary(dry_run)

    # ---- Mouse pup calls
    logger.info(
        f"Getting audio and annotations for canary song."
    )
    copy_wav_convert_annot_tweetynet_canary(dry_run)

    # ---- Human speech
    logger.info(
        f"Getting audio and annotations for canary song."
    )
    copy_wav_convert_annot_tweetynet_canary(dry_run)
