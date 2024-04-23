"""
Helper functinos for stage 1 of making BioSoundSegBench dataset:
1. copy raw audio
2. copy annotations if they are in the SimpleSeq format already;
convert annotations if they are in another format;
generate placeholder annotations if we need to.
"""
import logging
import pathlib
import shutil
from collections import defaultdict
from dataclasses import dataclass

import crowsetta
import dask.delayed
import numpy as np
import pandas as pd
import tqdm
import vocalpy as voc
from dask.diagnostics import ProgressBar

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
        pbar = tqdm.tqdm(
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
                simple_seq.to_file(simple_seq_path)

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
    """Copy audio files and convert annotation files canary song dataset from
    TweetyNet paper.

    We convert the annotation files from the 'generic-seq' format
    (a single annotation file contains annotations for multiple audio files)
    to the 'simple-seq' format (a single annotation file contains only
    annotations for a single audio file)"""
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
    # We convert from `"generic-seq"` (single file for all annotations)
    # to `"simple-seq"` (one annotation file per one audio file)
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
            f"Removing audio files without an annotation file for bird ID {bird_id} "
            "from Birdsong-Recognition dataset."
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


@dataclass
class JourjineEtAl2023Sample:
    this_file_segs_df: pd.DataFrame
    wav_path: pathlib.Path
    species_id_dst: pathlib.Path
    source_file: str


SEED = 42
RNG = np.random.default_rng(seed=SEED)


def make_clips_from_jourjine_et_al_2023(
    sample: JourjineEtAl2023Sample,
    clip_dur: float = 10.,  # seconds
    n_clips_per_file: int = 5,
    min_segs_for_random_clips: int = 200,
):
    simple_seq = crowsetta.formats.seq.SimpleSeq(
        onsets_s=sample.this_file_segs_df.start_seconds.values,
        offsets_s=sample.this_file_segs_df.stop_seconds.values,
        labels=np.array(['v'] * sample.this_file_segs_df.stop_seconds.values.shape[-1]),
        annot_path='dummy',
    )
    sound = voc.Audio.read(sample.wav_path)
    dur = sound.data.shape[-1] / sound.samplerate
    clip_times = np.arange(
        0., dur, clip_dur
    )
    new_clip_times = []
    for clip_time in clip_times:
        in_segment = np.logical_and(
            clip_time > simple_seq.onsets_s,
            clip_time < simple_seq.offsets_s
        )
        if np.any(in_segment):
            # move clip time to the middle of the preceding silent gap
            segment_ind = np.nonzero(in_segment)[0]
            if len(segment_ind) > 1:
                raise ValueError(
                    f"clip time detected in multiple segments: {segment_ind}"
                )
            segment_ind = segment_ind[0]
            prev_silent_gap_off = simple_seq.onsets_s[segment_ind]
            if segment_ind > 0:
                prev_silent_gap_on = simple_seq.offsets_s[segment_ind]
            else:
                prev_silent_gap_on = 0.
            new_clip_time = prev_silent_gap_on + (prev_silent_gap_off - prev_silent_gap_on) / 2
            new_clip_times.append(new_clip_time)
        else:
            # no need to move clip time
            new_clip_times.append(clip_time)
    new_clip_times.append(dur)
    new_clip_times = np.array(new_clip_times)

    if len(sample.this_file_segs_df) < min_segs_for_random_clips:
        # not enough segments to take random clips, sort by number of segments per clip
        clip_start_times = new_clip_times[:-1]
        clip_stop_times = new_clip_times[1:]
        n_segs_per_clip = []
        for clip_ind, (start, stop) in enumerate(zip(clip_start_times, clip_stop_times)):
            n_segs_per_clip.append(
                (clip_ind,
                    len(sample.this_file_segs_df[
                        (sample.this_file_segs_df.start_seconds > start) &
                        (sample.this_file_segs_df.stop_seconds < stop)
                        ]
                )
            ))
        clip_inds_sorted_by_n_segs = sorted(n_segs_per_clip, key=lambda x: x[1], reverse=True)
        # the result is that we take the clips with the most segments first,
        # but we still take clips with zero segments if there are any in the clips we get
        # using `n_clips_per_file`
        clip_inds = np.array(
            [clip_ind_n_segs_tup[0] for clip_ind_n_segs_tup in clip_inds_sorted_by_n_segs[:n_clips_per_file]]
        )
    else:
        clip_inds = np.sort(
            RNG.integers(new_clip_times.size - 1, size=n_clips_per_file)
        )

    clip_starts = new_clip_times[:-1][clip_inds]
    clip_stops = new_clip_times[1:][clip_inds]

    records = []
    for clip_num, (start, stop) in enumerate(
        zip(clip_starts, clip_stops)
    ):
        records.append(
            {
                'source_file': sample.source_file,
                'clip_num': clip_num,
                'start_time': start,
                'stop_time': stop,
            }
        )
        start_ind = int(start * sound.samplerate)
        stop_ind = int(stop * sound.samplerate)
        clip_sound = voc.Audio(
            data=sound.data[..., start_ind:stop_ind + 1],
            samplerate=sound.samplerate,
        )
        clip_wav_path = sample.species_id_dst / f"{sample.wav_path.stem}.clip-{clip_num}.wav"
        clip_sound.write(clip_wav_path)
        clip_onsets_s = simple_seq.onsets_s[
            np.logical_and(
                simple_seq.onsets_s > start,
                simple_seq.onsets_s < stop,
            )
        ] - start
        clip_offsets_s = simple_seq.offsets_s[
            np.logical_and(
                simple_seq.offsets_s > start,
                simple_seq.offsets_s < stop,
            )
        ] - start
        clip_simple_seq = crowsetta.formats.seq.SimpleSeq(
            onsets_s=clip_onsets_s,
            offsets_s=clip_offsets_s,
            labels=np.array(['v'] * clip_offsets_s.size),
            annot_path='dummy',
        )
        clip_csv_path = clip_wav_path.parent / (
            clip_wav_path.name + ".call.csv"
        )
        clip_simple_seq.to_file(clip_csv_path)
    all_clips_df = pd.DataFrame.from_records(records)
    all_clips_csv_path = sample.species_id_dst / (sample.wav_path.name + ".clip-times.csv")
    all_clips_df.to_csv(all_clips_csv_path)


def clip_wav_generate_annot_jourjine_et_al_2023(dry_run=True):
    """Make clips from wav files and generate annotation files
    for Jourjine et al. 2023 dataset

    For this dataset, we sub-sample the audio files,
    which are almost all 10 minutes long.
    We grab 5 10-second clips at random from each audio file,
    making sure to clip within silent intervals
    between vocalizations that are longer than a
    minimum duration (0.3 s).

    Additionally, we start with the segmentation files,
    to make sure we have segmentation for each audio file,
    and since we generate annotations from the segmentation on the fly.
    We generate the annotations by treating this as a binary classification task:
    segments (above threshold) are labeled "vocalization",
    all other periods are labeled "no vocalization".
    So we just label all the detected segments as "v" (for "vocalization")"""
    segs_csv_path = constants.JOURJINE_ET_AL_2023_SEGS_DIR / "all_development_vocs_with_start_stop_times.csv"
    segs_df = pd.read_csv(segs_csv_path)

    todo = []
    pbar = tqdm.tqdm(segs_df.source_file.unique())
    for source_file in pbar:
        pbar.set_description(
            f"Processing source file: {source_file}"
        )
        this_file_segs_df = segs_df[
            segs_df.source_file == source_file
        ]
        if len(this_file_segs_df) < 1:
            pbar.set_description(
                f"No segments in file, skipping: {source_file}"
            )
            continue
        species_id = source_file.split('_')[0]
        wav_path = constants.JOURJINE_ET_AL_2023_DATA / f"development{species_id}" / f"development_{species_id}" / source_file
        if wav_path.exists():
            species_id_dst = constants.MOUSE_PUP_CALL_DATA_DST / f"jourjine-et-al-2023-{species_id}"
            if not species_id_dst.exists():
                if not dry_run:
                    species_id_dst.mkdir(exist_ok=True)
            sample = JourjineEtAl2023Sample(
                this_file_segs_df,
                wav_path,
                species_id_dst,
                source_file
            )
            # un-parallelizing for now to catch error
            # make_clips_from_jourjine_et_al_2023(sample)
            todo.append(
                dask.delayed(make_clips_from_jourjine_et_al_2023)(sample)
            )
    if not dry_run:
        with ProgressBar():
            dask.compute(*todo)


def copy_wav_convert_annot_steinfath_et_al_2021_zebra_finch(dry_run=True):
    """
    """
    logger.info(
        f"Copying audio and annotation files for bird ID {constants.STEINFATH_ET_AL_2021_ZB_BIRD_ID} from Steinfath et al. 2021 dataset"
    )
    # only one zebra finch in the dataset,
    # because it's so rare to have zebra finch data </s>
    bird_id_dir = constants.STEINFATH_ET_AL_2021_RAW_DATA
    assert bird_id_dir.exists(), f"Couldn't find bird_id_dir: {bird_id_dir}"

    bird_id_dst = constants.ZB_DATA_DST / f"Steinfath-et-al-2021-{constants.STEINFATH_ET_AL_2021_ZB_BIRD_ID}"
    bird_id_dst.mkdir(exist_ok=True)

    # bfsongrepo is split up by days; we throw away that split here
    logger.info(
        f"Copying audio and annotation files from day: {bird_id_dir.name}"
    )
    wav_paths = sorted(bird_id_dir.glob("*wav"))
    logger.info(
        f"Found {len(wav_paths)} audio files."
    )
    csv_paths = sorted(bird_id_dir.glob("*csv"))
    logger.info(
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
    logger.info(
        "After filtering audio files by whether or not they have a valid annotation file, "
        f"there are {len(wav_paths_filtered)} audio files and {len(csv_paths_filtered)} annotation files."
    )
    pbar = tqdm.tqdm(
        zip(wav_paths_filtered, csv_paths_filtered)
    )
    for wav_path, csv_path in pbar:
        pbar.set_description(
            f"Copying wav + converting annotation for: {wav_path.name}"
        )
        if not dry_run:
            shutil.copy(wav_path, bird_id_dst)
        # NOTE we rename to include the unit in the annotation file name
        simple_seq = crowsetta.formats.seq.SimpleSeq.from_file(
            csv_path,
            columns_map={
                'start_seconds': 'onset_s', 'stop_seconds': 'offset_s', 'name': 'label'
            },
        )
        csv_path_dst = bird_id_dst / (wav_path.name + '.syllable.csv')
        if not dry_run:
            simple_seq.to_file(csv_path_dst)


def copy_audio_convert_annotations_timit(dry_run=True):
    """"""
    logger.info(
        "Processing data from TIMIT, NLTK Sample"
    )
    TIMIT_NLTK_DATA_DIRS = [
        subdir for subdir in constants.TIMIT_NLTK_RAW.iterdir()
        if subdir.is_dir() and subdir.name.startswith('dr')
    ]

    # keys will be dialect region,
    # values will be list of speaker IDs
    nltk_dr_spkr_map = defaultdict(list)

    for data_dir in TIMIT_NLTK_DATA_DIRS:
        dir_name_upper = data_dir.name.upper()
        dialect_region, speaker_id = dir_name_upper.split('-')
        logger.info(
            f"Processing data for dialect region '{dialect_region}', "
            f"speaker ID: {speaker_id}"
        )
        nltk_dr_spkr_map[dialect_region].append(speaker_id)

        wav_paths = sorted(data_dir.glob('*wav'))
        wrd_paths = sorted(data_dir.glob('*wrd'))
        phn_paths = sorted(data_dir.glob('*phn'))
        if not len(wav_paths) == len(wrd_paths) == len(phn_paths):
            raise ValueError(
                f"len(wav_paths)={len(wav_paths)} != len(wrd_paths)={len(wrd_paths)} "
                f"!= len(phn_paths)={len(phn_paths)}"
            )

        dst = constants.SPEECH_DATA_DST / f"TIMIT-NLTK-{dialect_region}-{speaker_id}"
        dst.mkdir(exist_ok=True)
        pbar = tqdm.tqdm(zip(
            wav_paths, wrd_paths, phn_paths
        ))
        for wav_path, wrd_path, phn_path in pbar:
            pbar.set_description(
                f"Processing audio and annotations for: {wav_path.name}"
            )
            if not dry_run:
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

    TIMIT_FULL_CORPUS_DATA_DIRS = sorted(constants.TIMIT_FULL_CORPUS_RAW.glob("T*/DR*/*/"))

    logger.info(
        "Processing data from TIMIT, full corpus"
    )

    n_skipped = 0
    for data_dir in TIMIT_FULL_CORPUS_DATA_DIRS:
        speaker_id = data_dir.name
        dialect_region = data_dir.parents[0].name
        if speaker_id in nltk_dr_spkr_map[dialect_region]:
            logger.info(
                f"Skipping speaker {speaker_id} because they are in NLTK TIMIT corpus sample"
            )
            continue
        logger.info(
            f"Processing data for dialect region '{dialect_region}', "
            f"speaker ID: {speaker_id}"
        )

        wav_paths = sorted(data_dir.glob('*wav'))
        wrd_paths = sorted(data_dir.glob('*WRD'))
        phn_paths = sorted(data_dir.glob('*PHN'))
        if not len(wav_paths) == len(wrd_paths) == len(phn_paths):
            raise ValueError(
                f"len(wav_paths)={len(wav_paths)} != len(wrd_paths)={len(wrd_paths)} "
                f"!= len(phn_paths)={len(phn_paths)}"
            )

        dst = constants.HUMAN_SPEECH_WE_CANT_SHARE / f"TIMIT-full-corpus-{dialect_region}-{speaker_id}"
        dst.mkdir(exist_ok=True)
        pbar = tqdm.tqdm(zip(
            wav_paths, wrd_paths, phn_paths
        ))
        for wav_path, wrd_path, phn_path in pbar:
            pbar.set_description(
                f"Processing audio and annotations for: {wav_path.name}"
            )
            wav_path_dst = dst / wav_path.stem.replace('.WAV', '.wav')
            if not dry_run:
                shutil.copy(wav_path, wav_path_dst)

            phn_seq = crowsetta.formats.seq.Timit.from_file(phn_path).to_seq()
            phn_simpleseq = crowsetta.formats.seq.SimpleSeq(
                onsets_s=phn_seq.onsets_s,
                offsets_s=phn_seq.offsets_s,
                labels=phn_seq.labels,
                annot_path='dummy',
            )
            phn_csv_dst = dst / f"{wav_path_dst.name}.phoneme.csv"
            if not dry_run:
                phn_simpleseq.to_file(phn_csv_dst)

            wrd_seq = crowsetta.formats.seq.Timit.from_file(wrd_path).to_seq()
            wrd_simpleseq = crowsetta.formats.seq.SimpleSeq(
                onsets_s=wrd_seq.onsets_s,
                offsets_s=wrd_seq.offsets_s,
                labels=wrd_seq.labels,
                annot_path='dummy',
            )
            wrd_csv_dst = dst / f"{wav_path_dst.name}.word.csv"
            if not dry_run:
                wrd_simpleseq.to_file(wrd_csv_dst)


def copy_audio_copy_make_annot_all(biosound_groups, dry_run=True):
    if "Bengalese-Finch-Song" in biosound_groups:
        # ---- Bengalese finch song
        logger.info(
            f"Getting audio and annotations for Bengalese finch song."
        )
        copy_wav_csv_files_bfsongrepo(dry_run)
        copy_wav_convert_annot_birdsongrec(dry_run)

    if "Canary-Song" in biosound_groups:
        # ---- Canary song
        logger.info(
            f"Getting audio and annotations for canary song."
        )
        copy_wav_convert_annot_tweetynet_canary(dry_run)

    if "Mouse-Pup-Call" in biosound_groups:
        # ---- Mouse pup calls
        logger.info(
            f"Getting audio and annotations for mouse pup song."
        )
        clip_wav_generate_annot_jourjine_et_al_2023(dry_run)

    # ---- Zebra finch song
    if "Zebra-Finch-Song" in biosound_groups:
        logger.info(
            f"Getting audio and annotations for zebra finch song."
        )
        copy_wav_convert_annot_steinfath_et_al_2021_zebra_finch(dry_run)

    # ---- Human speech
    if "Human-Speech" in biosound_groups:
        logger.info(
            f"Getting audio and annotations for human speech."
        )
        copy_audio_convert_annotations_timit(dry_run)
