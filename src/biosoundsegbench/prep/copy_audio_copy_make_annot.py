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

import buckeye
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
    segment_label: str
    source_file: str
    clip_dur: float = 10.
    n_clips_per_file: int = 10


SEED = 42
RNG = np.random.default_rng(seed=SEED)


def make_clips_from_jourjine_et_al_2023(
    sample: JourjineEtAl2023Sample,
    boundary_pad = 0.003,  # seconds
    min_segs_for_random_clips: int = 750,
):
    """Make "clips" from Jourjine et al. 2023 dataset.

    For a "sample" in the dataset (one audio file from one ID),
    we generate sone number of clips determined by the `clip_dur`
    attribute of the ``sample``.
    """
    simple_seq = crowsetta.formats.seq.SimpleSeq(
        onsets_s=sample.this_file_segs_df.start_seconds.values,
        offsets_s=sample.this_file_segs_df.stop_seconds.values,
        labels=np.array(['v'] * sample.this_file_segs_df.stop_seconds.values.shape[-1]),
        annot_path='dummy',
    )
    sound = voc.Sound.read(sample.wav_path)
    dur = sound.data.shape[-1] / sound.samplerate
    # find times to clip, using the specified clip duration
    clip_times = np.arange(
        0., sound.duration, sample.clip_dur
    )

    # we fix the clip times so they do not occur within a segment
    clip_times_not_in_segments = []
    for clip_time in clip_times:
        in_segment = np.logical_and(
            # we "pad" the boundaries here and use lte/gte
            # to avoid edge case where clip_time is literally
            # equal to the boundary time
            clip_time >= simple_seq.onsets_s - boundary_pad,
            clip_time <= simple_seq.offsets_s + boundary_pad
        )
        if np.any(in_segment):
            # move clip time to the middle of the preceding silent gap
            segment_ind = np.nonzero(in_segment)[0]
            if len(segment_ind) > 1:
                raise ValueError(
                    f"clip time detected in multiple segments: {segment_ind}"
                )
            segment_ind = segment_ind[0]
            prev_silent_gap_offset = simple_seq.onsets_s[segment_ind]
            if segment_ind > 0:
                prev_silent_gap_onset = simple_seq.offsets_s[segment_ind - 1]
            else:
                prev_silent_gap_onset = 0.
            new_clip_time = prev_silent_gap_onset + (prev_silent_gap_offset - prev_silent_gap_onset) / 2
            clip_times_not_in_segments.append(new_clip_time)
        else:
            # no need to move clip time
            clip_times_not_in_segments.append(clip_time)
    clip_times_not_in_segments.append(sound.duration)
    clip_times_not_in_segments = np.array(clip_times_not_in_segments)

    # we do some careful book-keeping to make sure that we only start clips
    # exactly where frames would start for the STFT used to compute
    # the signal energy that we then segment.
    # The goal is to avoid changing the result of the segmenting algorithm
    # because of where we clip the audio
    hop_length = voc.segment.JOURJINEETAL2023.nperseg - voc.segment.JOURJINEETAL2023.noverlap
    # we say that it is valid to clip at samples where a frame would start for the STFT;
    # again, the goal is to avoid shifting the frames in such a way that we change the segmentation
    valid_clip_sample_inds = np.arange(0, sound.data.shape[-1], hop_length)
    valid_clip_times = valid_clip_sample_inds / sound.samplerate

    # finally we go through clip times adjusted to not be in a segment,
    # and for each we find the nearest valid clip time
    # and the corresponding index
    # Because the hop length <<< number of samples in the audio
    # this is often a very small adjustment, ~a few milliseconds at most.
    # We use `clip_inds_to_use` to clip the sound itself,
    # and `clip_times_to_use` to make the new annotations
    clip_inds_to_use = []
    clip_times_to_use = []
    for clip_time in clip_times_not_in_segments:
        ind_of_nearest_valid_clip_time = np.argmin(np.abs(valid_clip_times - clip_time))

        clip_inds_to_use.append(
            valid_clip_sample_inds[ind_of_nearest_valid_clip_time]
        )
        clip_times_to_use.append(
            valid_clip_times[ind_of_nearest_valid_clip_time]
        )

    clip_inds_to_use = np.array(clip_inds_to_use)
    clip_times_to_use = np.array(clip_times_to_use)

    if len(sample.this_file_segs_df) < min_segs_for_random_clips:
        # not enough segments in this file to take random clips,
        # so sort clips by number of segments per clip
        clip_start_times = clip_times_to_use[:-1]
        clip_stop_times = clip_times_to_use[1:]
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
            [clip_ind_n_segs_tup[0]
            for clip_ind_n_segs_tup in clip_inds_sorted_by_n_segs[:sample.n_clips_per_file]
            ]
        )
    else:
        clip_inds = np.sort(
            # we use `choice` with arange to guarantee we sample without replacement
            # (as opposed to integers)
            RNG.choice(np.arange(clip_times_to_use.size - 1), size=sample.n_clips_per_file)
        )

    clip_start_times = clip_times_to_use[:-1][clip_inds]
    clip_stop_times = clip_times_to_use[1:][clip_inds]
    clip_start_inds = clip_inds_to_use[:-1][clip_inds]
    clip_stop_inds = clip_inds_to_use[1:][clip_inds]

    if not len(set(
        [len(clip_start_times), len(clip_stop_times), len(clip_start_inds), len(clip_stop_inds)]
    )) == 1:
        raise ValueError(
            "Clip times and inds arrays did not all have same length. "
            f"len(clip_start_times)={len(clip_start_times)}, "
            f"len(clip_stop_times)={len(clip_stop_times)}, "
            f"len(clip_start_inds)={len(clip_start_inds)}, "
            f"len(clip_stop_inds)={len(clip_stop_inds)}."
        )

    records = []
    for clip_num, (start_time, stop_time, start_ind, stop_ind) in enumerate(
        zip(clip_start_times, clip_stop_times, clip_start_inds, clip_stop_inds)
    ):
        records.append(
            {
                'source_file': sample.source_file,
                'clip_num': clip_num,
                'start_time': start_time,
                'stop_time': stop_time,
                'start_ind': start_ind,
                'stop_ind': stop_ind,
            }
        )
        clip_sound = voc.Sound(
            data=sound.data[..., start_ind:stop_ind + 1],
            samplerate=sound.samplerate,
        )
        clip_wav_path = sample.species_id_dst / f"{sample.wav_path.stem}.clip-{clip_num}.wav"
        clip_sound.write(clip_wav_path)

        # N.B.: we *re-segment* because in spite of careful book-keeping above,
        # the ava segmentation algorithm can give us slightly different segemnt boundaries
        # for these clips, relative to the boundaries we would get if we segmented
        # the entire sound file.
        # So our ground truth becomes the re-segmented clip.
        # This is in keeping with the idea that Jourjine et al. 2023 apply the segmenting
        # algorithm *without* doing any further manual clean-up
        clip_segments = voc.segment.ava(
            clip_sound,
            **voc.segment.JOURJINEETAL2023
        )
        clip_simple_seq = crowsetta.formats.seq.SimpleSeq(
            onsets_s=clip_segments.start_times,
            offsets_s=clip_segments.stop_times,
            labels=np.array([sample.segment_label] * clip_segments.start_times.size),
            annot_path='dummy',
        )
        clip_csv_path = clip_wav_path.parent / (
            clip_wav_path.name + ".call.csv"
        )
        clip_simple_seq.to_file(clip_csv_path)
    all_clips_df = pd.DataFrame.from_records(records)
    all_clips_csv_path = sample.species_id_dst / (sample.wav_path.name + ".clip-times.csv")
    all_clips_df.to_csv(all_clips_csv_path)


def clip_wav_generate_annot_jourjine_et_al_2023(max_clips_per_species=300,
                                                clip_dur: float = 10.,  # seconds
                                                n_clips_per_file: int = 10,
                                                dry_run=True):
    """Make clips from wav files and generate annotation files
    for Jourjine et al. 2023 mouse pup call dataset

    For this dataset, we sub-sample the audio files,
    which are almost all 10 minutes long.
    We grab 10 10-second clips at random from each audio file,
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

    n_clips_per_species = defaultdict(int)
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
        if n_clips_per_species[species_id] >= max_clips_per_species:
            # don't make anymore clips from this species
            continue
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
                segment_label=species_id,
                source_file=source_file,
                clip_dur=clip_dur,
                n_clips_per_file=n_clips_per_file,
            )
            todo.append(
                dask.delayed(make_clips_from_jourjine_et_al_2023)(sample)
            )
            n_clips_per_species[species_id] += n_clips_per_file
    if not dry_run:
        with ProgressBar():
            dask.compute(*todo)


def copy_wav_convert_annot_steinfath_et_al_2021_zebra_finch(dry_run=True):
    """Copy wav audio files and convert csv annotation files
    for zebra finch song dataset from Steinfath et al. 2021
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


class ClipPhones:
    """Class to represent a set of consecutive phones 
    that will go into a clip from the Buckeye corpus."""
    def __init__(self, phones=None):
        if phones is None:
            self.phones = []
        else:
            if not isinstance(phones, list) or not all([isinstance(phone, buckeye.buckeye.Phone) for phone in phones]):
                raise ValueError("`phones` was not a `list` of `buckeye.Phone` instances")
            self.phones = phones

    def __repr__(self):
        return f"Clip({self.phones})"
    
    def append(self, phone):
        if not isinstance(phone, buckeye.buckeye.Phone):
            raise TypeError(
                f"Clip.add expected `phone` instance but got: {type(phone)}"
            )
        self.phones.append(phone)

    def __eq__(self, other):
        if not isinstance(other, ClipPhones):
            return False
        return all([
            self_phone == other_phone
            for self_phone, other_phone in zip(self.phones, other.phones)
        ])

    def __getitem__(self, index):
        return self.phones[index]

    def __len__(self):
        return len(self.phones)
    
    @property
    def dur(self):
        return sum(
            [phone.dur for phone in self.phones]
        )


# the constant below was obtained with the following code run on clips + annotations *without* any filtering
# of "rare" phonemes
# from collections import Counter

# talker_dirs = sorted(
#     biosoundsegbench.prep.constants.HUMAN_SPEECH_WE_CANT_SHARE.glob('Buckeye-corpus-s*')
# )

# talker_count_map = {}
# pbar = tqdm(talker_dirs)
# for talker_dir in pbar:
#     name = talker_dir.name.split('-')[-1]
#     pbar.set_description(f"Counting occurences of classes for {name}")
#     csv_paths = sorted(talker_dir.glob('*.csv'))
#     simple_seqs = [
#         crowsetta.formats.seq.SimpleSeq.from_file(csv_path)
#         for csv_path in csv_paths
#     ]
#     labels = [
#         lbl
#         for simple_seq in simple_seqs
#         for lbl in simple_seq.labels
#     ]
    
#     talker_count_map[name] = Counter(labels)

# MAP_TO_UNK = {}
# MIN_COUNT_FOR_NOT_UNK = 4
# for talker, counter in talker_count_map.items():
#     MAP_TO_UNK[talker] = {}
#     for k, v in counter.items():
#         if v < MIN_COUNT_FOR_NOT_UNK:
#             MAP_TO_UNK[talker][k] = v

# classes that are phonemes uttered by talker,
# but are so uncommon ("rare") that we can't be sure we'll have them in train/val/test splits
# Some of these I think might be one-off transcription/annotation mistakes?
BUCKEYE_MAP_PHONEME_TO_BACKGROUND = {
    's01': {'eng': 1},
    's02': {},
    's03': {},
    's04': {'eng': 3},
    's05': {'eng': 1},
    's06': {'eng': 1},
    's07': {'eng': 3},
    's08': {'eng': 2},
    's09': {'eng': 3},
    's10': {'h': 2, 'eng': 2, 'a': 1},
    's11': {'x': 1, 'q': 1},
    's12': {'eng': 1, 'h': 1},
    's13': {'i': 1},
    's14': {'ih l': 1,
    'eng': 2,
    'ah r': 1,
    'ah l': 2,
    'x': 1,
    'a': 1,
    'ah ix': 1},
    's15': {'h': 2, 'x': 1, 'a': 2, 'i': 1, 'q': 1, 'uw ix': 1},
    's16': {'eng': 1, 'h': 1},
    's17': {'a': 1, 'h': 3, 'i': 1, 'id': 1},
    's18': {'eng': 3},
    's19': {},
    's20': {'h': 2, 'i': 1},
    's21': {},
    's22': {'no': 1, 'h': 1},
    's23': {'zh': 3},
    's24': {'h': 2, 'x': 1, 'oy': 2, 'a': 1},
    's25': {'eng': 2},
    's26': {},
    's27': {},
    's28': {},
    's29': {},
    's30': {},
    's31': {},
    's32': {},
    's33': {'eng': 1},
    's34': {'a': 1, 'eng': 1},
    's35': {'j': 1},
    's36': {},
    's37': {'eng': 1},
    's38': {'eng': 1},
    's39': {},
    's40': {'eng': 1}
}


MAX_CLIP_DUR_BUCKEYE = 5.0  # seconds


# classes that are not phonemes uttered by talker, we map to background classes
MAP_TO_BACKGROUND = [
    '<EXCLUDE>',
    '<exclude-Name>',
    'EXCLUDE',
    'IVER y',
    'IVER-LAUGH',
    '<EXCLUDE-name>',
    'IVER',
    'LAUGH',
    'NOISE',
    'SIL',
    'UNKNOWN',
    'VOCNOISE',
    '{B_TRANS}',
    '{E_TRANS}'
]


def clip_wav_generate_annot_buckeye(
    max_clip_dur: float = MAX_CLIP_DUR_BUCKEYE,
    dry_run: bool = True
) -> None:
    """Make clips and generate annotations from Buckeye corpus
    https://buckeyecorpus.osu.edu/

    Requires the `buckeye` package by Scott Seyfarth
    https://github.com/scjs/buckeye
    """
    for talker in buckeye.corpus(constants.BUCKEYE_ROOT):
        print(
            f"Making clips for talker: {talker.name}"
        )
        dst = constants.HUMAN_SPEECH_WE_CANT_SHARE / f"Buckeye-corpus-{talker.name}"
        print(
            f"Will save clips in: {dst}"
        )
        if not dry_run:
            dst.mkdir(exist_ok=True)

        clip_num = 1
        for track in talker:
            print(
                f"Making clips for track: {track.name}"
            )
            all_clip_phones = []
            current_clip_phones = ClipPhones()
            for phone in track.phones:
                current_clip_phones.append(phone)
                if current_clip_phones.dur >= max_clip_dur:
                    all_clip_phones.append(current_clip_phones)
                    current_clip_phones = ClipPhones()
            if not all_clip_phones[-1] == current_clip_phones:
                if len(current_clip_phones) > 0:
                    all_clip_phones.append(current_clip_phones)

            # now that we have all the phones for each clip from this track, actually make clips + annotation files
            wav_path = pathlib.Path(constants.BUCKEYE_ROOT) / (talker.name + "/" + (track.name + ".wav"))
            sound = voc.Sound.read(wav_path)
            for clip_phone_num, clip_phones in enumerate(tqdm.tqdm(all_clip_phones)):
                # sanitize phones, check for positive duration
                clip_phones.phones = [
                    phone for phone in clip_phones
                    if phone.dur > 0
                ]
                if len(clip_phones) < 1:
                    print(
                        f"Skipping clip {clip_phone_num} of {len(all_clip_phones)}, no phones."
                    )
                    continue
                start = clip_phones[0].beg
                stop = clip_phones[-1].end
                start_sample_ind = int(start * sound.samplerate)
                stop_sample_ind = int(stop * sound.samplerate)
                data = sound.data[..., start_sample_ind:stop_sample_ind + 1]
                if data.shape[-1] < 1:
                    print(
                        f"Skipping clip {clip_phone_num} of {len(all_clip_phones)}, audio file shorter than annotations."
                    )
                    continue
                clip_sound = voc.Sound(
                    data=data,
                    samplerate=sound.samplerate,
                )
                clip_wav_path = dst / f"{track.name}.clip-{clip_num}.wav"
                if not dry_run:
                    clip_sound.write(clip_wav_path)
                start_times = np.array([
                    phone.beg for phone in clip_phones
                ])
                start_time = start_times[0]
                start_times = start_times - start_time
                stop_times = np.array([
                    phone.end for phone in clip_phones
                ]) - start_time
                labels = [
                    # a handful of phonemes (~9) in the dataset have `None` as the label,
                    # at least when loaded with `buckeye`;
                    # these all appear to be silences (usually brief), so we label them <SIL>
                    phone.seg if phone.seg is not None else "SIL"
                    for phone in clip_phones
                ]
                # ** do further clean-up on labels**
                # map any labels that are not phonemes to 'background'
                labels = [
                    'background' if lbl in MAP_TO_BACKGROUND else lbl
                    for lbl in labels
                ]
                # group any nasalized vowels, labeled with n, with the vowels that are not nasalized
                # to reduce class imbalance
                labels = [
                    lbl.replace('n', '') if (lbl.endswith('n') and len(lbl) > 1) else lbl
                    for lbl in labels
                ]
                rare_phones = list(BUCKEYE_MAP_PHONEME_TO_BACKGROUND[talker.name].keys())
                labels = [
                    'background' if lbl in rare_phones else lbl
                    for lbl in labels
                ]
                labels = np.array(labels)

                # now, finally, remove any "background" segments
                # as if they were unlabeled silence, etc.,
                # to match how other datasets are annotated
                not_background = labels != 'background'
                start_times = start_times[not_background]
                stop_times = stop_times[not_background]
                labels = labels[not_background]

                # after removing background we can save annotations
                clip_simple_seq = crowsetta.formats.seq.SimpleSeq(
                    onsets_s=start_times,
                    offsets_s=stop_times,
                    labels=labels,
                    annot_path='dummy',
                )
                clip_csv_path = clip_wav_path.parent / (
                    clip_wav_path.name + ".phoneme.csv"
                )
                if not dry_run:
                    clip_simple_seq.to_file(clip_csv_path)
                clip_num += 1


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
        clip_wav_generate_annot_jourjine_et_al_2023(dry_run=dry_run)

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
        clip_wav_generate_annot_buckeye(dry_run=dry_run)
