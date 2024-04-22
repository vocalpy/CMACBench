"""Make inputs and targets for neural network models.

Uses raw data--audio and annotations--to make inputs and targets for neural network models.
"""
import logging
import math
import pathlib
from dataclasses import dataclass

import crowsetta
import dask.delayed
from dask.diagnostics import ProgressBar
import librosa
import numpy as np
import pandas as pd
import scipy.signal
import scipy.interpolate
import tqdm
import vak
import vocalpy as voc
from vak.prep.spectrogram_dataset.spect import spectrogram

from . import constants, labels


logger = logging.getLogger(__name__)


SCRIBE = crowsetta.Transcriber(format='simple-seq')


# ---- helper functions used to make different types of targets -------------------------
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


def frame_labels_multi_to_binary(
    frame_labels, labelmap, bg_class_name: str = 'unlabeled'
):
    """Converts vector of frame labels with multiple classes
    to a vector for binary classification."""
    bg_class_int = labelmap[bg_class_name]
    return (frame_labels != bg_class_int).astype(int)


# ---- helper functions to get filenames, for consistency across datasets --------------------
def get_frames_filename(audio_path, timebin_dur):
    """Get name for frames file, where frames are the input to the network.

    Helper function we use to standardize the name"""
    return audio_path.name + f".timebin-{timebin_dur}-ms.frames.npz"


def get_multi_frame_labels_filename(audio_path, timebin_dur, unit):
    """Get name for multiclass frame labels file,
    the target outputs for the network.

    Helper function we use to standardize the name"""
    return audio_path.name + f".timebin-{timebin_dur}-ms.{unit}.multi-frame-labels.npy"


def get_binary_frame_labels_filename(audio_path, timebin_dur, unit):
    """Get name for binary classification frame labels file,
    the target outputs for the network.

    Helper function we use to standardize the name"""
    return audio_path.name + f".timebin-{timebin_dur}-ms.{unit}.binary-frame-labels.npy"


def get_boundary_onehot_filename(audio_path, timebin_dur, unit):
    """Get name for boundary detection onehot encoding file,
    the target outputs for the network.

    Helper function we use to standardize the name"""
    return audio_path.name + f".timebin-{timebin_dur}-ms.{unit}.boundary-onehot.npy"


@dataclass
class SpectParams:
    """Parameters for spectrogramming.

    Most are arguments to
    func:`vak.prep.spectrogram_dataset.spect.spectrogram` function.

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


# ---- Bengalese finch song -------------------------------------------------------------


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


def make_inputs_targets_bf_id(id_dir, labelmap, unit='syllable', dry_run=True):
    """Make inputs and targets for one ID from Bengalese finch song

    Helper function called by
    :func:`make_inputs_targets_bengalese_finch_song`
    """
    wav_paths = voc.paths.from_dir(id_dir, '.wav')
    csv_paths = voc.paths.from_dir(id_dir, f'.{unit}.csv')

    for spect_params in BF_SPECT_PARAMS:
        logger.info(
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
        if not dry_run:
            with ProgressBar():
                dask.compute(*todo)


def make_inputs_and_targets_bengalese_finch_song(dry_run=True):
    """Make inputs and targets for neural network models,
    for Bengalese finch song

    1. Make spectrograms the way we did for the TweetyNet paper
    2. Make spectrograms same way but with 1 ms time bins
    3. For both sets of spectrograms, make following targets:
    multi-class frame label, binary frame label, boundary detection vector
    """
    ID_DIRS = [
        dir_ for dir_ in sorted(constants.BF_DATA_DST.glob("*/"))
        if dir_.is_dir()
    ]

    species_id_labelmap_map = labels.get_labelmaps()
    for id_dir in ID_DIRS:
        logger.info(
            f"Making neural network inputs and targets for: {id_dir.name}"
        )
        id = id_dir.name.split('-')[-1]
        labelmap = species_id_labelmap_map['Bengalese-Finch-Song']['syllable'][id]
        make_inputs_targets_bf_id(id_dir, labelmap, dry_run=dry_run)


# ---- canary song --------------------------------------------------------------------------


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


CANARY_SPECT_PARAMS = [
    # time bin size = 0.001 s
    SpectParams(
        fft_size=512,
        step_size=45,
        transform_type='log_spect',
        timebin_dur=1,
    ),
]


def make_inputs_targets_canary_id(raw_data_dir, id_dir, labelmap, unit='syllable', dry_run=True):
    """Make inputs and targets for one ID from canary song data"""
    spect_npz_paths = voc.paths.from_dir(raw_data_dir, '.npz')
    wav_paths = voc.paths.from_dir(id_dir, '.wav')
    csv_paths = voc.paths.from_dir(id_dir, f'.{unit}.csv')

    logger.info(
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
    if not dry_run:
        with ProgressBar():
            dask.compute(*todo)

    for spect_params in CANARY_SPECT_PARAMS:
        logger.info(
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
        if not dry_run:
            with ProgressBar():
                dask.compute(*todo)


def make_inputs_and_targets_canary_song(dry_run=True):
    """Make inputs and targets for neural network models, for canary song data

    1. Copy exact spectrograms used for the TweetyNet paper
    2. Make spectrograms with ~1 ms time bins
    3. For both sets of spectrograms, make following targets:
    multi-class frame label, binary frame label, boundary detection vector
    """
    species_id_labelmap_map = labels.get_labelmaps()

    for canary_id in constants.TWEETYNET_BIRD_IDS:
        logger.info(
            f"Making frames and targets for ID: {canary_id}"
        )
        raw_data_dir = constants.TWEETYNET_RAW_DATA / f"{canary_id}_data_matrices" / f"{canary_id}_data_matrices"
        assert raw_data_dir.exists(), f"Couldn't find raw_data_dir: {raw_data_dir}"
        id_dir = constants.CANARY_DATA_DST / f"tweetynet-{canary_id}"
        labelmap = species_id_labelmap_map['Canary-Song']['syllable'][canary_id]
        make_inputs_targets_canary_id(raw_data_dir, id_dir, labelmap, dry_run=dry_run)


# ---- mouse pup call ----------------------------------------------------------------------


def make_spectrogram_jourjine_et_al_2023(
    data,
    samplerate,
    noise_floor,
    nperseg=512,
    noverlap=128,
    min_freq=5000,
    max_freq=125000,
    num_freq_bins=128,
    fill_value=0.5,
    spec_max_val=10,
):
    """Spectrogram function for mouse pup calls from Jourjine et al. 2023 dataset

    Adapted from https://github.com/nickjourjine/peromyscus-pup-vocal-evolution/blob/main/src/spectrogramming.py
    """
    # we expect audio from vocalpy as float numpy array with range [0., 1.]
    # and convert to 16-bit int like AVA code expects
    data = ((data * 2**15).astype(np.int16))
    f, t, specgram = scipy.signal.stft(data, samplerate, nperseg=nperseg, noverlap=noverlap)

    #define the target frequencies for interpolation
    duration = np.max(t)
    target_freqs = np.linspace(min_freq, max_freq, num_freq_bins)
    # we don't interpolate times
    target_times = t
    # process
    specgram = np.log(np.abs(specgram) + 1e-12) # make a log spectrogram

    # NOTE we replace legacy code
    # interp = scipy.interpolate.interp2d(t, f, specgram, copy=False, bounds_error=False, fill_value=fill_value) #define the interpolation object
    # interp_spec = interp(target_times, target_freqs, assume_sorted=True) #interpolate
    # I don't find that np.allclose() is True for these (despite what scipy docs say), not sure if it's something I'm missing
    r = scipy.interpolate.RectBivariateSpline(t, f, specgram.T)
    interp_spec = r(target_times, target_freqs).T

    spec_min_val = noise_floor
    # min-max normalize
    spec_out = interp_spec - spec_min_val
    spec_out /= (spec_max_val - spec_min_val)
    # clip to ensure all values b/t 0 and 1
    spec_out = np.clip(spec_out, 0.0, 1.0)
    return spec_out, target_freqs, target_times


@dataclass
class JourjineEtAl2023SpectParams:
    """Dataclass that represents spectrogram parameters
    used with :func:`make_spectrogram_jourjine_et_al_2023`
    """
    npserseg: int = 512
    noverlap: int = 128
    min_freq: int = 5000
    max_freq: int = 125000
    num_freq_bins: int = 128
    fill_value: float = 0.5
    spec_max_val: int = 10
    timebin_dur: float = 1.5
    spect_key: str = 's'
    freqbins_key: str = 'f'
    timebins_key: str = 't'
    audio_path_key: str = 'audio_path'


def audio_and_annot_to_inputs_and_targets_jourjine_et_al_2023(
    audio_path: pathlib.Path,
    annot_path: pathlib.Path,
    spect_params: JourjineEtAl2023SpectParams,
    noise_floor: float,
    dst: pathlib.Path,
    labelmap: dict,
    unit: str = 'call',
):
    """Generate frames (spectrogram) and frame label / boundary detection vectors.

    This is a helper function used to parallelize, by calling it
    with `dask.delayed`.
    It is used with mouse pup call data from Jourjine et al 2023.
    """
    sound = voc.Audio.read(audio_path)

    s, f, t = make_spectrogram_jourjine_et_al_2023(
        data=sound.data,
        samplerate=sound.samplerate,
        noise_floor=noise_floor,
        nperseg=spect_params.npserseg,
        noverlap=spect_params.noverlap,
        min_freq=spect_params.min_freq,
        max_freq=spect_params.max_freq,
        num_freq_bins=spect_params.num_freq_bins,
        fill_value=spect_params.fill_value,
        spec_max_val=spect_params.spec_max_val,
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
    # NOTE: the data is labeled with a single label for all segments
    # so we do not save a multi-class frame label vector
    frame_labels_binary = vak.transforms.frame_labels.from_segments(
        lbls_int,
        annot.onsets_s,
        annot.offsets_s,
        t,
        unlabeled_label=labelmap["unlabeled"],
    )
    uniq_lbls = set(np.unique(frame_labels_binary))
    if not uniq_lbls == {0, 1}:
        raise ValueError(
            f"Expected unique values (0, 1) in frame labels but got: {uniq_lbls}"
        )
    frame_labels_binary_filename = get_binary_frame_labels_filename(
        audio_path, spect_params.timebin_dur, unit
    )
    frame_labels_binary_path = dst / frame_labels_binary_filename
    np.save(frame_labels_binary_path, frame_labels_binary)

    boundary_onehot = frame_labels_to_boundary_onehot(frame_labels_binary)
    boundary_onehot_filename = get_boundary_onehot_filename(
        audio_path, spect_params.timebin_dur, unit
    )
    boundary_onehot_path = dst / boundary_onehot_filename
    np.save(boundary_onehot_path, boundary_onehot)


JOURJINE_ET_AL_2023_SPECT_PARAMS = [
    JourjineEtAl2023SpectParams()
]


JOURJINE_ET_AL_2023_NOISE_FLOOR_CSV = constants.JOURJINE_ET_AL_2023_DATA / "processed_data" / "figure_1" / "acoustic_features" / "all_noise_floors.csv"
JOURJINE_ET_AL_2023_NOISE_FLOORS = pd.read_csv(JOURJINE_ET_AL_2023_NOISE_FLOOR_CSV)
# replace DataFrame with a dict mapping file name to noise floor,
# so we can just do a lookup table instead of dealing with pandas
JOURJINE_ET_AL_2023_NOISE_FLOORS = {
    source_file: noise_floor
    for source_file, noise_floor in zip(
        JOURJINE_ET_AL_2023_NOISE_FLOORS.source_file.values,
        JOURJINE_ET_AL_2023_NOISE_FLOORS.noise_floor.values
    )
}


def make_inputs_targets_jourjine_et_al_2023_id(id_dir, labelmap, unit='call', dry_run=True):
    """Make inputs and targets for one ID from Jourjine et al. 2023 dataset"""
    wav_paths = voc.paths.from_dir(id_dir, '.wav')
    csv_paths = voc.paths.from_dir(id_dir, f'.{unit}.csv')

    for spect_params in JOURJINE_ET_AL_2023_SPECT_PARAMS:
        print(
            f"Making inputs and targets with `spect_params`: {spect_params}"
        )
        todo = []
        pbar = tqdm.tqdm(zip(wav_paths, csv_paths))
        for wav_path, csv_path in pbar:
            noise_floor = JOURJINE_ET_AL_2023_NOISE_FLOORS[wav_path.name]
            if not dry_run:
                audio_and_annot_to_inputs_and_targets_jourjine_et_al_2023(
                    audio_path=wav_path,
                    annot_path=csv_path,
                    spect_params=spect_params,
                    noise_floor=noise_floor,
                    dst=id_dir,
                    labelmap=labelmap,
                    unit=unit,
                )
        # FIXME: when I run parallelized with dask I get `killed` -- out of memory error?
        #     todo.append(
        #         dask.delayed(audio_and_annot_to_inputs_and_targets_jourjine_et_al_2023)(
        #             audio_path=wav_path,
        #             annot_path=csv_path,
        #             spect_params=spect_params,
        #             noise_floor=noise_floor,
        #             dst=id_dir,
        #             labelmap=labelmap,
        #             unit=unit,
        #         )
        #     )
        # if not dry_run:
        #     with ProgressBar():
        #         dask.compute(*todo)


JOURJINE_ET_AL_2023_LABELSET = ["v"]
JOURJINE_ET_AL_2023_LABELSET = vak.common.converters.labelset_to_set(JOURJINE_ET_AL_2023_LABELSET)
JOURJINE_ET_AL_2023_LABELMAP = vak.common.labels.to_map(JOURJINE_ET_AL_2023_LABELSET)


def make_inputs_and_targets_mouse_pup_call(dry_run=True):
    """Make inputs and targets for neural network models,
    for mouse pup call data.

    1. Make spectrograms similiar to how they did for UMAP in Jourjine et al., 2023
    We post-process in the same way, and we interpolate
    in the frequency dimension to get 128 bins, but we don't interpolate in time.
    We only make 1 set of spectrograms here, since the time bin size is 1.5ms,
    and I'm not running a giant set of experiments
    just to test how much of a difference 0.5 ms makes.
    2. For this set of spectrograms, make following targets:
    multi-class frame label, binary frame label, boundary detection vector
    """
    MOUSE_ID_DIRS = sorted([
        dir_ for dir_ in constants.MOUSE_PUP_CALL_DATA_DST.iterdir() if dir_.is_dir()
    ])

    for id_dir in MOUSE_ID_DIRS:
        assert id_dir.name.startswith('jourjine-et-al-2023')
        logger.info(
            f"Making inputs and targets for species with ID: {id_dir.name.split('-')[-1]}."
        )
        make_inputs_targets_jourjine_et_al_2023_id(
            id_dir,
            labelmap=JOURJINE_ET_AL_2023_LABELMAP,
            unit='call',
            dry_run=dry_run
        )


# ---- zebra finch song -----------------------------------------------------------------------------------------------------------------------------


@dataclass
class VocSpectParams:
    """Parameters for :func:`vocalpy.spectrogram`

    Attributes
    ----------
    n_fft : int
        size of window for Fast Fourier transform, number of time bins.
    hop_length : int
        step size for Fast Fourier transform.
    timebin_dur : int
        Expected duration of timebins in spectrogram, in milliseconds.
        Used to validate output of spectrogram function,
        and in creating filename for spectrogram (as a form of metadata).
    spect_key : str        key for accessing spectrogram in files. Default is 's'.
    freqbins_key : str
        key for accessing vector of frequency bins in files. Default is 'f'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    audio_path_key : str
        key for accessing path to source audio file for spectogram in files.
        Default is 'audio_path'.
    """
    n_fft: int
    hop_length: int
    timebin_dur: int
    spect_key: str = 's'
    freqbins_key: str = 'f'
    timebins_key: str = 't'
    audio_path_key: str = 'audio_path'


import math


SCRIBE = crowsetta.Transcriber(format='simple-seq')


def audio_and_annot_to_inputs_and_targets_zf(
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
    It is used with zebra finch song.
    """
    sound = voc.Audio.read(audio_path)

    spect = voc.spectrogram(
        sound,
        spect_params.n_fft,
        spect_params.hop_length,
    )
    timebin_dur = np.diff(spect.times).mean()
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
        spect_params.spect_key: spect.data,
        spect_params.freqbins_key: spect.frequencies,
        spect_params.timebins_key: spect.times,
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
        spect.times,
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


ZF_SPECT_PARAMS = [
    VocSpectParams(
        n_fft=64,
        hop_length=16,
        timebin_dur=0.5, # ms!
    ),
    VocSpectParams(
        n_fft=512,
        hop_length=32,
        timebin_dur=1.0, # ms!
    ),
]


def make_inputs_and_targets_zf_id(id_dir, labelmap, unit='syllable', dry_run=True):
    """Make inputs and targets for one ID from zebra finch song"""
    wav_paths = voc.paths.from_dir(id_dir, '.wav')
    csv_paths = voc.paths.from_dir(id_dir, f'.{unit}.csv')

    for spect_params in ZF_SPECT_PARAMS:
        logger.info(
            f"Making inputs and targets with `spect_params`: {spect_params}"
        )
        todo = []
        for wav_path, csv_path in zip(wav_paths, csv_paths):
            todo.append(
                dask.delayed(audio_and_annot_to_inputs_and_targets_zf)(
                    audio_path=wav_path,
                    annot_path=csv_path,
                    spect_params=spect_params,
                    dst=id_dir,
                    labelmap=labelmap,
                    unit=unit,
                )
            )
        if not dry_run:
            with ProgressBar():
                dask.compute(*todo)


def make_inputs_and_targets_zebra_finch_song(dry_run=True):
    """Make inputs and targets for neural networks, for Zebra finch song

    1. Generate spectrograms using parameters for STFT front-end in DAS paper: n_fft=64, step_size=16
    2. Make spectrograms with ~1 ms time bins (and many more frequency bins): n_fft=512, step_size=32
    3. For both sets of spectrograms, make following targets:
    multi-class frame label, binary frame label, boundary detection vector
    """
    ZB_ID_DIRS = sorted([
        dir_ for dir_ in constants.ZB_DATA_DST.iterdir() if dir_.is_dir()
    ])

    species_id_labelmap_map = labels.get_labelmaps()
    for id_dir in ZB_ID_DIRS:
        logger.info(
            f"Making neural network inputs and targets for: {id_dir.name}"
        )
        id = id_dir.name.split('-')[-1]
        labelmap = species_id_labelmap_map['Zebra-Finch-Song']['syllable'][id]
        make_inputs_and_targets_zf_id(id_dir, labelmap, dry_run=dry_run)


# ---- human speech ---------------------------------------------------------------------------------------------------------------------------------


def speech_audio_to_MFCC_features(
    audio_path,
    hop_length_s=0.01,
    n_fft_s=0.025,
    n_mels=40,
    n_mfcc=13,
    add_dist_features=False,
):
    """Converts audio into framewise features

    Adapted from https://github.com/felixkreuk/SegFeat

    > ... we extracted 13 Mel-Frequency Cepstrum Coefficients (MFCCs),
    with delta and delta-delta features every 10 ms,
    with a processing window size of 10ms.

    If `add_dist_features` is True, this function adds additional features:

    > Moreover, we concatenated four additional features based
    on the spectral changes between adjacent frames, using
    MFCCs to represent the spectral properties of the frames.
    Define Dt,j = d(at−j , at+j ) to be the Euclidean distance
    between the MFCC feature vectors at−j and at+j , where
    at ∈ R for 1 ≤ t ≤ T . The features are denoted by Dt,j ,
    for j ∈ {1, 2, 3, 4}. We observed this set of features greatly
    improves performance over the standard MFCC features.

    See also: https://groups.google.com/g/librosa/c/V4Z1HpTKn8Q/m/1-sMpjxjCSoJ
    """
    y, sr = librosa.load(audio_path, sr=None)
    hop_length = int(hop_length_s * sr)
    n_fft = int(n_fft_s * sr)
    spect = librosa.feature.mfcc(y=y,
                                 sr=sr,
                                 n_fft=n_fft,
                                 hop_length=hop_length,
                                 n_mels=n_mels,
                                 n_mfcc=n_mfcc
                                )

    delta  = librosa.feature.delta(spect, order=1)
    delta2 = librosa.feature.delta(spect, order=2)
    frames  = np.concatenate([spect, delta, delta2], axis=0)
    if add_dist_features:
        dist = []
        for i in range(2, 9, 2):
            pad = int(i/2)
            d_i = np.concatenate([np.zeros(pad), ((spect[:, i:] - spect[:, :-i]) ** 2).sum(0) ** 0.5, np.zeros(pad)], axis=0)
            dist.append(d_i)
        dist = np.stack(dist)
        frames = np.concatenate([frames, dist], axis=0)
    times = librosa.frames_to_time(
        np.arange(frames.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft
    )
    return frames, times


@dataclass
class SpeechMFCCParams:
    n_fft_s: float = 0.25
    hop_length_s: float = 0.01
    n_mels: int = 40
    n_mfcc: int = 13
    timebin_dur: float = 10.  # ms
    add_dist_features: bool = False
    spect_key: str = 's'
    freqbins_key: str = 'f'
    timebins_key: str = 't'
    audio_path_key: str = 'audio_path'


def boundary_onehot_from_times(
    boundary_times: np.ndarray,
    time_bins: np.ndarray,
):
    """Get boundary detection vector encoded as one-hot
    labels for frames, from vector of boundary times
    and vector of frame times"""
    # we assume there will be a boundary at 0
    # which is true for TIMIT
    boundary_inds = np.array(
        # TODO: vectorize subtraction then get argmin row/column-wise
        [np.argmin(np.abs(time_bins - boundary_time))
         for boundary_time in boundary_times]
    )
    boundary_onehot = np.zeros_like(time_bins).astype(int)
    boundary_onehot[boundary_inds] = 1
    return boundary_onehot



def audio_and_annot_to_inputs_and_targets_speech(
    audio_path: pathlib.Path,
    mfcc_params: SpeechMFCCParams,
    unit_annot_path_map: dict[str: pathlib.Path],
    dst: pathlib.Path,
    labelmap: dict,
):
    """Generate frames (spectrogram) and frame label / boundary detection vectors.

    This is a helper function used to parallelize, by calling it
    with `dask.delayed`.
    It is used with human speech.
    """
    frames, times = speech_audio_to_MFCC_features(
        audio_path,
        hop_length_s=mfcc_params.hop_length_s,
        n_fft_s=mfcc_params.n_fft_s,
        n_mels=mfcc_params.n_mels,
        n_mfcc=mfcc_params.n_mfcc,
        add_dist_features=mfcc_params.add_dist_features,
    )

    timebin_dur = np.diff(times).mean()
    if not math.isclose(
        timebin_dur,
        mfcc_params.timebin_dur * 1e-3,
        abs_tol=0.001,
    ):
        raise ValueError(
            f"Expected MFCC frames with timebins of duration {mfcc_params.timebin_dur * 1e-3} "
            f"but got duration {timebin_dur} for audio path: {audio_path}"
        )
    frames_dict = {
        mfcc_params.spect_key: frames,
        mfcc_params.timebins_key: times,
    }

    frames_filename = get_frames_filename(
        audio_path, mfcc_params.timebin_dur
    )
    frames_path = dst / frames_filename
    np.savez(frames_path, **frames_dict)

    for unit, annot_path in unit_annot_path_map.items():
        if annot_path is None:
            continue
        annot = SCRIBE.from_file(annot_path)
        if unit == 'phoneme':
            # labelmap has phonemes (not words!)
            lbls_int = [labelmap[lbl] for lbl in annot.labels]
            frame_labels_multi = vak.transforms.frame_labels.from_segments(
                lbls_int,
                annot.onsets_s,
                annot.offsets_s,
                times,
                unlabeled_label=labelmap["unlabeled"],
            )
            frame_labels_multi_filename = get_multi_frame_labels_filename(
                audio_path, mfcc_params.timebin_dur, unit
            )
            frame_labels_multi_path = dst / frame_labels_multi_filename
            np.save(frame_labels_multi_path, frame_labels_multi)

            # we don't generate binary classification since
            # in general there are not silent segments
            # between phonemes

            boundary_times = np.unique(
                voc.metrics.segmentation.ir.concat_starts_and_stops(
                    annot.onsets_s, annot.offsets_s)
            )
            boundary_onehot = boundary_onehot_from_times(
                boundary_times,
                times
            )
            boundary_onehot_filename = get_boundary_onehot_filename(
                audio_path, mfcc_params.timebin_dur, unit
            )
            boundary_onehot_path = dst / boundary_onehot_filename
            np.save(boundary_onehot_path, boundary_onehot)
        elif unit == 'word':
            # we don't generate multi-class frame labels because
            # of the extreme imbalance between classes --
            # some words occur only once

            # we don't generate binary classification since
            # in general there are not silent segments
            # between words

            # so that leaves us ... boundary times
            boundary_times = np.unique(
                voc.metrics.segmentation.ir.concat_starts_and_stops(
                    annot.onsets_s, annot.offsets_s)
            )
            boundary_onehot = boundary_onehot_from_times(
                boundary_times,
                times
            )
            boundary_onehot_filename = get_boundary_onehot_filename(
                audio_path, mfcc_params.timebin_dur, unit
            )
            boundary_onehot_path = dst / boundary_onehot_filename
            np.save(boundary_onehot_path, boundary_onehot)


HUMAN_SPEECH_MFCC_PARAMS = [
    # (default) time bin dur of 10 ms
    SpeechMFCCParams(),
    # timebin dur of 1 ms
    SpeechMFCCParams(hop_length_s=0.001, timebin_dur=1.),
]


def make_inputs_targets_speech_speaker_id(speaker_id_dir, labelmap, dry_run=True):
    """
    """
    wav_paths = sorted(speaker_id_dir.glob('*.wav'))
    wrd_paths = sorted(speaker_id_dir.glob('*.word.csv'))
    phn_paths = sorted(speaker_id_dir.glob('*.phoneme.csv'))

    wav_with_unit_annot = []
    phn_paths_fnd = 0
    wrd_paths_fnd = 0
    for wav_path in wav_paths:
        unit_annot_path_map = {}
        expected_phn_path = speaker_id_dir / f"{wav_path.name}.phoneme.csv"
        if expected_phn_path in phn_paths:
            unit_annot_path_map['phoneme'] = expected_phn_path
            phn_paths_fnd += 1
        else:
            unit_annot_path_map['phoneme'] = None
        expected_wrd_path = speaker_id_dir / f"{wav_path.name}.word.csv"
        if expected_wrd_path in wrd_paths:
            unit_annot_path_map['word'] = expected_wrd_path
            wrd_paths_fnd += 1
        else:
            unit_annot_path_map['word'] = None
        wav_with_unit_annot.append(
            (wav_path, unit_annot_path_map)
        )
    if phn_paths_fnd != len(phn_paths):
        raise ValueError(
            f"Found {phn_paths_fnd} phoneme annotation paths when pairing "
            f"but found {len(phn_paths)} in directory:\n{speaker_id_dir}"
        )

    if wrd_paths_fnd != len(wrd_paths):
        raise ValueError(
            f"Found {wrd_paths_fnd} word annotation paths when pairing "
            f"but found {len(wrd_paths)} in directory:\n{speaker_id_dir}"
        )

    for mfcc_params in HUMAN_SPEECH_MFCC_PARAMS:
        logger.info(
            f"Making inputs and targets with `mfcc_params`: {mfcc_params}"
        )
        todo = []
        for wav_path, unit_annot_path_map in wav_with_unit_annot:
            todo.append(
                dask.delayed(audio_and_annot_to_inputs_and_targets_speech)(
                    wav_path,
                    mfcc_params=mfcc_params,
                    unit_annot_path_map=unit_annot_path_map,
                    dst=speaker_id_dir,
                    labelmap=labelmap,
                )
            )
        if not dry_run:
            with ProgressBar():
                dask.compute(*todo)


def make_inputs_and_targets_human_speech(dry_run=True):
    """
    """
    TIMIT_DIALECT_SPKR_DIRS = [
        dir_ for dir_ in constants.HUMAN_SPEECH_WE_CANT_SHARE.iterdir()
        if dir_.is_dir()
    ] + [dir_ for dir_ in constants.SPEECH_DATA_DST if dir_.is_dir()]

    species_id_labelmap_map = labels.get_labelmaps()
    labelmap = species_id_labelmap_map['Human-Speech']['phoneme']['all']
    for id_dir in TIMIT_DIALECT_SPKR_DIRS:
        logger.info(
            f"Making neural network inputs and targets for: {id_dir.name}"
        )
        id = id_dir.name.split('-')[-1]
        make_inputs_targets_speech_speaker_id(id_dir, labelmap, dry_run=dry_run)


# ---- all ------------------------------------------------------------------------------------------------------------------------------------------


def make_inputs_and_targets_all(biosound_groups, dry_run=True):
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
