from __future__ import annotation

from typing import Callable, Mapping

import json
import pathlib

import numpy as np
import numpy.typing as npt
import pandas as pd

from .. import transforms

# TODO: this should use importlib-resources
with pathlib.Path('./biosoundsegbench.json').open('r') as fp:
    BIOSOUNDSEGBENCH_META = json.load(fp)


FRAMES_PATH_COL_NAME = "frames_path"
FRAME_LABELS_EXT = ".frame_labels.npy"
FRAME_LABELS_NPY_PATH_COL_NAME = "frame_labels_npy_path"
SAMPLE_IDS_ARRAY_FILENAME = "sample_ids.npy"
INDS_IN_SAMPLE_ARRAY_FILENAME = "inds_in_sample.npy"
WINDOW_INDS_ARRAY_FILENAME = "window_inds.npy"
FRAME_CLASSIFICATION_DATASET_AUDIO_FORMAT = "wav"


# TODO: we need to use the metadata to validate
# I think it makes the most sense to organize by target type first?
# VALID_TARGET_TYPES = ('label-multi', 'boundary', 'label-binary')
# VALID_SPECIES = ('bengalese finch', 'canary')
# VALD_UNIT = ('syllable')

# paths here are relative to ``root`` of BioSoundSegBench
INIT_ARGS_CSV_MAP = {
    'bengalese finch': {
        'syllable': {
            'gy6or6': ''
        }
    }

}

VALID_TARGET_TYPES = (
    'boundary',
    'label-multi',
    'label-binary',
    ('boundary', 'label-binary'),
    ('boundary', 'label-multi'),
)


class BioSoundSegBench:
    def __init__(
            self,
            root: str | pathlib.Path,
            species: str | list[str] | tuple[str],
            target_type: str | list[str] | tuple[str],
            unit: str,
            id: str | None,
            split: str,
            window_size: int,
            item_transform: Callable,
            stride: int = 1,
            ):
        """BioSoundSegBench dataset."""
        root = pathlib.Path(root)
        if not root.exists() or not root.is_dir():
            raise NotADirectoryError()

        if target_type not in VALID_TARGET_TYPES:
            raise ValueError(
                f"Invalid `target_type`: {target_type}. "
                f"Valid target types are: {VALID_TARGET_TYPES}"
            )
        if isinstance(target_type, str):
            # make single str a tuple so we can do ``if 'some target' in self.target_type``
            target_type = (target_type,)
        self.target_type = target_type

        self.dataset_df = self._get_dataset_df(
            root,
            species,
            unit,
            id,
        )

        self.split = split
        dataset_df = dataset_df[dataset_df.split == split].copy()
        self.dataset_df = dataset_df

        self.frames_paths = self.dataset_df[
            FRAMES_PATH_COL_NAME
        ].values
        if 'label-multi' in self.target_type:
            self.multiclass_frame_labels_paths = self.dataset_df[
                FRAME_LABELS_NPY_PATH_COL_NAME
            ].values
        else:
            self

        sample_ids, inds_in_sample = self._get_frames_vectors()
        self.sample_ids = sample_ids
        self.inds_in_sample = inds_in_sample
        self.window_size = window_size
        self.frame_dur = float(frame_dur)
        self.stride = stride
        if window_inds is None:
            window_inds = vak.datasets.frame_classification.window_dataset.get_window_inds(
                sample_ids.shape[-1], window_size, stride
            )
        self.window_inds = window_inds
        self.item_transform = item_transform

        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        # used by vak functions that need to determine size of input,
        # e.g. when initializing a neural network model
        self.num_channels = tmp_item["frames"].shape[0]
        self.num_freqbins = tmp_item["frames"].shape[1]

    def _get_dataset_df(
            root,
            species,
            unit,
            id,
    ):
        species_dict = INIT_ARGS_CSV_MAP[species]
        if id is None:
            # we ignore individual ID and concatenate all CSVs
            # TODO: we will need to deal with labelmap in this case
            ids_dict = species_dict[unit]
            csv_paths = [
                csv_path for id, csv_path in ids_dict.items()
            ]
        else:
            csv_paths = [species_dict[unit][id]]
        dataset_df = []
        for csv_path in csv_paths:
            dataset_df.append(pd.read_csv(csv_path))
        dataset_df = pd.concat(dataset_df)
        return dataset_df

    def _get_frames_vectors(self):
        # we're going to have different sample_ids + inds_in_sample_path
        # for some of the same files,
        # two cases:
        # (1) where we train a model on multiple individuals;
        # (2) where we train a model on multiple species (for which we also train separate models, ignoring cases like USVSEG)
        # I don't know right now how we will make / save those?
        # I guess in a separate step where we build each dataset
        split_path = dataset_path / split
        sample_ids_path = split_path / SAMPLE_IDS_ARRAY_FILENAME
        sample_ids = np.load(sample_ids_path)
        inds_in_sample_path = (
            split_path / INDS_IN_SAMPLE_ARRAY_FILENAME
        )
        inds_in_sample = np.load(inds_in_sample_path)



def get_biosoundsegbench(
    root: str | pathlib.Path,
    species: str | list[str] | tuple[str],
    target_type: str | list[str] | tuple[str],
    unit: str,
    id: str | None,
    split: str,
    window_size: int,
    stride: int = 1,
    labelmap: Mapping | None = None
):
    """Get a :class:`DataPipe` instance
    for loading samples from the BioSoundSegBench.

    This function determines the correct data to use,
    according to the `species`, `unit`, and `id`
    specified.

    It also determines which `transform` to use,
    according to the `target_type`.
    """
    root = pathlib.Path(root)
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError()

    species_dict = INIT_ARGS_CSV_MAP[species]
    if id is None:
        # we ignore individual ID and concatenate all CSVs
        ids_dict = species_dict[unit]
        csv_paths = [
            csv_path for id, csv_path in ids_dict.items()
        ]
    else:
        csv_paths = [species_dict[unit][id]]
    dataset_df = []
    for csv_path in csv_paths:
        dataset_df.append(pd.read_csv(csv_path))
    dataset_df = pd.concat(dataset_df)

    # TODO: I think this is a case where we need an "item transform" for train,
    # to encapsulate the logic of dealing with different target types
    if split == 'train':
        # for boundary detection and binary classification, we use target transforms
        # instead of loading from separate vectors for now
        # TODO: fix this to load from separate data we prep -- be more frugal at runtime
        if target_type == 'boundary':
            target_transform = transforms.FrameLabelsToBoundaryOnehot()
        elif target_type == 'label-binary':
            target_transform = transforms.
        elif target_type == 'label-multi':
            # all we have to do is load the frame labels vector
            target_transform = None


    elif split in ('val', 'test', 'predict'):


class BioSoundSegBench:
    def __init__(
            self,
            root: str | pathlib.Path,
            species: str | list[str] | tuple[str],
            target_type: str | list[str] | tuple[str],
            unit: str,
            id: str | None,
            split: str,
            window_size: int,
    ):
        # dataset_csv_path = BIOSOUNDSEGBENCH_META[target_type][species][unit]
        self.root = pathlib.Path(root)

    def __getitem__(self, idx):
        if self.split == 'train':
            window_idx = self.window_inds[idx]
            sample_ids = self.sample_ids[
                window_idx : window_idx + self.window_size  # noqa: E203
            ]
            uniq_sample_ids = np.unique(sample_ids)
            if len(uniq_sample_ids) == 1:
                # we repeat ourselves here to avoid running a loop on one item
                sample_id = uniq_sample_ids[0]
                frames_path = self.dataset_path / self.frames_paths[sample_id]
                frames = self._load_frames(frames_path)
                if 'label-multi' in self.target_types:
                    frame_labels_multi = np.load(
                        self.dataset_path / self.frame_labels_paths[sample_id]
                    )

            elif len(uniq_sample_ids) > 1:
                frames = []
                frame_labels = []
                for sample_id in sorted(uniq_sample_ids):
                    frames_path = self.dataset_path / self.frames_paths[sample_id]
                    frames.append(self._load_frames(frames_path))
                    frame_labels.append(
                        np.load(
                            self.dataset_path / self.frame_labels_paths[sample_id]
                        )
                    )

                if all([frames_.ndim == 1 for frames_ in frames]):
                    # --> all 1-d audio vectors; if we specify `axis=1` here we'd get error
                    frames = np.concatenate(frames)
                else:
                    frames = np.concatenate(frames, axis=1)
                frame_labels = np.concatenate(frame_labels)
            else:
                raise ValueError(
                    f"Unexpected number of ``uniq_sample_ids``: {uniq_sample_ids}"
                )

            inds_in_sample = self.inds_in_sample[window_idx]
            frames = frames[
                ...,
                inds_in_sample : inds_in_sample + self.window_size,  # noqa: E203
            ]
            frame_labels = frame_labels[
                inds_in_sample : inds_in_sample + self.window_size  # noqa: E203
            ]
            if self.transform:
                frames = self.transform(frames)
            if self.target_transform:
                frame_labels = self.target_transform(frame_labels)

