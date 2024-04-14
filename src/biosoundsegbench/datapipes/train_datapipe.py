from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import vak
import vocalpy as voc
from vak.datasets.frame_classification import constants, helper
from vak.datasets.frame_classification.window_dataset import get_window_inds


class TrainDatapipe:
    """Datapipe used to train models.

    Durining training, we build batches from random grabs of windows.
    """
    def __init__(
        self,
        dataset_path: str | pathlib.Path,
        dataset_df: pd.DataFrame,
        split: str,
        sample_ids: npt.NDArray,
        inds_in_sample: npt.NDArray,
        window_size: int,
        frame_dur: float,
        stride: int = 1,
        subset: str | None = None,
        window_inds: npt.NDArray | None = None,
        item_transform: Callable | None = None,
    ):
        self.dataset_path = pathlib.Path(dataset_path)

        self.split = split
        self.subset = subset
        # subset takes precedence over split, if specified
        if subset:
            dataset_df = dataset_df[dataset_df.subset == subset].copy()
        else:
            dataset_df = dataset_df[dataset_df.split == split].copy()
        self.dataset_df = dataset_df

        self.frames_paths = self.dataset_df[
            constants.FRAMES_PATH_COL_NAME
        ].values
        self.frame_labels_paths = self.dataset_df[
            constants.FRAME_LABELS_NPY_PATH_COL_NAME
        ].values
        self.sample_ids = sample_ids
        self.inds_in_sample = inds_in_sample
        self.window_size = window_size
        self.frame_dur = float(frame_dur)
        self.stride = stride
        if window_inds is None:
            window_inds = get_window_inds(
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

    @property
    def duration(self):
        return self.sample_ids.shape[-1] * self.frame_dur

    @property
    def shape(self):
        tmp_x_ind = 0
        one_x, _ = self.__getitem__(tmp_x_ind)
        # used by vak functions that need to determine size of window,
        # e.g. when initializing a neural network model
        return one_x.shape

    def _load_frames(self, frames_path):
        return helper.load_frames(frames_path, "spect")

    def __getitem__(self, idx):
        window_idx = self.window_inds[idx]
        sample_ids = self.sample_ids[
            window_idx : window_idx + self.window_size  # noqa: E203
        ]
        uniq_sample_ids = np.unique(sample_ids)
        if len(uniq_sample_ids) == 1:
            # we repeat ourselves here to avoid running a loop on one item
            sample_id = uniq_sample_ids[0]
            frames_path = self.dataset_path / self.frames_paths[sample_id]
            frames, frame_times = self._load_frames(frames_path)
            frame_labels = np.load(
                self.dataset_path / self.frame_labels_paths[sample_id]
            )

        elif len(uniq_sample_ids) > 1:
            frames = []
            frame_labels = []
            for sample_id in sorted(uniq_sample_ids):
                frames_path = self.dataset_path / self.frames_paths[sample_id]
                frames_, frame_times = self._load_frames(frames_path)
                frames.append(frames_)
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
            inds_in_sample : inds_in_sample + self.window_size  # noqa: E203
        ]
        frame_labels = frame_labels[
            inds_in_sample : inds_in_sample + self.window_size  # noqa: E203
        ]
        item = self.item_transform(frames, frame_labels)
        return item

    def __len__(self):
        """number of batches"""
        return len(self.window_inds)

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str | pathlib.Path,
        window_size: int,
        stride: int = 1,
        split: str = "train",
        subset: str | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        dataset_path = pathlib.Path(dataset_path)
        metadata = vak.datasets.frame_classification.Metadata.from_dataset_path(dataset_path)
        frame_dur = metadata.frame_dur

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)

        split_path = dataset_path / split
        if subset:
            sample_ids_path = (
                split_path
                / helper.sample_ids_array_filename_for_subset(subset)
            )
        else:
            sample_ids_path = split_path / constants.SAMPLE_IDS_ARRAY_FILENAME
        sample_ids = np.load(sample_ids_path)

        if subset:
            inds_in_sample_path = (
                split_path
                / helper.inds_in_sample_array_filename_for_subset(subset)
            )
        else:
            inds_in_sample_path = (
                split_path / constants.INDS_IN_SAMPLE_ARRAY_FILENAME
            )
        inds_in_sample = np.load(inds_in_sample_path)

        window_inds_path = split_path / constants.WINDOW_INDS_ARRAY_FILENAME
        if window_inds_path.exists():
            window_inds = np.load(window_inds_path)
        else:
            window_inds = None

        return cls(
            dataset_path,
            dataset_df,
            split,
            sample_ids,
            inds_in_sample,
            window_size,
            frame_dur,
            stride,
            subset,
            window_inds,
            transform,
            target_transform,
        )
