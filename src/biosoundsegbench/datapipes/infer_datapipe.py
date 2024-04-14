from __future__ import annotations

import pathlib
from typing import Callable

import crowsetta
import numpy as np
import numpy.typing as npt
import pandas as pd
import vak
from vak.datasets.frame_classification import constants, helper


class InferDatapipe:
    """Datapipe used for inference from models.

    Used to evaluate models and generate predictions from models.
    During inference, we convert spectrograms / series of frames
    into batches of consecutive non-overlapping windows.
    We then flatten the output of the model along the time dimension
    to recover outputs for the entire spectrogram or series of frames.
    """
    def __init__(
        self,
        dataset_path: str | pathlib.Path,
        dataset_df: pd.DataFrame,
        split: str,
        sample_ids: npt.NDArray,
        inds_in_sample: npt.NDArray,
        frame_dur: float,
        item_transform: Callable,
    ):
        self.dataset_path = pathlib.Path(dataset_path)
        self.split = split
        self.dataset_df = dataset_df
        self.frames_paths = self.dataset_df[
            constants.FRAMES_PATH_COL_NAME
        ].values
        if split != "predict":
            self.frame_labels_paths = self.dataset_df[
                constants.FRAME_LABELS_NPY_PATH_COL_NAME
            ].values
            self.annots = [
                crowsetta.formats.seq.SimpleSeq.from_file(self.dataset_path / annot_path).to_annot()
                for annot_path in self.dataset_df['annot_path']
            ]
        else:
            self.frame_labels_paths = None
            self.annots = None
        self.sample_ids = sample_ids
        self.inds_in_sample = inds_in_sample
        self.frame_dur = float(frame_dur)
        self.item_transform = item_transform

    @property
    def duration(self):
        return self.sample_ids.shape[-1] * self.frame_dur

    @property
    def shape(self):
        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        return tmp_item["frames"].shape

    def _load_frames(self, frames_path):
        return helper.load_frames(frames_path, "spect")

    def __getitem__(self, idx):
        frames_path = self.dataset_path / self.frames_paths[idx]
        frames, frame_times = self._load_frames(frames_path)
        item = {"frames": frames, "frames_path": frames_path, "frame_times": frame_times}
        if self.frame_labels_paths is not None:
            frame_labels = np.load(
                self.dataset_path / self.frame_labels_paths[idx]
            )
            item["frame_labels"] = frame_labels
        if self.annots is not None:
            item["annot"] = self.annots[idx]
        if self.item_transform:
            item = self.item_transform(**item)

        return item

    def __len__(self):
        """number of batches"""
        return len(np.unique(self.sample_ids))

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str | pathlib.Path,
        split: str = "val",
        window_size,
        spect_standardizer=None,
        frames_padval=0.0,
        frame_labels_padval=-1,

    ):
        dataset_path = pathlib.Path(dataset_path)
        metadata = vak.datasets.frame_classification.Metadata.from_dataset_path(dataset_path)
        frame_dur = metadata.frame_dur
        input_type = metadata.input_type

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)
        dataset_df = dataset_df[dataset_df.split == split].copy()

        split_path = dataset_path / split
        sample_ids_path = split_path / constants.SAMPLE_IDS_ARRAY_FILENAME
        sample_ids = np.load(sample_ids_path)

        inds_in_sample_path = (
            split_path / constants.INDS_IN_SAMPLE_ARRAY_FILENAME
        )
        inds_in_sample = np.load(inds_in_sample_path)

        item_transform = EvalItemTransform(
            window_size=window_size,
            spect_standardizer=spect_standardizer,
            frames_padval=frames_padval,
            frame_labels_padval=frame_labels_padval,

        )

        return cls(
            dataset_path,
            dataset_df,
            split,
            sample_ids,
            inds_in_sample,
            frame_dur,
            item_transform,
        )
