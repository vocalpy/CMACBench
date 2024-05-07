"""Transforms used by the `__getitem__` method of DataPipes.

These transforms encapsulate the logic of handing different `target_types`.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torchvision
import vak

from .transforms import FrameLabelsMultiToBinary, FrameLabelsToBoundaryOnehot, ViewAsWindowBatch


TARGET_TYPES = (
    'boundary_frame_labels',
    'multi_frame_labels',
    'binary_frame_labels',
)


class TrainItemTransform:
    """Transform used when training models"""
    def __init__(
        self,
        spect_standardizer=None,
    ):
        if spect_standardizer is not None:
            if isinstance(spect_standardizer, vak.transforms.StandardizeSpect):
                frames_transform = [spect_standardizer]
            else:
                raise TypeError(
                    f"invalid type for spect_standardizer: {type(spect_standardizer)}. "
                    "Should be an instance of vak.transforms.StandardizeSpect"
                )
        else:
            frames_transform = []

        frames_transform.extend(
            [
                vak.transforms.ToFloatTensor(),
                vak.transforms.AddChannel(),
            ]
        )
        self.frames_transform = torchvision.transforms.Compose(
            frames_transform
        )
        self.frame_labels_transform = vak.transforms.ToLongTensor()

    def __call__(self, item: dict):
        item['frames'] = self.frames_transform(item['frames'])
        for target_type in TARGET_TYPES:
            if target_type in item:
                item[target_type] = self.frame_labels_transform(item[target_type])
        return item


class EvalItemTransform:
    def __init__(
        self,
        window_size,
        spect_standardizer=None,
        frames_padval=0.0,
        frame_labels_padval=-1,
        return_padding_mask=True,
        channel_dim=1,
    ):
        self.window_size = window_size

        if spect_standardizer is not None:
            if not isinstance(
                spect_standardizer, vak.transforms.StandardizeSpect
            ):
                raise TypeError(
                    f"invalid type for spect_standardizer: {type(spect_standardizer)}. "
                    "Should be an instance of vak.transforms.StandardizeSpect"
                )
        self.spect_standardizer = spect_standardizer

        self.pad_to_window = vak.transforms.PadToWindow(
            window_size, frames_padval, return_padding_mask=return_padding_mask
        )

        self.frames_transform_after_pad = torchvision.transforms.Compose(
            [
                vak.transforms.ViewAsWindowBatch(self.window_size),
                vak.transforms.ToFloatTensor(),
                # below, add channel at first dimension because windows become batch
                vak.transforms.AddChannel(channel_dim=channel_dim),
            ]
        )

        self.frame_labels_padval = frame_labels_padval
        self.frame_labels_transform = torchvision.transforms.Compose(
            [
                vak.transforms.PadToWindow(
                    self.window_size, self.frame_labels_padval, return_padding_mask=False
                ),
                ViewAsWindowBatch(window_size),
                vak.transforms.ToLongTensor()
            ]
        )

    def __call__(self, item, frame_times=None, annot=None, frames_path=None):
        # ---- handle spectrogram
        if self.spect_standardizer:
            frames = self.spect_standardizer(item['frames'])
        else:
            frames = item['frames']

        if self.pad_to_window.return_padding_mask:
            frames, padding_mask = self.pad_to_window(frames)
        else:
            frames = self.pad_to_window(frames)
            padding_mask = None

        item['frames'] = self.frames_transform_after_pad(frames)
        if padding_mask is not None:
            item["padding_mask"] = padding_mask

        for target_type in TARGET_TYPES:
            if target_type in item:
                item[target_type] = self.frame_labels_transform(item[target_type])

        if frame_times:
            item['frame_times'] = frame_times
        if annot:
            item['annot'] = annot
        if frames_path is not None:
            # make sure frames_path is a str, not a pathlib.Path
            item["frames_path"] = str(frames_path)

        return item
