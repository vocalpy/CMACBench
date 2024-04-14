"""Transforms used by the `__getitem__` method of DataPipes.

These transforms encapsulate the logic of handing different `target_types`.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torchvision
import vak

from .transforms import FrameLabelsMultiToBinary, FrameLabelsToBoundaryOnehot, ViewAsWindowBatch


class TrainItemTransform:
    """Transform used when training frame classification models"""
    def __init__(
        self,
        target_type: str,
        spect_standardizer=None,
        labelmap: Mapping | None = None,
        bg_class_name: str = 'unlabeled',
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

        if target_type == 'boundary':
            self.to_boundary_onehot = torchvision.transforms.Compose([
                FrameLabelsToBoundaryOnehot(),
                vak.transforms.ToLongTensor(),
            ])
            self.frame_labels_transform = None
        elif target_type == 'label-binary':
            if labelmap is None:
                raise ValueError(
                    "`target_type` was 'label-binary' but `labelmap` was None. "
                    "Must pass in a `labelmap` with an `unlabeled` class to convert "
                    "multi-class labels to binary labels"
                )
            self.frame_labels_transform = torchvision.transforms.Compose([
                FrameLabelsMultiToBinary(labelmap=labelmap, bg_class_name=bg_class_name),
                vak.transforms.ToLongTensor(),
            ])
            self.to_boundary_onehot = None
        elif target_type == 'label-multi':
            self.frame_labels_transform = vak.transforms.ToLongTensor()
            self.to_boundary_onehot = None
        elif target_type == ('boundary', 'label-multi'):
            self.frame_labels_transform = vak.transforms.ToLongTensor()
            self.to_boundary_onehot = torchvision.transforms.Compose([
                FrameLabelsToBoundaryOnehot(),
                vak.transforms.ToLongTensor(),
            ])
        else:
            raise ValueError(
                f"Invalid `target_type`: {target_type}. "
                "Must be one of: {'boundary', 'label-binary', 'label-multi', ('boundary', 'label-multi')}"
            )

        self.target_type = target_type
        self.labelmap = labelmap
        self.bg_class_name = bg_class_name

    def __call__(self, frames, frame_labels):
        frames = self.frames_transform(frames)
        item = {
            "frames": frames,
        }
        if self.frame_labels_transform is not None:
            item["frame_labels"] = self.frame_labels_transform(frame_labels)
        if self.to_boundary_onehot is not None:
            item["boundary_onehot"] = self.to_boundary_onehot(frame_labels)

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

        self.source_transform = torchvision.transforms.Compose(
            [
                vak.transforms.ToFloatTensor(),
                # for frames we add channel at dim 0, not 1
                vak.transforms.AddChannel(),
            ]
        )

        self.pad_to_window = vak.transforms.PadToWindow(
            self.window_size, frames_padval, return_padding_mask=return_padding_mask
        )

        self.transform_after_pad = torchvision.transforms.Compose(
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
            ]
        )
        self.frame_labels_to_boundary_onehot = FrameLabelsToBoundaryOnehot()

    def __call__(self, frames, frame_labels, frame_times, annot=None, frames_path=None):
        # ---- handle spectrogram
        if self.spect_standardizer:
            frames = self.spect_standardizer(frames)

        if self.pad_to_window.return_padding_mask:
            frames_window, padding_mask = self.pad_to_window(frames)
        else:
            frames_window = self.pad_to_window(frames)
            padding_mask = None
        frames_window = self.transform_after_pad(frames_window)

        # ---- handle frame labels
        boundary_onehot = self.frame_labels_to_boundary_onehot(frame_labels)
        boundary_onehot = self.frame_labels_transform(boundary_onehot)

        item = {
            "frames": frames_window,
            "boundary_onehot": boundary_onehot,
            "frame_times": frame_times,
            "annot": annot,
        }

        if padding_mask is not None:
            item["padding_mask"] = padding_mask

        if frames_path is not None:
            # make sure frames_path is a str, not a pathlib.Path
            item["frames_path"] = str(frames_path)

        return item
