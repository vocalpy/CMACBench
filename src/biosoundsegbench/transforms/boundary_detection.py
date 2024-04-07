import numpy as np
import torchvision
import vak

from .transforms import FrameLabelsToBoundaryOnehot, ViewAsWindowBatch


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
