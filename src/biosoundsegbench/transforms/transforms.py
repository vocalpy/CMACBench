from __future__ import annotations

from typing import Mapping

import numpy as np

from . import functional as F


class FrameLabelsToBoundaryOnehot:
    """Converts vector of frame labels to one-hot vector
    indicating "boundary" (1) or "not a boundary" (0).
    """
    def __call__(self, frame_labels):
        # a boundary occurs in frame labels
        # wherever the first-order difference is not 0.
        # The actually change occurs at :math:`i`=np.diff(frame_labels) + 1,
        # but we shift everything to the right by 1 when we add a first index indicating a boundary.
        # This first index we add is the onset of the "first" segment -- typically will be background class.
        boundary_frame_labels = (np.diff(frame_labels, axis=0) != 0).astype(int)
        boundary_frame_labels = np.insert(boundary_frame_labels, 0, 1)

        return boundary_frame_labels


class FrameLabelsMultiToBinary:
    """Converts vector of frame labels with multiple classes
    to a vector for binary classification.
    """
    def __init__(self, labelmap: Mapping, bg_class_name='unlabeled'):
        try:
            self.bg_class_int = labelmap[bg_class_name]
        except KeyError as e:
            raise KeyError(
                f"The background class name `bg_class_name={bg_class_name}` "
                f"is not a key in `labelmap`; keys in `labelmap` are: {list(labelmap.keys())}"
            ) from e
        self.labelmap = labelmap
        self.bg_class_name = bg_class_name

    def __call__(self, frame_labels):
        return (frame_labels != self.bg_class_int).astype(int)


class ViewAsWindowBatch:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, arr):
        return F.view_as_window_batch(arr, self.window_size)
