import numpy as np
import torchvision
import vak

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
        boundary_onehot = (np.diff(frame_labels, axis=0) != 0).astype(int)
        boundary_onehot = np.insert(boundary_onehot, 0, 1)

        return boundary_onehot


class ViewAsWindowBatch:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, arr):
        return F.view_as_window_batch(arr, self.window_size)
