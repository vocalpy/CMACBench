import numpy as np


def view_as_window_batch(arr, window_width):
    """This transform is built into vak but there's a bug where it won't work for 1-D arrays"""
    if not isinstance(window_width, int) or window_width < 1:
        raise ValueError(
            f"`window_width` must be a positive integer, but was: {window_width}"
        )

    if arr.ndim == 1:
        window_shape = (window_width,)
    elif arr.ndim == 2:
        height, _ = arr.shape
        window_shape = (height, window_width)
    else:
        raise ValueError(
            f"input array must be 1d or 2d but number of dimensions was: {arr.ndim}"
        )

    window_shape = np.array(window_shape)
    arr_shape = np.array(arr.shape)
    if (arr_shape % window_shape).sum() != 0:
        raise ValueError(
            "'window_width' does not divide evenly into with 'arr' shape. "
            "Use 'pad_to_window' transform to pad array so it can be windowed."
        )

    new_shape = tuple(arr_shape // window_shape) + tuple(window_shape)
    new_strides = tuple(arr.strides * window_shape) + arr.strides
    batch_windows = np.lib.stride_tricks.as_strided(
        arr, shape=new_shape, strides=new_strides
    )

    if arr.ndim == 2: # avoids bug in vak version of transform, by not doing this to 1-D arrays
        # TODO: figure out if there's a better way to do this where we don't need to squeeze
        # The current version always add an initial dim of size 1
        batch_windows = np.squeeze(batch_windows, axis=0)
        # By squeezing just that first axis, we always end up with (batch, freq. bins, time bins) for a spectrogram

    return batch_windows