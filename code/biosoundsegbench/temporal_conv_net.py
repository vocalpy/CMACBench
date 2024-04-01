#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import torch
import vak
from vak import (
    metrics,
    nets
)
from vak.models.frame_classification_model import FrameClassificationModel
from vak.models.decorator import model


class CausalConv1d(torch.nn.Module):
    """Causal 1-d Convolution.

    Attributes
    ----------
    in_channels: int
        Number of channels in the input.
    out_channels: int
        Number of channels produced by the convolution.
    kernel_size: int, tuple
        Size of the convolving kernel,
    stride: int or tuple, optional
        Stride of the convolution. Default: 1.
    dilation: int or tuple, optional
        Spacing between kernel elements. Default: 1.
    groups: int, optional
        Number of blocked connections from input
        channels to output channels. Default: 1.
    bias: bool, optional
        If ``True``, adds a learnable bias to the output.
        Default: ``True``axis.

    Notes
    -----
    Note there is no padding parameter,
    since we determine how to left-pad the input
    to produce causal convolutions
    according to the dilation value and kernel size.

    This implements causal padding as in `keras`,
    since this is what is used by the TCN implementation:
    https://github.com/janclemenslab/das/blob/9ea349f13bdde7f44ba0506c5601078b2617cb45/src/das/tcn/tcn.py#L56
    See: https://theblog.github.io/post/convolution-in-autoregressive-neural-networks/
    The original Bai et al. 2018 paper instead pads both sides,
    then removes extra padding on the right with a `Chomp1D` operation.
    The left padding is slightly more efficient.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 device: str | None = None,
                 dtype: str | None = None,
                 weight_norm: bool = False):
        super().__init__()
        self.leftpad = (kernel_size - 1) * dilation
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                                     stride=stride, padding='valid', dilation=dilation,
                                     groups=groups, bias=bias, device=device, dtype=dtype)
        if weight_norm:
            self.conv1 = torch.nn.utils.weight_norm(self.conv1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a causal convolution.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        out : torch.Tensor
            Output of convolution operation.
        """
        x = torch.nn.functional.pad(x, (self.leftpad, 0))  # left pad last dimension, time
        return self.conv1(x)


# Adapted from https://github.com/locuslab/TCN/tree/master under MIT license
class TemporalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size,
                                  stride=stride, dilation=dilation, weight_norm=True)

        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size,
                                  stride=stride, dilation=dilation, weight_norm=True)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
        self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.conv1.weight.data.normal_(0, 0.01)
        self.conv2.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# TODO: namespace me so we can have model be same name as network
class TCN(torch.nn.Module):
    """A simple Temporal Convolutional Network
    with an initial (1x1) convolution applied to
    a spectrogram, that converts each frequency bin
    in the input into a channel in the output
    with the same number of time steps.
    """
    def __init__(self, num_classes: int, num_freqbins: int,
                 out_channels: int, num_layers: int,
                 num_input_channels: int = 1,  # not used, dummy arg
                 kernel_size:int = 2, dropout: float = 0.2):
        super().__init__()
        layers = [
            # This first layer uses a 1x1 convolution to
            # map every frequency bin in the spectrogram to a channel
            # from a 1-D convolution.
            CausalConv1d(in_channels=num_freqbins, out_channels=out_channels,
                         kernel_size=1, stride=1, dilation=1)
        ]

        # this is the standard TCN as used in Bai et al. 2018
        for ind in range(num_layers):
            dilation_size = 2 ** ind
            layers.append(
                TemporalBlock(out_channels, out_channels, kernel_size, stride=1,
                              dilation=dilation_size, dropout=dropout)
            )

        self.network = torch.nn.Sequential(*layers)
        # We use this fully connected layer to map the convolutional channel features
        # in every time bin of the output of `network` to a class
        self.fc = torch.nn.Linear(in_features=out_channels,
                                  out_features=num_classes)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)  # get rid of channel dim, if there is one
        x = self.network(x)
        # this permutation is necessary in pytorch (but not the original keras implementation of DAS)
        # because pytorch is "channels first" by default.
        x = torch.permute(x, (0, 2, 1))
        x = self.fc(x)
        # Permute again to have (batch, class, time step) as required by loss function.
        # Also, any upsample op expects to operate along the last dimension, so it should be time.
        x = torch.permute(x, (0, 2, 1))
        return x


@model(family=FrameClassificationModel)
class TemporalConvNet:
    """A simple Temporal Convolutional Network
    with an initial (1x1) convolution applied to
    a spectrogram, that converts each frequency bin
    in the input into a channel in the output
    with the same number of time steps."""
    network = TCN
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'character_error_rate': metrics.CharacterErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        'optimizer':
            {'lr': 0.003}
    }


if __name__ == '__main__':
    parser = vak.__main__.get_parser()
    args = parser.parse_args()
    vak.cli.cli.cli(command=args.command, config_file=args.configfile)




