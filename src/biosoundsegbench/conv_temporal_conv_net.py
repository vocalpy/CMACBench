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

from .temporal_conv_net import CausalConv1d, TemporalBlock


class ConvTCN(torch.nn.Module):
    """A hybrid Convolutional-Temporal Convolutional Network,
    that first extracts features from a spectrogram
    with the convolutional blocks,
    then stacks all features across all channels
    and maps those to 1-dimensional time series
    with a 1x1 convolution, that is then fed into
    the Temporal Convolutional Network."""
    def __init__(self, num_classes: int, num_freqbins: int,
                 out_channels: int, num_layers: int,
                 num_input_channels: int = 1, padding="SAME",
                 conv1_filters=32, conv1_kernel_size=(5, 5),
                 conv2_filters=64, conv2_kernel_size=(5, 5),
                 pool1_size=(8, 1), pool1_stride=(8, 1),
                 pool2_size=(8, 1), pool2_stride=(8, 1),
                 kernel_size:int = 2, dropout: float = 0.2):
        super().__init__()

        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.num_freqbins = num_freqbins

        self.cnn = torch.nn.Sequential(
            vak.nn.modules.Conv2dTF(
                in_channels=self.num_input_channels,
                out_channels=conv1_filters,
                kernel_size=conv1_kernel_size,
                padding=padding,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=pool1_size, stride=pool1_stride),
            vak.nn.modules.Conv2dTF(
                in_channels=conv1_filters,
                out_channels=conv2_filters,
                kernel_size=conv2_kernel_size,
                padding=padding,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=pool2_size, stride=pool2_stride),
        )

        N_DUMMY_TIMEBINS = (
            256  # some not-small number. This dimension doesn't matter here
        )
        batch_shape = (
            1,
            self.num_input_channels,
            self.num_freqbins,
            N_DUMMY_TIMEBINS,
        )
        tmp_tensor = torch.rand(batch_shape)
        tmp_out = self.cnn(tmp_tensor)
        cnn_out_channels, freqbins_out = tmp_out.shape[1], tmp_out.shape[2]
        self.in_channels = cnn_out_channels * freqbins_out

        layers = [
            # This first layer uses a 1x1 convolution to
            # map every frequency bin in the spectrogram to a channel
            # from a 1-D convolution.
            CausalConv1d(in_channels=self.in_channels, out_channels=out_channels,
                         kernel_size=1, stride=1, dilation=1)
        ]

        # this is the standard TCN as used in Bai et al. 2018
        for ind in range(num_layers):
            dilation_size = 2 ** ind
            layers.append(
                TemporalBlock(out_channels, out_channels, kernel_size, stride=1,
                              dilation=dilation_size, dropout=dropout)
            )

        self.tcn = torch.nn.Sequential(*layers)
        # We use this fully connected layer to map the convolutional channel features
        # in every time bin of the output of `network` to a class
        self.fc = torch.nn.Linear(in_features=out_channels,
                                  out_features=num_classes)

    def forward(self, x):
        x = self.cnn(x)
        # stack channels, to give tensor shape (batch, tcn input channels, num time bins)
        x = x.view(x.shape[0], self.in_channels, -1)

        x = self.tcn(x)
        # this permutation is necessary in pytorch (but not the original keras implementation of DAS)
        # because pytorch is "channels first" by default.
        x = torch.permute(x, (0, 2, 1))
        x = self.fc(x)
        # Permute again to have (batch, class, time step) as required by loss function.
        # Also, any upsample op expects to operate along the last dimension, so it should be time.
        x = torch.permute(x, (0, 2, 1))
        return x




@model(family=FrameClassificationModel)
class ConvTemporalConvNet:
    """A hybrid Convolutional-Temporal Convolutional Network,
    that first extracts features from a spectrogram
    with the convolutional blocks,
    then stacks all features across all channels
    and maps those to 1-dimensional time series
    with a 1x1 convolution, that is then fed into
    the Temporal Convolutional Network."""
    network = ConvTCN
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
