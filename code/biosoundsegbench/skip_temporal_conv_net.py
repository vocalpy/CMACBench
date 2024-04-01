#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import torch
import torch.nn.functional as F
import vak
from vak import (
    metrics,
    nets
)
from vak.models.frame_classification_model import FrameClassificationModel
from vak.models.decorator import model


from .temporal_conv_net import CausalConv1d

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
        return self.relu(out + res), out  # ``out`` is the skip output


# TODO: namespace me so we can have model be same name as network
class SkipTCN(torch.nn.Module):
    """A simple Temporal Convolutional Network
    with an initial (1x1) convolution applied to
    a spectrogram, that converts each frequency bin
    in the input into a channel in the output
    with the same number of time steps.
    This version includes skip connections
    like those in DAS (Steinfath et al 2022),
    SELD-TCN (Guirguise et al), and WaveNet.
    """
    def __init__(self, num_classes: int, num_freqbins: int,
                 out_channels: int, num_layers: int,
                 num_input_channels: int = 1,  # not used, dummy arg
                 kernel_size:int = 2, dropout: float = 0.2):
        super().__init__()
        # This first layer uses a 1x1 convolution to
        # map every frequency bin in the spectrogram to a channel
        # from a 1-D convolution.
        self.conv1d_1 = CausalConv1d(in_channels=num_freqbins, out_channels=out_channels,
                                     kernel_size=1, stride=1, dilation=1)

        self.tcn_layers = torch.nn.ModuleList()
        # this is the standard TCN as used in Bai et al. 2018
        for ind in range(num_layers):
            dilation_size = 2 ** ind
            self.tcn_layers.append(
                TemporalBlock(out_channels, out_channels, kernel_size, stride=1,
                              dilation=dilation_size, dropout=dropout)
            )

        # We use this fully connected layer to map the convolutional channel features
        # in every time bin of the output of `network` to a class
        self.fc = torch.nn.Linear(in_features=out_channels,
                                  out_features=num_classes)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)  # get rid of channel dim, if there is one
        x = self.conv1d_1(x)  # map frequency bins in spectrograms to conv channels
        skip_connections = []
        for layer in self.tcn_layers:
            x, skip_out = layer(x)
            skip_connections.append(skip_out)
        # WaveNet-like summing of skip connections to get output
        x = torch.sum(torch.stack(skip_connections), dim=0)
        x = F.relu(x)
        # this permutation is necessary in pytorch
        # because it is "channels first" by default.
        x = torch.permute(x, (0, 2, 1))
        x = self.fc(x)
        # Permute again to have (batch, class, time step) as required by loss function.
        # Also, any upsample op expects to operate along the last dimension, so it should be time.
        x = torch.permute(x, (0, 2, 1))
        return x


@model(family=FrameClassificationModel)
class SkipTemporalConvNet:
    """A simple Temporal Convolutional Network
    with an initial (1x1) convolution applied to
    a spectrogram, that converts each frequency bin
    in the input into a channel in the output
    with the same number of time steps.
    This version includes skip connections
    like those in DAS (Steinfath et al 2022),
    SELD-TCN (Guirguise et al), and WaveNet.
    """
    network = SkipTCN
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




