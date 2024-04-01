"""Multi-stack temporal convolutional network"""
#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
import vak
from vak import (
    metrics,
)
from vak.models.frame_classification_model import FrameClassificationModel
from vak.models.decorator import model

from .temporal_conv_net import CausalConv1d


# Adapted from https://github.com/locuslab/TCN/tree/master under MIT license
class TemporalBlockWithSkip(torch.nn.Module):
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
        # WaveNet-style parametrized skip connection
        self.skip = torch.nn.Conv1d(out_channels, out_channels, 1)

        self.net = torch.nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2,
            self.skip,
        )

        self.relu = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.conv1.weight.data.normal_(0, 0.01)
        self.conv2.conv1.weight.data.normal_(0, 0.01)
        self.skip.weight.data.normal_(0, 0.01)

    def forward(self, x):
        skip_out = self.net(x)
        residual = self.relu(skip_out + x)
        return residual, skip_out


class TCNStacks(torch.nn.Module):
    """A stack of TCNs with optional skip connections"""

    def __init__(self,
                 num_inputs: int,
                 out_channels: int = 64,
                 num_stacks: int = 1,
                 kernel_size: int = 2,
                 dropout: float = 0.0,
                 stride: int | tuple = 1,
                 dilations: Sequence[int] = (1, 2, 4, 8, 16,),
                 use_skip_connections: bool = True,
                 ):
        super().__init__()
        self.num_stacks = num_stacks
        self.dilations = dilations
        self.use_skip_connections = use_skip_connections

        # Note this first layer is (1x1), has the effect of making output be (num time bins x num time bins)
        self.conv1 = CausalConv1d(in_channels=num_inputs, out_channels=out_channels,
                                  kernel_size=1, stride=1, dilation=1)

        self.tcn_layers = torch.nn.ModuleList()
        for _ in range(num_stacks):
            for layer_ind, dilation in enumerate(dilations):
                self.tcn_layers.append(
                    TemporalBlockWithSkip(out_channels, out_channels,
                                          kernel_size, stride=stride,
                                          dilation=dilation, dropout=dropout)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        if self.use_skip_connections:
            skip_connections = []
        else:
            skip_connections = None

        for tcn_layer in self.tcn_layers:
            x, skip_out = tcn_layer(x)
            if skip_connections:
                skip_connections.append(skip_out)

        if skip_connections:
            x = torch.add(*skip_connections)
            x = F.relu(x)

        return x


class MultiStackTCN(torch.nn.Module):
    """
    """
    def __init__(self,
                 num_classes: int,
                 num_freqbins: int,
                 num_input_channels: int = 1,
                 out_channels: int = 64,
                 kernel_size: int = 2,
                 num_stacks: int = 2,
                 use_skip_connections: bool = True,
                 dilations: tuple[int] = (1, 2, 4, 8, 16,),
                 dropout: float = 0.2,
                 ):
        super().__init__()

        # add config params as attribs so we can inspect
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.num_freqbins = num_freqbins
        self.num_stacks = num_stacks
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.dropout = dropout

        self.tcn = TCNStacks(num_freqbins, out_channels, num_stacks,
                             kernel_size, dropout,
                             dilations=dilations,
                             use_skip_connections=use_skip_connections)
        self.fc = torch.nn.Linear(in_features=out_channels,
                                  out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(x, dim=1)  # get rid of channel dim, if there is one
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
class MultiStackTemporalConvNet:
    """"""

    network = MultiStackTCN
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
