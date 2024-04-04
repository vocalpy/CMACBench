#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import json
import pathlib

import joblib
import pandas as pd
import torch
import pytorch_lightning as lightning
import vak

from .model import FrameClassificationModel


def eval_frame_classification_model(
    dataset_path,
    labelmap_path,
    spect_scaler_path,
    model_name,
    window_size,
    num_workers,
    post_tfm_kwargs,
    network_class,
    network_kwargs,
    loss_class,
    loss_kwargs,
    checkpoint_path,
    output_dir,
):
    dataset_path = pathlib.Path(dataset_path)
    metadata = vak.datasets.frame_classification.Metadata.from_dataset_path(
        dataset_path
    )

    timenow = datetime.now().strftime("%y%m%d_%H%M%S")

    with labelmap_path.open("r") as f:
        labelmap = json.load(f)

    spect_standardizer = joblib.load(spect_scaler_path)

    transform_params = {
        "spect_standardizer": spect_standardizer,
        "window_size": window_size,
    }
    item_transform = vak.transforms.defaults.get_default_transform(
        model_name, "eval", transform_params
    )
    val_dataset = vak.datasets.frame_classification.FramesDataset.from_dataset_path(
        dataset_path=dataset_path,
        split="test",
        item_transform=item_transform,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        # batch size 1 because each spectrogram reshaped into a batch of windows
        batch_size=1,
        num_workers=num_workers,
    )

    frame_dur = metadata.frame_dur
    post_tfm = vak.transforms.frame_labels.PostProcess(
        timebin_dur=frame_dur,
        **post_tfm_kwargs,
    )

    input_shape = val_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]
    network = network_class(
        num_classes=len(labelmap),
        num_freqbins=input_shape[1],
        num_input_channels=input_shape[0],
        **network_kwargs
    )

    loss = loss_class(**loss_kwargs)

    metrics = {
        "acc": vak.metrics.Accuracy(),
        "levenshtein": vak.metrics.Levenshtein(),
        "character_error_rate": vak.metrics.CharacterErrorRate(),
        "loss": loss,
    }

    model = FrameClassificationModel(
        network=network,
        loss=loss,
        metrics=metrics,
        labelmap=labelmap,
        post_tfm=post_tfm
    )
    model.load_state_dict_from_path(checkpoint_path)

    trainer_logger = lightning.loggers.TensorBoardLogger(save_dir=output_dir)
    trainer = lightning.Trainer(accelerator="gpu", logger=trainer_logger)
    # TODO: check for hasattr(model, test_step) and if so run test
    # below, [0] because validate returns list of dicts, length of no. of val loaders
    metric_vals = trainer.validate(model, dataloaders=val_loader)[0]
    metric_vals = {f"avg_{k}": v for k, v in metric_vals.items()}
    for metric_name, metric_val in metric_vals.items():
        if metric_name.startswith("avg_"):
            print(f"{metric_name}: {metric_val:0.5f}")

    # create a "DataFrame" with just one row which we will save as a csv;
    # the idea is to be able to concatenate csvs from multiple runs of eval
    row = OrderedDict(
        [
            ("model_name", model_name),
            ("checkpoint_path", checkpoint_path),
            ("labelmap_path", labelmap_path),
            ("spect_scaler_path", spect_scaler_path),
            ("dataset_path", dataset_path),
        ]
    )
    # TODO: is this still necessary after switching to Lightning? Stop saying "average"?
    # order metrics by name to be extra sure they will be consistent across runs
    row.update(
        sorted(
            [(k, v) for k, v in metric_vals.items() if k.startswith("avg_")]
        )
    )

    # pass index into dataframe, needed when using all scalar values (a single row)
    # throw away index below when saving to avoid extra column
    eval_df = pd.DataFrame(row, index=[0])
    eval_csv_path = output_dir.joinpath(f"eval_{model_name}_{timenow}.csv")
    eval_df.to_csv(
        eval_csv_path, index=False
    )  # index is False to avoid having "Unnamed: 0" column when loading