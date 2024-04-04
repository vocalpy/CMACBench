#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import json
import pathlib

import joblib
import pandas as pd
import torch
import pytorch_lightning as lightning
import vak

from .model import FrameClassificationModel


def train_frame_classification_model(
    dataset_path,
    subset,
    results_path,
    model_name,
    window_size,
    batch_size,
    num_workers,
    network_class,
    network_kwargs,
    loss_class,
    loss_kwargs,
    ckpt_step,
    patience,
    val_step,
    num_epochs,
):
    dataset_path = pathlib.Path(dataset_path)
    metadata = vak.datasets.frame_classification.Metadata.from_dataset_path(
        dataset_path
    )
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)

    results_path = vak.common.converters.expanded_user_path(results_path)

    labelmap_path = dataset_path / "labelmap.json"
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)
    # copy to new results_path
    with open(results_path.joinpath("labelmap.json"), "w") as f:
        json.dump(labelmap, f)

    spect_standardizer = vak.transforms.StandardizeSpect.fit_dataset_path(
        dataset_path,
        "train",
        subset,
    )
    joblib.dump(
        spect_standardizer, results_path.joinpath("StandardizeSpect")
    )

    train_transform_params = {"spect_standardizer": spect_standardizer}
    transform, target_transform = vak.transforms.defaults.get_default_transform(
        model_name, "train", transform_kwargs=train_transform_params
    )

    train_dataset = vak.datasets.frame_classification.WindowDataset.from_dataset_path(
        dataset_path=dataset_path,
        split="train",
        transform=transform,
        target_transform=target_transform,
        window_size=window_size,
        subset=subset,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_transform_params = {
        'window_size': window_size,
        "spect_standardizer": spect_standardizer
    }
    item_transform = vak.transforms.defaults.get_default_transform(
        model_name, "eval", val_transform_params
    )

    val_dataset = vak.datasets.frame_classification.FramesDataset.from_dataset_path(
        dataset_path=dataset_path,
        split="val",
        item_transform=item_transform,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        # batch size 1 because each spectrogram reshaped into a batch of windows
        batch_size=1,
        num_workers=num_workers,
    )

    network = network_class(
        num_classes=len(labelmap),
        num_freqbins=train_dataset.shape[1],
        num_input_channels=train_dataset.shape[0],
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
        labelmap=labelmap
    )

    results_model_root = results_path.joinpath(model_name)
    results_model_root.mkdir()
    ckpt_root = results_model_root.joinpath("checkpoints")
    ckpt_root.mkdir()

    ckpt_callback = lightning.callbacks.ModelCheckpoint(
        dirpath=ckpt_root,
        filename="checkpoint",
        every_n_train_steps=ckpt_step,
        save_last=True,
        verbose=True,
    )
    ckpt_callback.CHECKPOINT_NAME_LAST = "checkpoint"
    ckpt_callback.FILE_EXTENSION = ".pt"

    val_ckpt_callback = lightning.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=ckpt_root,
        save_top_k=1,
        mode="max",
        filename="checkpoint-best",
        auto_insert_metric_name=False,
        verbose=True,
    )
    val_ckpt_callback.FILE_EXTENSION = ".pt"

    early_stopping = lightning.callbacks.EarlyStopping(
        mode="max",
        monitor="val_acc",
        patience=patience,
        verbose=True,
    )
    callbacks = [ckpt_callback, val_ckpt_callback, early_stopping]

    logger = lightning.loggers.TensorBoardLogger(save_dir=results_model_root)

    max_steps = num_epochs * len(train_loader)
    trainer = lightning.Trainer(
        callbacks=callbacks,
        val_check_interval=val_step,
        max_steps=max_steps,
        accelerator="gpu",
        logger=logger,
    )
    trainer.fit(model, train_loader, val_loader)