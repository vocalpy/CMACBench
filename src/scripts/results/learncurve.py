#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import pathlib
import shutil

import torch
import vak

import biosoundsegbench


MODEL_NAME_NET_CLASS_MAP = {
    'ConvTemporalConvNet': biosoundsegbench.ConvTCN,
    'MultiStackTemporalConvNet': biosoundsegbench.MultiStackTCN,
    'SkipTemporalConvNet': biosoundsegbench.SkipTCN,
    'TemporalConvNet': biosoundsegbench.TCN,
    'TweetyNet': vak.nets.TweetyNet,
}

LOSS_NAME_CLASS_MAP = {
    "ce": torch.nn.CrossEntropyLoss,
    "ce-tmse": biosoundsegbench.loss.CrossEntropyWithTruncatedMSE,
    "ce-gstmse": biosoundsegbench.loss.CrossEntropyWithGaussianSimilarityTruncatedMSE,
}


def main(toml_path):
    toml_path = pathlib.Path(toml_path)
    cfg = vak.config.parse.from_toml_path(toml_path)

    # ---- set up directory to save output -----------------------------------------------------------------------------
    results_path = vak.common.paths.generate_results_dir_name_as_path(
        cfg.learncurve.root_results_dir
    )
    results_path.mkdir(parents=True)
    # copy config file into results dir now that we've made the dir
    shutil.copy(toml_path, results_path)

    model_name: str = cfg.learncurve.model
    model_config: dict = vak.config.model.config_from_toml_path(toml_path, model_name)
    network_class = MODEL_NAME_NET_CLASS_MAP[model_name]
    network_kwargs = model_config["network"]
    loss_name = model_config["loss"].get("name", "ce")
    loss_class = LOSS_NAME_CLASS_MAP[loss_name]
    loss_kwargs = model_config["loss"]

    # HACK: window_size is a train_dataset_param and a val_transform_param
    window_size = cfg.learncurve.train_dataset_params["window_size"]

    biosoundsegbench.learncurve.learning_curve(
        dataset_path=cfg.learncurve.dataset_path,
        results_path=results_path,
        model_name=cfg.learncurve.model,
        window_size=window_size,
        batch_size=cfg.learncurve.batch_size,
        num_workers=cfg.learncurve.num_workers,
        network_class=network_class,
        network_kwargs=network_kwargs,
        loss_class=loss_class,
        num_epochs=cfg.learncurve.num_epochs,
        val_step=cfg.learncurve.val_step,
        ckpt_step=cfg.learncurve.ckpt_step,
        patience=cfg.learncurve.patience,
        post_tfm_kwargs=cfg.learncurve.post_tfm_kwargs,
    )


if __name__ == '__main__':
    parser = vak.__main__.get_parser()
    args = parser.parse_args()
    main(toml_path=args.configfile)
