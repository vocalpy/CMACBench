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

    model_name: str = cfg.learncurve.model
    model_config: dict = vak.config.model.config_from_toml_path(toml_path, model_name)
    network_class = MODEL_NAME_NET_CLASS_MAP[model_name]
    network_kwargs = model_config["network"]
    loss_name = model_config["loss"].get("name", "ce")
    loss_class = LOSS_NAME_CLASS_MAP[loss_name]

    # HACK: window_size is a train_dataset_param and a val_transform_param
    window_size = cfg.learncurve.train_dataset_params["window_size"]

    biosoundsegbench.predict.predict_with_frame_classification_model(
        model_name=model_name,
        model_config=model_config,
        dataset_path=cfg.predict.dataset_path,
        checkpoint_path=cfg.predict.checkpoint_path,
        labelmap_path=cfg.predict.labelmap_path,
        num_workers=cfg.predict.num_workers,
        transform_params=cfg.predict.transform_params,
        dataset_params=cfg.predict.dataset_params,
        timebins_key=cfg.predict.timebins_key,
        spect_scaler_path=cfg.predict.spect_scaler_path,
        device=cfg.predict.device,
        annot_csv_filename=cfg.predict.annot_csv_filename,
        output_dir=cfg.predict.output_dir,
        min_segment_dur=cfg.predict.min_segment_dur,
        majority_vote=cfg.predict.majority_vote,
        save_net_outputs=cfg.predict.save_net_outputs,
    )


if __name__ == '__main__':
    parser = vak.__main__.get_parser()
    args = parser.parse_args()
    main(toml_path=args.configfile)
