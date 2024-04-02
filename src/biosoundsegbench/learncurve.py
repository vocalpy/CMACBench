#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import pathlib

import pandas as pd
import vak

from .eval import eval_frame_classification_model
from .train import train_frame_classification_model


def learning_curve(
    dataset_path,
    results_path,
    model_name,
    window_size,
    batch_size,
    num_workers,
    network_class,
    network_kwargs,
    loss_class,
    num_epochs,
    val_step,
    ckpt_step,
    patience,
    post_tfm_kwargs,
):
    dataset_path = pathlib.Path(dataset_path)
    metadata = vak.datasets.frame_classification.Metadata.from_dataset_path(
        dataset_path
    )
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)

    results_path = vak.common.converters.expanded_user_path(results_path)

    dataset_df = dataset_df[
        (dataset_df.train_dur.notna()) & (dataset_df.replicate_num.notna())
    ]
    train_durs = sorted(dataset_df["train_dur"].unique())
    replicate_nums = [
        int(replicate_num)
        for replicate_num in sorted(dataset_df["replicate_num"].unique())
    ]

    to_do = []
    for train_dur in train_durs:
        for replicate_num in replicate_nums:
            to_do.append((train_dur, replicate_num))

    for train_dur, replicate_num in to_do:
        results_path_this_train_dur = results_path / vak.learncurve.dirname.train_dur_dirname(
            train_dur
        )
        if not results_path_this_train_dur.exists():
            results_path_this_train_dur.mkdir()

        results_path_this_replicate = (
            results_path_this_train_dur / vak.learncurve.dirname.replicate_dirname(replicate_num)
        )
        results_path_this_replicate.mkdir()

        # `subset` lets us use correct subset of training set for this duration / replicate
        subset = vak.common.learncurve.get_train_dur_replicate_subset_name(
            train_dur, replicate_num
        )

        train_frame_classification_model(
            dataset_path,
            subset,
            results_path_this_replicate,
            model_name,
            window_size,
            batch_size,
            num_workers,
            network_class,
            network_kwargs,
            loss_class,
            ckpt_step,
            patience,
            val_step,
            num_epochs,
        )

        results_model_root = results_path_this_replicate.joinpath(model_name)
        ckpt_root = results_model_root.joinpath("checkpoints")
        ckpt_paths = sorted(ckpt_root.glob("*.pt"))
        if any([ckpt_path.name.startswith("checkpoint-best") for ckpt_path in ckpt_paths]):
            ckpt_paths = [
                ckpt_path
                for ckpt_path in ckpt_paths
                if ckpt_path.name.startswith("checkpoint-best")
            ]
            if len(ckpt_paths) != 1:
                raise ValueError(
                    f"did not find a single checkpoint-best path, instead found:\n{ckpt_paths}"
                )
            ckpt_path = ckpt_paths[0]
        else:
            if len(ckpt_paths) != 1:
                raise ValueError(
                    f"did not find a single checkpoint path, instead found:\n{ckpt_paths}"
                )
            ckpt_path = ckpt_paths[0]

        labelmap_path = results_path_this_replicate.joinpath("labelmap.json")
        spect_scaler_path = results_path_this_replicate.joinpath(
            "StandardizeSpect"
        )

        eval_frame_classification_model(
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
            ckpt_path,
            results_path_this_replicate,
        )

    # ---- make a csv for analysis -------------------------------------------------------------------------------------
    # use one of the eval csvs just to get columns, that we use below to re-order
    eval_csv_paths = sorted(results_path.glob("**/eval*.csv"))
    eval_df_0 = pd.read_csv(eval_csv_paths[0])
    eval_columns = eval_df_0.columns.tolist()

    eval_dfs = []
    for train_dur, replicate_num in to_do:
        results_path_this_train_dur = results_path / vak.learncurve.dirname.train_dur_dirname(
            train_dur
        )
        results_path_this_replicate = (
            results_path_this_train_dur / vak.learncurve.dirname.replicate_dirname(replicate_num)
        )
        eval_csv_path = sorted(results_path_this_replicate.glob("eval*.csv"))
        if not len(eval_csv_path) == 1:
            raise ValueError(
                "Did not find exactly one eval results csv file in replicate directory after running learncurve. "
                f"Directory is: {results_path_this_replicate}."
                f"Result of globbing directory for eval csv file: {eval_csv_path}"
            )
        eval_csv_path = eval_csv_path[0]
        eval_df = pd.read_csv(eval_csv_path)
        eval_df["train_dur"] = train_dur
        eval_df["replicate_num"] = replicate_num
        eval_dfs.append(eval_df)

    all_eval_df = pd.concat(eval_dfs)
    all_eval_columns = ["train_dur", "replicate_num", *eval_columns]
    all_eval_df = all_eval_df[all_eval_columns]
    all_eval_df.sort_values(by=["train_dur", "replicate_num"])
    learncurve_csv_path = results_path.joinpath("learning_curve.csv")
    all_eval_df.to_csv(
        learncurve_csv_path, index=False
    )  # index=False to avoid adding "Unnamed: 0" column
