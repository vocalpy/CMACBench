#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import torch
import pytorch_lightning as lightning
import vak
import vocalpy as voc

from .loss import CrossEntropyWithGaussianSimilarityTruncatedMSE


class FrameClassificationModel(lightning.LightningModule):
    def __init__(
        self,
        labelmap: Mapping,
        network: torch.nn.Module | dict | None = None,
        loss: torch.nn.Module | Callable | None = None,
        metrics: dict | None = None,
        post_tfm: Callable | None = None,
    ):
        super().__init__()

        self.network = network
        self.loss = loss
        self.optimizer = self.configure_optimizers()
        self.metrics = metrics

        self.labelmap = labelmap
        # replace any multiple character labels in mapping
        # with single-character labels
        # so that we do not affect edit distance computation
        # see https://github.com/NickleDave/vak/issues/373
        labelmap_keys = [lbl for lbl in labelmap.keys() if lbl != "unlabeled"]
        if any(
            [len(label) > 1 for label in labelmap_keys]
        ):  # only re-map if necessary
            # (to minimize chance of knock-on bugs)
            self.eval_labelmap = vak.common.labels.multi_char_labels_to_single_char(
                labelmap
            )
        else:
            self.eval_labelmap = labelmap

        self.to_labels_eval = vak.transforms.frame_labels.ToLabels(
            self.eval_labelmap
        )
        self.post_tfm = post_tfm

    def configure_optimizers(self):
        """Returns the model's optimizer.

        Method required by ``lightning.LightningModule``.
        This method returns the ``optimizer`` instance passed into ``__init__``.
        If None was passed in, an instance that was created
        with default arguments will be returned.
        """
        return self.optimizer

    def training_step(self, batch: tuple, batch_idx: int):
        x, y = batch[0], batch[1]
        out = self.network(x)
        if isinstance(self.loss, CrossEntropyWithGaussianSimilarityTruncatedMSE):
            loss = self.loss(out, y, x)
        else:
            loss = self.loss(out, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        x_window, y = batch["frames_window"], batch["frame_labels"]
        # remove "batch" dimension added by collate_fn to x
        # we keep for y because loss still expects the first dimension to be batch
        # TODO: fix this weirdness. Diff't collate_fn?
        if x_window.ndim in (5, 4):
            if x_window.shape[0] == 1:
                x_window = torch.squeeze(x_window, dim=0)
        else:
            raise ValueError(f"invalid shape for x: {x_window.shape}")

        out = self.network(x_window)
        # permute and flatten out into y_pred
        # so that it has shape (1, number classes, number of time bins)
        # ** NOTICE ** just calling out.reshape(1, out.shape(1), -1) does not work, it will change the data
        out = out.permute(1, 0, 2)
        out = torch.flatten(out, start_dim=1)
        out = torch.unsqueeze(out, dim=0)
        # reduce to predictions, assuming class dimension is 1
        y_pred = torch.argmax(
            out, dim=1
        )  # y_pred has dims (batch size 1, predicted label per time bin)

        if "padding_mask" in batch:
            padding_mask = batch[
                "padding_mask"
            ]  # boolean: 1 where valid, 0 where padding
            # remove "batch" dimension added by collate_fn
            # because this extra dimension just makes it confusing to use the mask as indices
            if padding_mask.ndim == 2:
                if padding_mask.shape[0] == 1:
                    padding_mask = torch.squeeze(padding_mask, dim=0)
            else:
                raise ValueError(
                    f"invalid shape for padding mask: {padding_mask.shape}"
                )

            out = out[:, :, padding_mask]
            y_pred = y_pred[:, padding_mask]

        y_labels = self.to_labels_eval(y.cpu().numpy())
        y_pred_labels = self.to_labels_eval(y_pred.cpu().numpy())

        if self.post_tfm:
            y_pred_tfm = self.post_tfm(
                y_pred.cpu().numpy(),
            )
            y_pred_tfm_labels = self.to_labels_eval(y_pred_tfm)
            # convert back to tensor so we can compute accuracy
            y_pred_tfm = torch.from_numpy(y_pred_tfm).to(self.device)

        # TODO: figure out smarter way to do this
        for metric_name, metric_callable in self.metrics.items():
            if metric_name == "loss":
                # NOTE this inflates the loss since we pass in the entire song/spectrogram
                # instead of windows
                if isinstance(self.loss, CrossEntropyWithGaussianSimilarityTruncatedMSE):
                    x = batch["frames"]  # note "frames" is not padded
                    metric_val = metric_callable(out, y, x)
                else:
                    metric_val = metric_callable(out, y)
                self.log(
                    f"val_{metric_name}",
                    metric_val,
                    batch_size=1,
                    on_step=True,
                    sync_dist=True,
                )
            elif metric_name == "acc":
                self.log(
                    f"val_{metric_name}",
                    metric_callable(y_pred, y),
                    batch_size=1,
                    on_step=True,
                    sync_dist=True,
                )
                if self.post_tfm:
                    self.log(
                        f"val_{metric_name}_tfm",
                        metric_callable(y_pred_tfm, y),
                        batch_size=1,
                        on_step=True,
                        sync_dist=True,
                    )
            elif (
                metric_name == "levenshtein"
                or metric_name == "character_error_rate"
            ):
                metric_val = metric_callable(y_pred_labels, y_labels)
                # next line to avoid bug
                metric_val = torch.tensor(metric_val).float().to(self.device)
                self.log(
                    f"val_{metric_name}",
                    metric_val,
                    batch_size=1,
                    on_step=True,
                    sync_dist=True,
                )
                if self.post_tfm:
                    metric_val = metric_callable(y_pred_tfm_labels, y_labels)
                    # next line to avoid bug
                    metric_val = torch.tensor(metric_val).float().to(self.device)
                    self.log(
                        f"val_{metric_name}_tfm",
                        metric_val,
                        batch_size=1,
                        on_step=True,
                        sync_dist=True,
                    )

        # TODO: write custom collate_fn that avoids need for this
        frame_times = np.squeeze(batch["frame_times"].cpu().numpy())
        onsets_s = np.squeeze(batch["onsets_s"].cpu().numpy())
        offsets_s = np.squeeze(batch["offsets_s"].cpu().numpy())

        # in zebra finch annotation there are cases where there's no silent gaps between
        # syllable "notes" so we need to remove the onsets to avoid crashing the metrics function
        if np.any(
            np.isclose(offsets_s[:-1], onsets_s[1:])
        ):
            offsets_to_remove = np.nonzero(np.isclose(offsets_s[:-1], onsets_s[1:]))[0]
            offsets_s = np.delete(offsets_s, offsets_to_remove)
            reference = np.sort(np.concatenate((onsets_s, offsets_s)))
        else:
            reference = voc.metrics.segmentation.ir.concat_starts_and_stops(onsets_s, offsets_s)

        y_pred_labels_segments, onsets_s_pred, offsets_s_pred = vak.transforms.frame_labels.to_segments(
            np.squeeze(y_pred.cpu().numpy()),
            labelmap=self.labelmap,
            frame_times=frame_times,
        )
        if y_pred_labels_segments is None and onsets_s_pred is None and offsets_s_pred is None:
            # handle the case when all time bins are predicted to be unlabeled
            onsets_s_pred = np.array([])
            offsets_s_pred = np.array([])
        # when we don't apply any post-processing transform, it is possible for the model to predict
        # consecutive segments with no "unlabeled" / "background" class between them,
        # meaning that the offset of the first segment will be the onset of the following segment
        # (as would be the case if we left in the segments representing background classes,
        # but the `to_segment` transform removes them)
        # We want to remove the offsets that overlap with onsets before we concatenate all the boundaries together.
        # We don't do this with the other offsets, because other offsets are actually just
        # "the onset time of the unlabeled segment".
        # This is not an issue when we apply the majority vote transform since it forces a consistent class
        # within each segment, bordered on both sides by the "background" / "unlabeled" segments
        if np.any(
            np.isclose(offsets_s_pred[:-1], onsets_s_pred[1:])
        ):
            offsets_to_remove = np.nonzero(np.isclose(offsets_s_pred[:-1], onsets_s_pred[1:]))[0]
            offsets_s_pred = np.delete(offsets_s_pred, offsets_to_remove)
            hypothesis = np.sort(np.concatenate((onsets_s_pred, offsets_s_pred)))
        else:
            hypothesis = voc.metrics.segmentation.ir.concat_starts_and_stops(onsets_s_pred, offsets_s_pred)

        if self.post_tfm:
            y_pred_tfm_labels_segments, onsets_s_pred_tfm, offsets_s_pred_tfm = vak.transforms.frame_labels.to_segments(
                np.squeeze(y_pred_tfm.cpu().numpy()),
                labelmap=self.labelmap,
                frame_times=frame_times,
            )
            hypothesis_tfm = voc.metrics.segmentation.ir.concat_starts_and_stops(onsets_s_pred_tfm, offsets_s_pred_tfm)

        for metric_name, metric_callable in zip(
            ("precision", "recall", "fscore"),
            (
                voc.metrics.segmentation.ir.precision,
                voc.metrics.segmentation.ir.recall,
                voc.metrics.segmentation.ir.fscore,
            )
        ):
            metric_val = metric_callable(hypothesis, reference, 0.004)[0]
            # next line to avoid bug
            metric_val = torch.tensor(metric_val).float().to(self.device)
            self.log(
                f"val_{metric_name}",
                metric_val,
                batch_size=1,
                on_step=True,
                sync_dist=True,
            )
            if self.post_tfm:
                self.log(
                    f"val_{metric_name}_tfm",
                    metric_callable(hypothesis_tfm, reference, 0.004)[0],
                    batch_size=1,
                    on_step=True,
                    sync_dist=True,
                )

    def configure_optimizers(self):
        return torch.optim.Adam(lr=0.003, params=self.parameters())

    def load_state_dict_from_path(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["state_dict"])
