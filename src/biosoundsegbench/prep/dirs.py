"""Helper functions for setting up directories."""
import logging

from . import constants


logger = logging.getLogger(__name__)


def mkdirs(dry_run=True):
    """Make BioSoundSegBench directories"""
    logger.info(
        f"Making directory DATASET_ROOT: {constants.DATASET_ROOT}"
    )
    if not dry_run:
        constants.DATASET_ROOT.mkdir(exist_ok=True)

    logger.info(
        f"Making directory BF_DATA_DST: {constants.BF_DATA_DST}"
    )
    if not dry_run:
        constants.BF_DATA_DST.mkdir(exist_ok=True)

    logger.info(
        f"Making directory CANARY_DATA_DST: {constants.CANARY_DATA_DST}"
    )
    if not dry_run:
        constants.CANARY_DATA_DST.mkdir(exist_ok=True)

    logger.info(
        f"Making directory MOUSE_PUP_CALL_DATA_DST: {constants.MOUSE_PUP_CALL_DATA_DST}"
    )
    if not dry_run:
        constants.MOUSE_PUP_CALL_DATA_DST.mkdir(exist_ok=True)

    logger.info(
        f"Making directory ZB_DATA_DST: {constants.ZB_DATA_DST}"
    )
    if not dry_run:
        constants.ZB_DATA_DST.mkdir(exist_ok=True)

    logger.info(
        f"Making directory SPEECH_DATA_DST: {constants.SPEECH_DATA_DST}"
    )
    if not dry_run:
        constants.SPEECH_DATA_DST.mkdir(exist_ok=True)

    logger.info(
        f"Making directory DATA_WE_CANT_SHARE: {constants.DATA_WE_CANT_SHARE}"
    )
    if not dry_run:
        constants.DATA_WE_CANT_SHARE.mkdir(exist_ok=True)

    logger.info(
        f"Making directory HUMAN_SPEECH_WE_CANT_SHARE: {constants.HUMAN_SPEECH_WE_CANT_SHARE}"
    )
    if not dry_run:
        constants.HUMAN_SPEECH_WE_CANT_SHARE.mkdir(exist_ok=True)

    logger.info(
        f"Making directory SPLITS_DIR: {constants.SPLITS_DIR}"
    )
    if not dry_run:
        constants.SPLITS_DIR.mkdir(exist_ok=True)
