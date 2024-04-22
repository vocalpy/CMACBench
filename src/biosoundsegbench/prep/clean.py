"""Functions for 'clean' stage of BioSoundSegBench dataset"""
import logging
import shutil

from . import constants


logger = logging.getLogger(__name__)


def clean(dry_run=True):
    """Removes all generated directories:
    BioSoundSegBench and DATA_WE_CANT_SHARE"""
    logger.info(
        f"Removing DATASET_ROOT: {constants.DATASET_ROOT}"
    )
    if not dry_run:
        shutil.rmtree(constants.DATASET_ROOT)

    logger.info(
        f"Removing DATA_WE_CANT_SHARE: {constants.DATA_WE_CANT_SHARE}"
    )
    if not dry_run:
        shutil.rmtree(constants.DATA_WE_CANT_SHARE)

