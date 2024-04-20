"""
Helper functinos for stage 1 of making BioSoundSegBench dataset:
1. copy raw audio
2. copy annotations if they are in the SimpleSeq format already;
convert annotations if they are in another format;
generate placeholder annotations if we need to.
"""
import logging
import shutil

import tqdm

from . import constants


logger = logging.getLogger(__name__)


def copy_wav_csv_files_bfsongrepo(dry_run=True):
    for bird_id in constants.BFSONGREPO_BIRD_IDS:
        logger.info(
            f"Copying audio and annotation files for bird ID {bird_id} from BFSongRepo"
        )
        bird_id_dir = constants.BFSONGREPO_RAW_DATA / bird_id
        assert bird_id_dir.exists(), f"Couldn't find bird_id_dir: {bird_id_dir}"

        bird_id_dst = constants.BF_DATA_DST / f"bfsongrepo-{bird_id}"
        bird_id_dst.mkdir(exist_ok=True)

        # bfsongrepo is split up by days; we throw away that split here
        day_dirs = [
            subdir for subdir in bird_id_dir.iterdir() if subdir.is_dir()
        ]
        for day_dir in day_dirs:
            logger.info(
                f"Copying audio and annotation files from day: {day_dir.name}"
            )
            wav_paths = sorted(day_dir.glob("*wav"))
            logger.info(
                f"Found {len(wav_paths)} audio files."
            )
            csv_paths = sorted(day_dir.glob("*csv"))
            logger.info(
                f"Found {len(csv_paths)} annotation files."
            )
            wav_paths_filtered = []
            csv_paths_filtered = []
            for wav_path in wav_paths:
                csv_path_that_should_exist = wav_path.parent / (wav_path.name + '.csv')
                if csv_path_that_should_exist in csv_paths:
                    wav_paths_filtered.append(wav_path)
                    csv_paths_filtered.append(csv_path_that_should_exist)
            logger.info(
                "After filtering audio files by whether or not they have an annotation file, "
                f"there are {len(wav_paths_filtered)} audio files and {len(csv_paths_filtered)} annotation files."
            )
            pbar = tqdm.tqdm(
                zip(wav_paths_filtered, csv_paths_filtered)
            )
            for wav_path, csv_path in pbar:
                pbar.set_description(f"Copying: {wav_path.name}")
                if not dry_run:
                    shutil.copy(wav_path, bird_id_dst)
                # NOTE we rename to include the unit in the annotation file name
                csv_path_dst = bird_id_dst / (csv_path.stem + '.syllable.csv')
                pbar.set_description(f"Copying/renaming: {csv_path_dst.name}")
                if not dry_run:
                    shutil.copy(csv_path, csv_path_dst)


def copy_audio_copy_make_annot_all(dry_run=True):
    copy_wav_csv_files_bfsongrepo(dry_run=True)