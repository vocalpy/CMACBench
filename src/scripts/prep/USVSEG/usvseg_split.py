# coding: utf-8
from math import floor
from pathlib import Path
import shutil

HERE = Path(__file__).parent
DATA_DIR = HERE.joinpath('../../data')
USVSEG_DATA = DATA_DIR.joinpath('usvseg')

def main():
    """split USVSEG data set into train and test sets"""
    USVSEG_DATA.joinpath('train').mkdir()
    USVSEG_DATA.joinpath('test').mkdir()

    usv_dirs = sorted([usv_dir for usv_dir in USVSEG_DATA.iterdir() if usv_dir.is_dir()])

    for usv_dir in usv_dirs:
        wavs = sorted(usv_dir.glob('*.wav'))
        n_train = len(wavs) // 2
        wavs_train = wavs[:n_train]
        wavs_test = wavs[n_train:]
        for split, wavs_split in zip(('train', 'test'), (wavs_train, wavs_test)):
            for wav in wavs_split:
                shutil.copy(wav, dst=Path(split).joinpath(wav.parents[0].name + '-' + wav.name))
                csv = wav.parent.joinpath(wav.stem + '.csv')
                shutil.copy(csv, dst=Path(split).joinpath(csv.parents[0].name + '-' + csv.name))


main()
