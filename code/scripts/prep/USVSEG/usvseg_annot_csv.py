# coding: utf-8
from argparse import ArgumentParser
from pathlib import Path

from crowsetta import Sequence, Annotation, Transcriber
import numpy as np
import pandas as pd


USV_LABEL = 's'


def main(split_root,
         csv_filename):
    csvs = sorted(Path(split_root).glob('*.csv'))

    annots = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df = df.dropna(axis=1)
        assert df.shape[1] == 2, f'did not have two columns after dropping NaN columns from {csv}'
        df.columns = ['onsets_s', 'offsets_s']
        onsets_s, offsets_s = df.onsets_s.values, df.offsets_s.values
        labels = np.array([USV_LABEL] * onsets_s.shape[0])
        seq = Sequence.from_keyword(onsets_s=onsets_s, offsets_s=offsets_s, labels=labels)
        audio_path = csv.parent.joinpath(csv.stem + '.wav')
        annot = Annotation(seq=seq, annot_path=csv, audio_path=audio_path)
        annots.append(annot)

    scribe = Transcriber(format='csv')
    scribe.to_csv(annots, csv_filename)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('split_root', default='./data/usvseg/train',
                        help='root of dataset split for which annotation .csv should be made')
    parser.add_argument('csv_filename', default='./results/usvseg/train_annot.csv',
                        help='filename / path of .csv that is saved with annotation for split')
    return parser



parser = get_parser()
args = parser.parse_args()
main(split_root=args.split_root, csv_filename=args.csv_filename)
