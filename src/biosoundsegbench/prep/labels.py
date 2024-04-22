"""Make labelset + labelmaps used to filter data"""
import json
import logging

import crowsetta
import vak

from . import constants


logger = logging.getLogger(__name__)


# we write these out in code for now because it's easier
# but long term we will want this as metadata in static files, not code.
# We dump them so we can get them later from a static file
SPECIES_ID_LABELSTR_MAP = {
    'Bengalese-Finch-Song': {
        'syllable': {
            'bl26lb16': "iabcdef",
            'gr41rd51': "iabcdefgjkm",
            'gy6or6': "iabcdefghjk",
            'or60yw70': "iabcdefg",
            'Bird0': "0123456789",
            'Bird4': "01234567",
            'Bird7': "0123456",
            'Bird9': "012345",
        },
    },
    'Canary-Song': {
        'syllable': {
            'llb3': "range: 1-20",
            'llb11': "range: 1-30",
            'llb16': "range: 1-30",
        },
    },
    'Zebra-Finch-Song': {
        'syllable': {
            'blu285': ['syll_0', 'syll_1', 'syll_2', 'syll_3', 'syll_4', 'syll_5']
        },
    }
}


def make_labelsets_from_constant(dry_run=True):
    """Make labelsets from module-level constant SPECIES_ID_LABELSTR_MAP"""
    species_id_labelsets_map = {}
    for species in SPECIES_ID_LABELSTR_MAP.keys():
        species_id_labelsets_map[species] = {}
        for unit in SPECIES_ID_LABELSTR_MAP[species].keys():
            id_labelset_map = SPECIES_ID_LABELSTR_MAP[species][unit]
            id_labelset_map = {
                # we convert the set to a list so we can dump to json
                id: list(
                    vak.common.converters.labelset_to_set(labelset)
                )
                for id, labelset in id_labelset_map.items()
            }
        species_id_labelsets_map[species][unit] = id_labelset_map

    return species_id_labelsets_map


SCRIBE = crowsetta.Transcriber(format="simple-seq")


def make_timit_labelsets_from_annot_files():
    """Make TIMIT labelsets from phoneme annotation files.

    Instead of computing per-speaker ID labelset,
    for human speech we assume the same set of phoneme classes
    for all speakers.
    """
    TIMIT_DIALECT_SPKR_DIRS = [
        dir_ for dir_ in constants.HUMAN_SPEECH_WE_CANT_SHARE.iterdir()
        if dir_.is_dir()
    ]

    id_labelset_map = {}

    labels = []
    for dir_ in TIMIT_DIALECT_SPKR_DIRS:
        speaker_id = dir_.name.split('-')[-1]
        logger.info(
            f"Computing labelset for TIMIT speaker ID: {speaker_id}"
        )
        csv_paths = sorted(dir_.glob(f"*.phoneme.csv"))
        labels.extend(
            [lbl
             for csv_path in csv_paths
             for lbl in SCRIBE.from_file(csv_path).to_seq().labels]
        )
    # next line, set to find unique labels; list so we can dump to json
    return list(set(labels))


def set_to_map(species_id_labelsets_map):
    """Convert sets of labels to maps,
    that map from labels to consecutive integers"""
    species_id_labelmap_map = {}
    for species in species_id_labelsets_map.keys():
        species_id_labelmap_map[species]= {}
        for unit in species_id_labelsets_map[species].keys():
            id_labelset_map = species_id_labelsets_map[species][unit]
            id_labelmap_map = {
                id: vak.common.labels.to_map(
                    # we need to convert from list back to set when loading from json
                    set(labelset),
                    map_unlabeled=True,
                )
                for id, labelset in id_labelset_map.items()
            }
            species_id_labelmap_map[species][unit] = id_labelmap_map
    return species_id_labelmap_map


def make_labelsets_and_labelmaps(dry_run=True):
    """Make labelsets and labelmaps from labelsets,
    then save as json files in BioSoundSegBench dataset root.

    These have two purposes:
    1. Labelsets are used to filter files so we keep only those
    that have labels in the labelset, i.e. closed-set classification.
    2. Labelmaps are used when training models, to determine the number
    of integer output classes, or at inference time, to map integer outputs
    back to string labels.
    """
    logger.info(
        f"Making labelsets from module-level constant SPECIES_ID_LABELSTR_MAP"
    )
    species_id_labelsets_map = make_labelsets_from_constant()
    logger.info(
        f"Making labelsets for human speech, from TIMIT dataset phoneme annotations"
    )
    species_id_labelsets_map['Human-Speech'] = {}
    species_id_labelsets_map['Human-Speech']['phoneme'] = {}
    phoneme_labelset = make_timit_labelsets_from_annot_files()
    # we assume same set of classes for all speakers
    species_id_labelsets_map['Human-Speech']['phoneme']['all'] = phoneme_labelset

    logger.info(
        f"Final species_id_labelsets_map:\n{species_id_labelsets_map}"
    )

    if not dry_run:
        with open(constants.LABELSETS_JSON, "w") as fp:
            json.dump(species_id_labelsets_map, fp, indent=4)

    logger.info(
        f"Converting labelsets to labelmaps."
    )
    species_id_labelmaps_map = set_to_map(species_id_labelsets_map)
    logger.info(
        f"Final species_id_labelmap_map:\n{species_id_labelmaps_map}"
    )
    if not dry_run:
        with open(constants.LABELMAPS_JSON, "w") as fp:
            json.dump(species_id_labelmaps_map, fp, indent=4)


def get_labelsets():
    with open(constants.LABELSETS_JSON, "r") as fp:
        return json.load(fp)


def get_labelmaps():
    with open(constants.LABELMAPS_JSON, "r") as fp:
        return json.load(fp)
