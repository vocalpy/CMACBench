"""Make labelset + labelmaps used to filter data"""
import json
import logging

import crowsetta
import pandera.errors
import vak
from tqdm import tqdm

from . import constants


logger = logging.getLogger(__name__)


BUCKEYE_PHONES = [
    # this is the unique set of phone labels **across all IDs** in the Buckeye corpus
    # that remain after the filtering done by the function in `copy_audio_copy_make_annot.py`
    'aa', 'ae', 'ah', 'ah ', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'e', 'eh', 'el', 'em', 'eng',
    'er', 'ey', 'f', 'g', 'h', 'hh', 'i', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'r',
    's', 'sh', 't', 'th', 'tq', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'
]

# we write these out in code for now because it's easier
# but long term we will want this as metadata in static files, not code.
# We dump them so we can get them later from a static file
GROUP_UNIT_ID_LABELSTR_MAP = {
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
            'llb11': "range: 1-27",
            'llb16': "range: 1-30",
        },
    },
    'Mouse-Pup-Call': {
        'call': {
            'all': ['BK', 'BW', 'GO', 'LL', 'LO', 'MU', 'MZ', 'NB', 'PO', 'SW']
        },
    },
    'Zebra-Finch-Song': {
        'syllable': {
            'blu285': ['syll_0', 'syll_1', 'syll_2', 'syll_3', 'syll_4', 'syll_5']
        },
    },
}





def make_labelsets(dry_run=True):
    """Make sets of unique labels for each ID in each group
    
    For now all groups use module-level constant GROUP_UNIT_ID_LABELSTR_MAP
    except for human speech, where we compute the per-ID labelsets
    from the actual annotation files
    """
    group_unit_id_labelsets_map = {}
    for species in GROUP_UNIT_ID_LABELSTR_MAP.keys():
        group_unit_id_labelsets_map[species] = {}
        for unit in GROUP_UNIT_ID_LABELSTR_MAP[species].keys():
            id_labelset_map = GROUP_UNIT_ID_LABELSTR_MAP[species][unit]
            id_labelset_map = {
                # we convert the set to a list so we can dump to json
                id: list(
                    vak.common.converters.labelset_to_set(labelset)
                )
                for id, labelset in id_labelset_map.items()
            }
        group_unit_id_labelsets_map[species][unit] = id_labelset_map

    # ---- human speech labelsets
    talker_dirs = sorted(
        constants.HUMAN_SPEECH_WE_CANT_SHARE.glob('Buckeye-corpus-s*')
    )

    talker_labelsets = {}
    pbar = tqdm(talker_dirs)
    for talker_dir in pbar:
        labelset = set()
        name = talker_dir.name.split('-')[-1]
        pbar.set_description(f"Counting occurences of classes for Buckeye ID: {name}")
        csv_paths = sorted(talker_dir.glob('*.csv'))
        simple_seqs = []
        for csv_path in csv_paths:
            try:
                simple_seqs.append(
                    crowsetta.formats.seq.SimpleSeq.from_file(csv_path)
                )
            except pandera.errors.SchemaError:
                continue
        for simple_seq in simple_seqs:
            labelset = labelset.union(set(simple_seq.labels))
        talker_labelsets[name] = labelset
    group_unit_id_labelsets_map["Human-Speech"] = {}   
    group_unit_id_labelsets_map["Human-Speech"]["phoneme"] = {
        talker: list(labelset) for talker, labelset in talker_labelsets.items()
    }

    return group_unit_id_labelsets_map


SCRIBE = crowsetta.Transcriber(format="simple-seq")


def set_to_map(group_unit_id_labelsets_map):
    """Convert sets of labels to maps,
    that map from labels to consecutive integers.
    
    Note that for human speech, 
    we use a fixed labelmap across all IDs.
    """
    species_id_labelmap_map = {}
    for species in group_unit_id_labelsets_map.keys():
        species_id_labelmap_map[species]= {}
        for unit in group_unit_id_labelsets_map[species].keys():
            id_labelset_map = group_unit_id_labelsets_map[species][unit]
            if species == "Human-Speech" and unit == "phoneme":
                # we want to use the same labelmap across all IDs in Buckeye corpus,
                # even though there are different per-ID labelsets
                id_labelmap_map = {
                    id: vak.common.labels.to_map(
                        # we need to convert from list back to set when loading from json
                        set(BUCKEYE_PHONES),
                        map_background=True,
                        # next line: just bein' explicit, like ya do
                        background_label=vak.common.constants.DEFAULT_BACKGROUND_LABEL,
                    )
                    for id, _ in id_labelset_map.items()
                }
            else:
                id_labelmap_map = {
                    id: vak.common.labels.to_map(
                        # we need to convert from list back to set when loading from json
                        set(labelset),
                        map_background=True,
                        # next line: just bein' explicit, like ya do
                        background_label=vak.common.constants.DEFAULT_BACKGROUND_LABEL,
                    )
                    for id, labelset in id_labelset_map.items()
                }
            species_id_labelmap_map[species][unit] = id_labelmap_map
    return species_id_labelmap_map


def make_labelsets_and_labelmaps(dry_run=True):
    """Make labelsets and labelmaps from labelsets,
    then save as json files in CMACBench dataset root.

    These have two purposes:
    1. Labelsets are used to filter files so we keep only those
    that have labels in the labelset, i.e. closed-set classification.
    2. Labelmaps are used when training models, to determine the number
    of integer output classes, or at inference time, to map integer outputs
    back to string labels.
    """
    logger.info(
        f"Making labelsets from module-level constant GROUP_UNIT_ID_LABELSTR_MAP"
    )
    group_unit_id_labelsets_map = make_labelsets()

    logger.info(
        f"Final group_unit_id_labelsets_map:\n{group_unit_id_labelsets_map}"
    )

    if not dry_run:
        with open(constants.LABELSETS_JSON, "w") as fp:
            json.dump(group_unit_id_labelsets_map, fp, indent=4)

    logger.info(
        f"Converting labelsets to labelmaps."
    )
    species_id_labelmaps_map = set_to_map(group_unit_id_labelsets_map)
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
