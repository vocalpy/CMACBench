"""CLI parser for prep_biosoundsegbench script"""
import argparse

from . import constants


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=constants.PREP_STAGES,
        default="all",
        help=f"Stage of prep to run. Stages are: {constants.PREP_STAGES}"
    )
    parser.add_argument(
        "--biosound-groups",
        choices=constants.BIOSOUND_GROUPS,
        nargs="+",
        default=constants.BIOSOUND_GROUPS,
        help=(f"Space-separated list of which group(s) of biosound to prep. "\
        f"Groups are: {constants.BIOSOUND_GROUPS}. Default is all of them.")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If this option is set, do a 'dry run' without copying or generating files, just to test code"
    )
    return parser
