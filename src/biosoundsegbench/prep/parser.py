"""CLI parser for prep_biosoundsegbench script"""
import argparse

from . import constants


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=constants.PREP_STAGES,
        default="all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If this option is set, do a 'dry run', just to test code"
    )
    return parser
