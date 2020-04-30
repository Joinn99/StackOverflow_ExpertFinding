import os
import sys
import argparse
from model import stackoptimizer

sys.path.append('./')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--scoring", help="The scoring method.",
        choices=['f1', 'precision', 'recall'], default='f1')
    parser.add_argument(
        "-p", "--period", help="The experiment period.",
        choices=[60, 90, 120], default=60, type=int)
    return parser


if __name__ == "__main__":
    PARSER = parse_arguments()
    ARGS = PARSER.parse_args()
    if os.path.exists("Data/StackExpert{:d}.db".format(ARGS.period)):
        SO = stackoptimizer.StackExpOptimizer(
            "Data", scoring=ARGS.scoring, period=ARGS.period)
        SO.random_process()
    else:
        print("[Error] Data not detected.")
