import os
import sys
from model import stackoptimizer

sys.path.append('./')


if __name__ == "__main__":
    if os.path.exists("Data/StackExpert.db"):
        SO = stackoptimizer.StackExpOptimizer("Data")
        SO.random_process()
    else:
        print("[Error] Data not detected.")
