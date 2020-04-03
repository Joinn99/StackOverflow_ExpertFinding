import sys
from model import stackoptimizer

sys.path.append('./')


if __name__ == "__main__":
    SO = stackoptimizer.StackExpOptimizer("Data")
    SO.random_process()
