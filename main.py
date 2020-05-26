import os
import sys
from model.stackoptimizer import StackExpOptimizer
from model.analysis import find_expert, best_solution, performance
from model.utils import get_tags

sys.path.append('./')

INFO = """
    \b\b\b\b{line}\n\n StackOverflow Expert Prediction\n\n{line}
    Options:
        [0] Exit
        [1] Run grid search
        [2] Find experts in one tag
        [3] Show best parameters in specific tag
        [4] Show model performance
    Input(0-4):
"""

if __name__ == "__main__":
    TAGS = get_tags()
    TAGS_TEXT = ["\t[{:d}] {:s}\n".format(ind, tag) for ind, tag in enumerate(TAGS)]
    while True:
        os.system('cls||clear')
        OPTION = input(INFO.format(**{"line": "*" * 50}))
        if OPTION == "1":
            StackExpOptimizer().random_process()
        elif OPTION == "2":
            TAG = input("\tSelect tag:\n" + "".join(TAGS_TEXT))
            find_expert(TAGS[int(TAG)])
            input("Press 'Enter' to coutinue...")
        elif OPTION == "3":
            TAG = input("\tSelect tag:\n" + "".join(TAGS_TEXT))
            print("{st}Best Parameters{st}".format(**{"st": "*"*15}))
            print(best_solution(TAGS[int(TAG)]).squeeze().to_string())
            input("Press 'Enter' to coutinue...")
        elif OPTION == "4":
            performance()
            input("Press 'Enter' to coutinue...")
        elif OPTION == "0":
            break
