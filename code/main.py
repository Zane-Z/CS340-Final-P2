import os
import utils
import numpy as np

if __name__ == "__main__":

    f = open(r"../path_to_data.txt", "r")
    path = f.read()
    f.close()
    pathXtr = path + "train/X/X_"

    NXtr = 2307
    print(utils.flatten_csv(pathXtr, NXtr)[0])