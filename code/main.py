import os
import utils

if __name__ == "__main__":

    f = open(r"../path_to_data.txt", "r")
    path = f.read()
    f.close()
    path = path + "train/X/X_"

    Ntr = 2307
    print(utils.flatten_csv(path, Ntr))