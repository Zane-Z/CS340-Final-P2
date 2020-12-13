import os
import utils

if __name__ == "__main__":

    f = open("../path_to_data.txt", "r")
    path = f.read()
    f.close()

    Ntr = 2307
    print(utils.flatten_csv(path, Ntr))