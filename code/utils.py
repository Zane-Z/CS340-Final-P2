import os
import numpy as np
import pandas as pd

# returns numpy ndarray x0,y0 to x9,y9 coordinates of a all X_###.csv where each csv is flattened into a single row
# pass the relative path to the directory containingn the csv's numbered X_0.csv to X_<nfiles>.csv
# throws regex separaters warning but seems to cause no issue with parsing the csv
def flatten_csv(path, nfiles):

    # flatten each train/X_###
    fl = np.zeros((nfiles+1,220))

    for i in range(0,nfiles + 1):
        f = open(path + "train/X/X_" + str(i) + ".csv", "rb")
        orig = pd.read_csv(f, header=0, sep=', ', usecols=["x0","y0","x1","y1","x2","y2","x3","y3","x4","y4","x5","y5","x6","y6","x7","y7","x8","y8","x9","y9"])
        row = orig.to_numpy().flatten()
        fl = row
        f.close()
    return fl