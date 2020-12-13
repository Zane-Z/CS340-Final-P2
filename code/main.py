import os
import utils
import time
from neural_net import NeuralNet

from sklearn.metrics import mean_squared_error

# from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd

if __name__ == "__main__":

    f = open(r"../path_to_data.txt", "r")
    path = f.read()
    f.close()

    pathXtr = path + "train/X/X_"
    NXtr = 2307
    Xtr = utils.flatten_csv_X(pathXtr, NXtr)

    pathytr = path + "train/y/y_"
    Nytr = NXtr
    ytr = utils.flatten_csv_y(pathytr,Nytr)
    Y = ytr
    # binarizer = LabelBinarizer()
    # Y = binarizer.fit_transform(ytr)

    hidden_layer_sizes = [220]
    model = NeuralNet(hidden_layer_sizes)

    t = time.time()
    model.fit(Xtr,Y)
    # model.fit_sgd(X,Y)
    print("Fitting took %d seconds" % (time.time()-t))

    # Comput training error

    # yhat = model.predict(Xtr)
    # trainError = np.sqrt(mean_squared_error(ytr.flatten(),yhat.flatten()))
    # print("Training error = ", trainError)

    # NXval = 523
    # pathXval = path + "val/X/X_"
    # Xval = utils.flatten_csv_X(pathXval, NXval)

    # yhat_val = model.predict(Xval)

    # pathyval = path + "val/y/y_"
    # yval = utils.flatten_csv_y(pathyval,NXval)

    # valError = np.sqrt(mean_squared_error(yval.flatten(),yhat_val.flatten()))
    # print("Validation error = ", valError)

    NXtest = 19
    pathXtest = path + "test/X"
    Xtest = utils.flatten_csv_X(pathXtest, NXtest)
    ypred = model.predict(Xtest)
    fl_ytest = ypred.flatten()
    labels = [""]*NXtest
    ids = [""]*NXtest*60
    for i in range(NXtest + 1):
        for j in range(1,31):
            idx = str(i) + "_x_" + str(j)
            idy = str(i) + "_y_" + str(j)
    