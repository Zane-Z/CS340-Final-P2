import os
import utils
import time
from neural_net import NeuralNet

from sklearn.metrics import mean_squared_error

# from sklearn.preprocessing import LabelBinarizer
import numpy as np

if __name__ == "__main__":

    f = open(r"../path_to_data.txt", "r")
    path = f.read()
    f.close()

    pathXtr = path + "train/X/X_"
    pathXtest = path + "test/X/X_"
    n_test = 19
    NXtr = 2307
    Xtr = utils.flatten_csv_X(pathXtr, NXtr)
    X_test = utils.flatten_csv_X(pathXtest, n_test)

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
    yhat = model.predict(Xtr)
    trainError = np.sqrt(mean_squared_error(ytr.flatten(),yhat.flatten()))
    print("Training error = ", trainError)

    # Compute testing error
    y_test = model.predict(X_test)

    print(y_test.shape)
    y_test = y_test.flatten()
    print(y_test.shape)
    y_test = y_test.T
    print(y_test.shape)
    np.savetxt("y_test.csv", y_test, delimiter=",")