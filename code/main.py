import os
import utils
import time
from neural_net import NeuralNet

# from sklearn.preprocessing import LabelBinarizer
import numpy as np

if __name__ == "__main__":

    f = open(r"../path_to_data.txt", "r")
    path = f.read()
    f.close()

    pathXtr = path + "train/X/X_"
    NXtr = 100
    Xtr = utils.flatten_csv_X(pathXtr, NXtr)

    pathytr = path + "train/y/y_"
    Nytr = NXtr
    ytr = utils.flatten_csv_y(pathytr,Nytr)
    hidden_layer_sizes = [220]
    model = NeuralNet(hidden_layer_sizes)

    t = time.time()
    model.fit(Xtr,ytr)
    print("Fitting took %d seconds" % (time.time()-t))

    # Comput training error
    yhat = model.predict(Xtr)

    # print("ytr", ytr)
    # print ("yhat", yhat)
    # print (yhat.shape)

    print("Comparing the first 5 x and y coords:")
    for i in range(5):
        print("real x and y:", ytr[0][i*2], ytr[0][i*2 + 1])
        print("pred x and y:", yhat[0][i*2], yhat[0][i*2 + 1])

    sse = np.sum((yhat - ytr)**2)
    print ("Sum of squared error", sse)