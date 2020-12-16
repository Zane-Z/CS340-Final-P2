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
    path_X_val = path + "val/X/X_"
    n_val = 523
    n_test = 19
    NXtr = 2307
    Xtr = utils.flatten_csv_X(pathXtr, NXtr)
    X_test = utils.flatten_csv_X(pathXtest, n_test)
    X_val = utils.flatten_csv_X(path_X_val, n_val)

    pathytr = path + "train/y/y_"
    path_y_val = path + "val/y/y_"
    Nytr = NXtr
    ytr = utils.flatten_csv_y(pathytr,Nytr)
    y_val = utils.flatten_csv_y(path_y_val, n_val)
    Y = ytr
    # binarizer = LabelBinarizer()
    # Y = binarizer.fit_transform(ytr)

    hidden_layer_sizes = [220]
    min_error = 1000000
    min_lam = 0
    min_iter = 0
    min_hidden_layers = []
    for s2 in [40, 45, 50, 55, 60, 65]:
            for lam in [0.2, 1.3]:
                for iteration in [ 1200, 1400]:
                    hidden_layer_sizes = [220]
                    hidden_layer_sizes.append(s2)
                    model = NeuralNet(hidden_layer_sizes, lammy=lam, max_iter = iteration)

                    t = time.time()
                    model.fit(Xtr,Y)
                    # model.fit_sgd(X,Y)
                    print("Fitting took %d seconds" % (time.time()-t))

                    # Comput training error
                    yhat = model.predict(Xtr)
                    trainError = np.sqrt(mean_squared_error(ytr.flatten(),yhat.flatten()))

                    # Compute validation error
                    yhat_val = model.predict(X_val)
                    valError = np.sqrt(mean_squared_error(y_val.flatten(),yhat_val.flatten()))

                    print("Training error = ", trainError)
                    print("Validation Error = ", valError)
                    print(f"Lambda: {lam}, iterations: {iteration}")
                    
                    if (valError < min_error):
                        min_error = valError
                        min_lam = lam
                        min_iter = iteration
                        min_hidden_layers = hidden_layer_sizes
                    print(f"Current min Error: {min_error} with lambda: {min_lam} and iterations: {min_iter} hidden layers {min_hidden_layers}")
                    print("========================================= \n")

                    
                    # Compute testing error
                    y_test = model.predict(X_test)

                    print(y_test.shape)
                    y_test = y_test.flatten()
                    print(y_test.shape)
                    y_test = y_test.T
                    print(y_test.shape)
                    np.savetxt(f"y_test_lam_{lam}_iter_{iteration}_layers_{hidden_layer_sizes[0]}-{hidden_layer_sizes[1]}.csv", y_test, delimiter=",")
                    
            print("\n\n\n =====================================================")
            print(f"Final min Error: {min_error} with lambda: {min_lam} and iterations: {min_iter} layers {min_hidden_layers}")
