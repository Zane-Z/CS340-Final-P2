import os.path
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.optimize import approx_fprime

# returns numpy ndarray x0,y0 to x9,y9 coordinates of a all X_###.csv where each csv is flattened into a single row
# pass the relative path to the directory containingn the csv's numbered X_0.csv to X_<nfiles>.csv
# throws regex separaters warning but seems to cause no issue with parsing the csv
def flatten_csv_X(path, nfiles):

    fl = np.zeros((nfiles+1,220))

    for i in range(0,nfiles + 1):
        f = open(path + str(i) + ".csv", "rb")
        # cols= ["x0","x1","x2","x3","x4","x5","x6","x7","x8","x9"]
        # cols = ["y0","y1","y2","y3","y4","y5","y6","y7","y8","y9"]
        cols = ["x0","y0","x1","y1","x2","y2","x3","y3","x4","y4","x5","y5","x6","y6","x7","y7","x8","y8","x9","y9"]
        orig = pd.read_csv(f, header=0, sep=', ', usecols=cols) 
        row = orig.to_numpy().flatten()
        fl[i] = row
        f.close()
    return fl

def flatten_csv_y(path, nfiles):
    fl = np.zeros((nfiles+1,60))
    for i in range(0,nfiles + 1):
        f = open(path + str(i) + ".csv", "rb")
        orig = pd.read_csv(f, header=0, sep=', ', usecols=["x", "y"])
        row = orig.to_numpy().flatten()
        if row.size != 60:
            filler = np.full(60-row.size, row[row.size - 1])
            row = np.append(row, filler)
        fl[i] = row
        f.close()
    return fl

# Code from assignments

def savefig(fname, verbose=True):
    plt.tight_layout()
    path = os.path.join('..', 'figs', fname)
    plt.savefig(path)
    if verbose:
        print("Figure saved as '{}'".format(path))

def dijkstra(G, i=None, j=None):
    '''Computes shortest distance between all pairs of nodes given an adjacency matrix G,
    where G[i,j]=0 implies there is no edge from i to j.

    Parameters
    ----------
    G : an N by N numpy array

    '''
    dist = scipy.sparse.csgraph.dijkstra(G, directed=False)
    if i is not None and j is not None:
        return dist[i,j]
    else:
        return dist

def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma

def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """

    # add extra dimensions so that the function still works for X and/or Xtest are 1-D arrays.
    if X.ndim == 1:
        X = X[None]
    if Xtest.ndim == 1:
        Xtest = Xtest[None]

    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)


def check_gradient(model, X, y, dimensionality, verbose=True, epsilon=1e-6):
    # This checks that the gradient implementation is correct
    w = np.random.randn(dimensionality)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0],
                                       epsilon=epsilon)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient))/np.linalg.norm(estimated_gradient) > 1e-6:
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        if verbose:
            print('User and numerical derivatives agree.')
