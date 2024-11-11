"""Implementation of method of moments [Nilesh Tripuraneni, Chi Jin, and Michael Jordan, 2021] Provable meta-learning of linear representations"""

import numpy as np


def estimate_B_using_mom(X, Y, r):
    """
    Estimate the matrix B using the method of moments.

    Parameters
    ----------
    X : array-like, shape (T, n, d)
        The context data.
    Y : array-like, shape (T, n)
        The reward data.
    r : int
        The number of singular values to retain.

    Returns
    -------
    B : array-like, shape (d, r)
        The estimation of the matrix B.
    """
    T, n, d = X.shape

    result_matrix = np.zeros((d, d))

    for t in range(T):
        for i in range(n):
            y_squared = Y[t, i] ** 2
            x_i = X[t, i].reshape(d, 1)
            result_matrix += y_squared * (x_i @ x_i.T)

    result_matrix /= n * T

    U, _, _ = np.linalg.svd(result_matrix, full_matrices=False)
    B = U[:, :r]
    return B
