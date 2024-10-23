import numpy as np


trace_norm = np.linalg.norm(X, ord='nuc')




def compute_gradient(A, X, Y):
    """Compute the gradient of the first part of the objective function.
    
    Parameters
    ----------
    A : array-like, shape (d, T)
        The matrix to optimize.
    X : array-like, shape (T, n, d)
        The context data.
    Y : array-like, shape (T, n)
        The reward data.
    
    Returns
    -------
    grad : array-like, shape (d, T)
        The gradient of the first part of the objective function.
    """

    T = X.shape[0]
    n = X.shape[1]
    grad = np.zeros_like(A)

    for t in range(T):
        grad[:,t] += np.transpose(X[t]) @(np.transpose(Y[t]) - X[t]@A[:,t])
    grad *= -2/(T*n)
    return grad


def optimize_under_trace_norm_regularization():
    
    """Compute an estimate using proximal gradient descent of the problem : min ||X - A||_F^2 + λ||A||_*

    Parameters
    ----------


    Returns
    -------

    """
