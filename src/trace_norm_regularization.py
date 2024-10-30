"""Calculate the matrix B for a given dataset then save it."""

import numpy as np

def compute_gradient(A, X, Y):
    """Compute the gradient of the our function f.
    
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
        The gradient of the our function f.
    """

    T = X.shape[0]
    n = X.shape[1]
    grad = np.zeros_like(A)

    for t in range(T):
        grad[:,t] += X[t].T @(Y[t].T- X[t]@A[:,t])
    grad *= -2/(T*n)
    return grad

def singular_value_thresholding(A, λ):
    """Apply the singular value thresholding operator.

    Parameters
    ----------
    A : array-like, shape (d, T)
        The matrix to threshold.
    λ : float
        The threshold.
    
    Returns
    -------
    A_thresh : array-like, shape (d, T)
        The thresholded matrix.
    """
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    s = np.maximum(s - λ, 0)
    return u @ np.diag(s) @ vh




def update_rule(W,X,Y,L,λ):
    """Update the matrix W using the proximal gradient descent.
    
    Parameters
    ----------
    W : array-like, shape (d, T)
        The matrix to update.
    X : array-like, shape (T, n, d)
        The context data.
    Y : array-like, shape (T, n)
        The reward data.
    L : float
        The Lipschitz constant of the gradient of f.
    λ : float
        The regularization parameter.
    
    Returns
    -------
    W_new : array-like, shape (d, T)
        The updated matrix.
    """

    C = W - 1/L * compute_gradient(W,X,Y)
    return singular_value_thresholding(C,λ/L)
    
def calculate_lipschitz_constant(X):
    """
    Calculate the Lipschitz constant L for the gradient of f.
    
    Parameters
    ----------
    X : array-like, shape (T, n, d)
        The context data.
    
    Returns
    -------
    L : float
        The Lipschitz constant.
    """
    T, n, _ = X.shape
    max_singular_value_squared = 0
    for t in range(T):
        # Compute the largest singular value of X_t
        singular_values = np.linalg.svd(X[t], compute_uv=False)
        sigma_max = np.max(singular_values)
        max_singular_value_squared = max(max_singular_value_squared, sigma_max ** 2)
    L = (2 / (n * T)) * max_singular_value_squared
    return L


def optimize_under_trace_norm_regularization(A0, X, Y, λ=0.1, n_iter=100):
    
    """Compute an estimate using Accelerated Gradient Method for Trace Norm Minimization [Ji, 2009] of the problem : min f(A) + λ||A||_*

    Parameters
    ----------
    A0 : array-like, shape (d, T)
        The initial matrix.
    X : array-like, shape (T, n, d)
        The context data.
    Y : array-like, shape (T, n)
        The reward data.
    λ : float
        The regularization parameter.
    n_iter : int
        The number of optimization iterations.

    Returns
    -------
    A : array-like, shape (d, T)
        The optimized matrix.
    """

    alpha_k = 1
    Z_k = A0
    W_k = A0 
    L = calculate_lipschitz_constant(X)
    print(L)
    for _ in range(n_iter):

        W_k_next = update_rule(Z_k,X,Y,L,λ)

        alpha_k_next = (1+np.sqrt(1+4*alpha_k**2))/2

        Z_k = W_k + (alpha_k - 1)/alpha_k_next * (W_k_next - W_k)
        alpha_k = alpha_k_next
        W_k = W_k_next
    
    return W_k

def extract_left_singular_vectors(A, threshold=0.01):
    """Extract the left singular vectors of a matrix using Singular Value Decomposition (SVD),
    retaining only those corresponding to singular values greater than a specified threshold.
    
    Parameters
    ----------
    A : array-like, shape (d, T)
        The matrix to factorize.
    threshold : float, optional
        The threshold for singular values.

    Returns
    -------
    B : array-like, shape (d, r)
        The matrix of left singular vectors of A, where r is the number of singular values 
        greater than the threshold.
    """
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    
    r = np.sum(S > threshold)
    
    B = U[:, :r]        # Shape: (d, r)

    return B

X = np.load('../data/X_ml.npy')
Y = np.load('../data/Y_ml.npy')
A0 = np.random.rand(X.shape[2], X.shape[0])
λ = 0.005
n_iter = 100

A = optimize_under_trace_norm_regularization(A0, X, Y, λ, n_iter)

B = extract_left_singular_vectors(A, threshold=0.01)

np.save('../data/B_ml.npy', B)

