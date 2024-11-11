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


X = np.load("../data/X_ml_80test.npy")
Y = np.load("../data/Y_ml_80test.npy")


B = estimate_B_using_mom(X, Y, 2)
np.save("../data/B_ml_mom_80test.npy", B)

"""
T = X.shape[0]
split_point = int(T * 0.8)
X_train = X[:split_point]
X_val = X[split_point:]
Y_train = Y[:split_point]
Y_val = Y[split_point:]

max_r = min(X_train.shape[0], X_train.shape[2])
r_values = range(1, max_r)

best_regret = float('inf')
best_r = None
best_B = None

for r in r_values:
    B = estimate_B_using_mom(X_train, Y_train, r)

    current_regret = calculate_regret(X_val, Y_val, B,20)[-1]
    print(f"r={r}: Regret={current_regret}")
    
    if current_regret < best_regret:
        best_regret = current_regret
        best_r = r
        best_B = B
"""
