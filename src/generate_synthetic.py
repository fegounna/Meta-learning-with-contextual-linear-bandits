"""Module to generate synthetic data."""

import numpy as np
import pickle as pkl

np.random.seed(42)

T = 100
d = 60
r = 5
N = 100

W_left = np.random.randn(d, r)
W_right = np.random.randn(r, T)
W = np.dot(W_left, W_right)

U, S, Vt = np.linalg.svd(W, full_matrices=False)
B = U[:, :r]
X = np.random.randn(T, N, d)

Y = np.zeros((T, N))
for t in range(T):
    W_t = W[:, t]
    eta = np.random.randn(N)
    Y[t] = X[t] @ W_t + eta

T_test, N_test = 10, 50
X_test = np.random.randn(T_test, N_test, d)
Y_test = np.zeros((T_test, N_test))
for t in range(T_test):
    alpha_test = np.random.multivariate_normal(
        mean=np.zeros(r), cov=(1 / r) * np.eye(r)
    )
    W_t_test = B @ alpha_test
    eta = np.random.randn(N_test)
    Y_test[t] = X_test[t] @ W_t + eta

np.save("./data/X_synthetic.npy", X)
np.save("./data/Y_synthetic.npy", Y)
np.save("./data/B_oracle.npy", B)
with open("./data/Y_test_synthetique.pkl", "wb") as f:
    pkl.dump(Y_test, f)
with open("./data/X_test_synthetique.pkl", "wb") as f:
    pkl.dump(X_test, f)
