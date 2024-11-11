import numpy as np
from method_of_moments import estimate_B_using_mom
from trace_norm_regularization import optimize_under_trace_norm_regularization, extract_left_singular_vectors
import pickle as pkl

# Fixer la graine pour la reproductibilité
np.random.seed(0)

# Paramètres
T = 100  # Nombre de colonnes de W
d = 60   # Nombre de lignes de W
r = 5    # Rang de W
N = 100  # Nombre d'échantillons

# Générer W de dimension (d, T) de rang r
# En créant une matrice aléatoire de rang r via multiplication de deux matrices
W_left = np.random.randn(d, r)
W_right = np.random.randn(r, T)
W = np.dot(W_left, W_right)

# Décomposer W avec SVD pour obtenir les r premiers vecteurs singuliers de gauche
U, S, Vt = np.linalg.svd(W, full_matrices=False)
B = U[:, :r]  # B de dimension (d, r) avec les r premiers vecteurs singuliers de gauche

# Générer les données gaussiennes x_t_n de dimension (T, N, d)
X = np.random.randn(T, N, d)  # T x N échantillons gaussiens de dimension d

# Générer y_t_n = W_t . X + eta_t_n
Y = np.zeros((T, N))  # Initialisation de Y de dimension (T, N)
for t in range(T):
    W_t = W[:, t]  # La t-ème colonne de W
    eta = np.random.randn(N)  # Bruit gaussien centré réduit
    # Calcul de Y pour chaque échantillon n à temps t
    Y[t] = X[t] @ W_t + eta

B_mom = estimate_B_using_mom(X, Y, r)
A0 = np.random.rand(X.shape[2], X.shape[0])
A = optimize_under_trace_norm_regularization(A0, X, Y, λ=0.01, n_iter=100)
B_trace_norm = extract_left_singular_vectors(A, threshold=0.001)

T_test, N_test = 10, 50
X_test = np.random.randn(T_test, N_test, d)
Y_test = np.zeros((T_test, N_test))
for t in range(T_test):
    alpha_test = np.random.multivariate_normal(mean=np.zeros(r), cov=(1/r) * np.eye(r))
    W_t_test = B @ alpha_test
    eta = np.random.randn(N_test)
    Y_test[t] = X_test[t] @ W_t + eta

np.save("./data/B_oracle.npy", B)
np.save("./data/B_mom_synthetique.npy", B_mom)
np.save("./data/B_trace_norm_synthetique.npy", B_trace_norm)
with open('./data/Y_test_synthetique.pkl', 'wb') as f:
    pkl.dump(Y_test, f)
with open('./data/X_test_synthetique.pkl', 'wb') as f:
    pkl.dump(X_test, f)