"""Implementation of the Algorithm Meta-Represented Greedy Policy"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

np.random.seed(42)

with open('../data/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('../data/Y_test.pkl', 'rb') as f:
    Y_test = pickle.load(f)

B = np.load('../data/B_ml.npy')

# User t
t = 4

X_test_array = np.array(X_test[t]) 
Y_test_array = np.array(Y_test[t])  

n, d = X_test_array.shape


# Algorithm : Meta-Represented Greedy Policy
arms = set(range(n))
arm = np.random.randint(0, n)
arms.remove(arm)

X_i = np.array([X_test_array[arm]])  
Y_i = np.array([Y_test_array[arm]]) 

for i in range(1, min(n//2,30)):
    X_iTX_i = X_i.T @ X_i  
    BTX_iTX_iB = B.T @ X_iTX_i @ B  
    BTX_iTY_i = B.T @ X_i.T @ Y_i  

    reg = 1e-6 * np.eye(BTX_iTX_iB.shape[0])
    alpha_i = np.linalg.inv(BTX_iTX_iB + reg) @ BTX_iTY_i  

    arm = max(arms, key=lambda a: np.dot(B @ alpha_i, X_test_array[a]))

    X_i = np.vstack([X_i, X_test_array[arm]]) 
    Y_i = np.append(Y_i, Y_test_array[arm])  

    arms.remove(arm)


# Algorithm : Random Selection of the arms

arms = set(range(n))
arm = np.random.randint(0, n)
arms.remove(arm)

Y_i_random = np.array([Y_test_array[arm]]) 

for i in range(1,min(n//2,30)):
    arm = np.random.choice(list(arms))
    arms.remove(arm)
    Y_i_random = np.append(Y_i_random, Y_test_array[arm])



fig, axs = plt.subplots(2, 1, figsize=(10, 12))

axs[0].plot(Y_i, marker='o', linestyle='', color='b')
axs[0].set_title(f"Reward of Y_{t} with the meta-represented greedy policy")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Rating")
axs[0].grid(True)

axs[1].plot(Y_i_random, marker='o', linestyle='', color='b')
axs[1].set_title(f"Reward of Y_{t} with the random selection of the arms")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Rating")
axs[1].grid(True)

plt.savefig(f'../results/user{t}')
plt.tight_layout()
plt.show()

