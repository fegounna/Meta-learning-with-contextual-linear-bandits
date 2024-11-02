"""Implementation of the Algorithm Meta-Represented Greedy Policy"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

np.random.seed(42)

with open("../data/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("../data/Y_test.pkl", "rb") as f:
    Y_test = pickle.load(f)

B_trace = np.load("../data/B_ml.npy")
B_mom = np.load("../data/B_ml_mom.npy")

total_iterations = 20


def calculate_regret(X_test, Y_test, B):
    regrets = np.zeros(total_iterations)

    for t in range(5):
        X_test_array = np.array(X_test[t])
        Y_test_array = np.array(Y_test[t])

        n, d = X_test_array.shape

        arms = set(range(n))
        arm = np.random.randint(0, n)
        arms.remove(arm)

        X_i = np.array([X_test_array[arm]])
        Y_i = np.array([Y_test_array[arm]])

        best_reward = np.max(Y_test_array)
        regrets[0] += best_reward - Y_test_array[arm]

        for i in range(1, total_iterations):
            X_iTX_i = X_i.T @ X_i
            BTX_iTX_iB = B.T @ X_iTX_i @ B
            BTX_iTY_i = B.T @ X_i.T @ Y_i

            reg = 1e-6 * np.eye(BTX_iTX_iB.shape[0])
            alpha_i = np.linalg.inv(BTX_iTX_iB + reg) @ BTX_iTY_i

            arm = max(arms, key=lambda a: np.dot(B @ alpha_i, X_test_array[a]))
            best_reward = np.max([Y_test_array[a] for a in arms])

            X_i = np.vstack([X_i, X_test_array[arm]])
            Y_i = np.append(Y_i, Y_test_array[arm])

            regrets[i] += best_reward - Y_test_array[arm]
            arms.remove(arm)
    return regrets


# Algorithm : Meta-Represented Greedy Policy
regrets_trace = calculate_regret(X_test, Y_test, B_trace)

# Algorithm : Method of Moments
regrets_mom = calculate_regret(X_test, Y_test, B_mom)

# Algorithm : Random Selection of the arms
regrets_random = np.zeros(total_iterations)

for t in range(5):
    X_test_array = np.array(X_test[t])
    Y_test_array = np.array(Y_test[t])

    n, d = X_test_array.shape
    arms = set(range(n))

    for i in range(total_iterations):
        arm = np.random.choice(list(arms))
        best_reward = np.max([Y_test_array[a] for a in arms])
        arms.remove(arm)
        regrets_random[i] += best_reward - Y_test_array[arm]


cumulative_regrets_trace = np.cumsum(regrets_trace)
cumulative_regrets_random = np.cumsum(regrets_random)
cumulative_regrets_mom = np.cumsum(regrets_mom)


plt.figure(figsize=(12, 6))
plt.plot(
    cumulative_regrets_trace / total_iterations,
    label="Trace norm regularization Method",
)
plt.plot(cumulative_regrets_mom / total_iterations, label="Method of Moments")
plt.plot(cumulative_regrets_random / total_iterations, label="Random selection")
plt.xlabel("Iteration")
plt.ylabel("Average Cumulative Regret")
plt.title("Comparison of the Meta-represented greedy policies with random selection")
plt.legend()
plt.grid(True)
plt.savefig("../results/Average Cumulative Regret of 5 test users.png")
plt.show()
